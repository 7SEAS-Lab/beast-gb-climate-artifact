import warnings

warnings.simplefilter("ignore", UserWarning)

import os
import time

import mlflow
import torch
import torch.distributed as dist
from beast.comm import (
    execute_in_waves,
)
from beast.layers.transformer import USE_TE_DPA

# beast imports
from beast.utils import set_all_seeds
from jigsaw.parallel import clip_grad_norm_

from gb.checkpoint import (
    save_checkpoint,
)
from gb.metrics import df_pm, mse_loss_fn
from gb.mlflow_utils import get_logging_functions, setup_mlflow_tracking
from gb.utils import (
    count_flops,
)

log_metric = None
log_metrics = None

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

from setup import setup_model, setup_paths

USE_TE_FP8 = (
    USE_TE_DPA
    and torch.version.hip is None
    and torch.cuda.is_available()  # is cuda, not rocm
    and os.getenv("BEAST_USE_TE_FP8", "1") in ["1", "True"]  # check env var for fp8
    and torch.cuda.get_device_capability() >= (8, 9)  # check if hardware supports fp8
)


def has_nan_grad(model):
    return any(
        p.grad is not None and not torch.isfinite(p.grad).all()
        for p in model.parameters()
    )


def training_loop(config_dict, measure_flops: bool = True, use_mlflow: bool = False):
    (
        model,
        optimizer,
        train_dataloader,
        valid_dataloader,
        loss_fn,
        up_group,
        dtp_group,
        channel_group,
        device,
        n_samples,
        bayesian,
        start_epoch,
        start_step,
        sqrt_spatial_weights,
        scheduler,
        autocast,
        feature_weights,
    ) = setup_model(config_dict, mode="training")

    global log_metric, log_metrics
    if use_mlflow:
        run_id = setup_mlflow_tracking(config_dict)
        log_metric, log_metrics = get_logging_functions(run_id)
    else:
        run_id = None

        def log_metric(key, value, step=None):
            pass

        def log_metrics(metrics, step=None):
            pass

    config_dict["run_id"] = run_id

    if dist.get_rank(group=None) == 0:
        print(f"Run id: {run_id}")

    _, _, checkpoint_path = setup_paths(config_dict)

    torch.cuda.empty_cache()
    set_all_seeds(17 + up_group.rank())

    # Per-step forward, backward, and total flops
    # These numbers are already summed across all ranks (not just the dtp group)

    if measure_flops:
        if dist.get_rank(group=None) == 0:
            print("Counting FLOP")
        forward_flops, backward_flops, total_flops = count_flops(
            model,
            train_dataloader,
            loss_fn,
            device,
            dtp_group,
            n_samples,
            bayesian,
        )

        if dist.get_rank(group=None) == 0 and use_mlflow:
            mlflow.log_params(
                {
                    "forward_flop": forward_flops,
                    "backward_flop": backward_flops,
                    "total_flop": total_flops,
                },
                synchronous=True,
            )

    if dist.get_rank(group=None) == 0 and use_mlflow:
        mlflow.log_params(
            {"len_train_dataloader": len(train_dataloader)}, synchronous=True
        )

    ### Run Training
    if dist.get_rank(group=None) == 0:
        print("Starting Epochs...")

    for epoch in range(start_epoch, config_dict["training"]["n_epochs"]):
        start_time = time.perf_counter()
        avg_loss = train_one_epoch(
            config_dict=config_dict,
            epoch=epoch,
            start_step=start_step if epoch == start_epoch else 0,  #
            dataloader=train_dataloader,
            spatial_weights=sqrt_spatial_weights,
            model=model,
            bayesian=bayesian,
            n_samples=n_samples,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            dtp_group=dtp_group,
            channel_group=channel_group,
            autocast=autocast,
            checkpoint_path=checkpoint_path,
            feature_weights=feature_weights,
        )

        # Save checkpoint after validation to have the validation loss in the checkpoint
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch,
            step=None,
            loss=avg_loss,
            run_id=run_id,
            log_to_mlflow=use_mlflow,
        )
        end_time = time.perf_counter()

        avg_tflops = None
        if dist.get_rank(group=None) == 0:
            flop_str = ""
            if measure_flops:
                avg_tflops = (
                    total_flops * len(train_dataloader) / (end_time - start_time) / 1e12
                )
                flop_str = f", average TFLOPs = {avg_tflops}"
            print(
                f"Epoch {epoch} average loss {avg_loss}, time [min]: {(end_time - start_time) / 60.0}"
                + flop_str
            )

        e_metrics = {
            "train:avg_loss": avg_loss if avg_loss else -1,
            "train:avg_tflops": avg_tflops if avg_tflops else -1,
        }
        log_metrics(e_metrics, step=epoch)

        if not config_dict["skip_validation"]:
            model.eval()
            with torch.no_grad():
                eval_loss_sum = torch.tensor([0.0], device=device)
                mse_sum = torch.tensor([0.0], device=device)
                mse_fn = mse_loss_fn(dtp_group)

                for x, y, _, _ in valid_dataloader:
                    x = x.to(device)
                    y = y.to(device)

                    if bayesian:
                        pred = model(x, samples=n_samples)
                        mse_sum += mse_fn(
                            pred.mean(dim=(0)) * feature_weights, y * feature_weights
                        )
                    else:
                        pred = model(x)
                    eval_loss_sum += loss_fn(
                        pred * sqrt_spatial_weights, y * sqrt_spatial_weights
                    )

                # The UP loss is this rank's share; the MSE losses are global.
                if bayesian:
                    dist.all_reduce(eval_loss_sum, group=dtp_group)
                avg_eval_loss = eval_loss_sum / max(1, len(valid_dataloader))

                if bayesian:
                    avg_df, avg_pm = df_pm(loss_fn)
                    avg_mse = mse_sum / max(1, len(valid_dataloader))

                    if dist.get_rank(group=None) == 0:
                        print(
                            f"E{epoch} - Avg. valid loss: {avg_eval_loss}, Avg. MSE: {avg_mse}, Data fitting: {avg_df}, Prior Matching: {avg_pm}"
                        )
                        log_metrics(
                            {
                                "eval:avg_loss": avg_eval_loss,
                                "eval:avg_mse": avg_mse,
                                "eval:data_fitting": avg_df,
                                "eval:prior_matching": avg_pm,
                            },
                            step=epoch,
                        )
                else:
                    if dist.get_rank(group=None) == 0:
                        print(f"E{epoch} - Avg. valid loss: {avg_eval_loss}")

                        log_metric("eval:avg_loss", avg_eval_loss, step=epoch)


def train_one_epoch(
    *,
    # Config
    config_dict,
    # Data
    dataloader,
    # Model
    model,
    bayesian,
    n_samples,
    # Loss and Optimizer
    spatial_weights,
    optimizer,
    loss_fn,
    feature_weights,
    # Training
    epoch,
    start_step,
    device,
    autocast,
    # Comm,
    dtp_group,
    channel_group,
    # Checkpoints
    checkpoint_path,
    # Optional
    lr_scheduler=None,
):

    model.train()
    loss_sum = torch.tensor([0.0], device=device, requires_grad=False)
    if lr_scheduler is not None:
        lr_scheduler.step(epoch)
        optim_lrs = {
            f"train:param_lr:{i}": param_group["lr"]
            for i, param_group in enumerate(optimizer.param_groups)
        }
        log_metrics(optim_lrs, step=epoch)

    start_lr = 1e-6
    log_interval = config_dict["training"]["log_interval"]
    dsync_interval = config_dict["training"]["dsync_interval"]
    cp_interval = config_dict["training"]["checkpoint_interval"]

    mse_fn = mse_loss_fn(dtp_group)

    if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(epoch)
        # If we are restarting a run, start_step will be > 0.
        # Else, every other epoch should start from 0.
        dataloader.sampler.resume_idx = start_step * dataloader.batch_size

    data_access_rank = int(dist.get_rank(group=None) // 8)
    # Steps left in this epoch; the sampler restores the full length once iterated.
    dataloader_size = len(dataloader)
    dl_iter = iter(dataloader)

    start_fwd = torch.cuda.Event(enable_timing=True)
    end_fwd = torch.cuda.Event(enable_timing=True)

    start_loss = torch.cuda.Event(enable_timing=True)
    end_loss = torch.cuda.Event(enable_timing=True)

    start_bwd = torch.cuda.Event(enable_timing=True)
    end_bwd = torch.cuda.Event(enable_timing=True)

    start_opt = torch.cuda.Event(enable_timing=True)
    end_opt = torch.cuda.Event(enable_timing=True)

    cum_batch_time_s = 0
    real_time_start = time.perf_counter_ns()

    # Transformer Engine

    fp_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
    recipe = DelayedScaling(
        fp8_format=fp_format,
        amax_history_len=16,
        amax_compute_algo="max",
        # using True here required that head_dim % 64 == 0 for ALL TE attention layers,
        # or otherwise the code will error out.
        fp8_dpa=False,
        # leave it on for now and see:
        fp8_mha=False,
    )

    for i in range(start_step, len(dataloader) + start_step):
        if (epoch == 0) and (i < 10000) and (lr_scheduler is not None):
            max_values = lr_scheduler.get_last_lr()
            for index, g in enumerate(optimizer.param_groups):
                warm = i * (max_values[index] - start_lr) / 10000 + start_lr
                g["lr"] = warm

        # Desync on first iteration
        if i % dsync_interval == 0:
            x, y, dl_start, dl_end = execute_in_waves(lambda: next(dl_iter))
        else:
            x, y, dl_start, dl_end = next(dl_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        gpu_time = time.perf_counter_ns()

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast):
            with te.autocast(enabled=USE_TE_FP8, recipe=recipe):
                start_fwd.record()
                if bayesian:
                    pred = model(x, samples=n_samples)
                else:
                    pred = model(x)
                end_fwd.record()

                # pred = pred[:,]
                start_loss.record()
                loss = loss_fn(pred * spatial_weights, y * spatial_weights)
                end_loss.record()

        # Backpropagation
        start_bwd.record()
        loss.backward()
        end_bwd.record()

        if not bayesian:
            if dtp_group.size() > 1:
                clipped_norm = clip_grad_norm_(model.parameters(), 0.5, dtp_group)
            else:
                clipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        start_opt.record()
        if has_nan_grad(model):
            print(f"R{dist.get_rank()}: NaN/inf grad detected — skipping step")
            optimizer.zero_grad()
        else:
            optimizer.step()
        end_opt.record()

        loss_sum = loss_sum + loss.detach()

        torch.cuda.synchronize()
        cum_batch_time_s += start_fwd.elapsed_time(end_opt) / 1e3

        if i > 0 and i % cp_interval == 0:
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                i,
                loss_sum,
                run_id=config_dict["run_id"],
            )

        if i > 0 and i % log_interval == 0:
            # The UP loss is this rank's share; the MSE losses are global.
            if bayesian:
                dist.all_reduce(loss, group=dtp_group)
                mse = mse_fn(
                    pred.detach().mean(dim=0) * feature_weights, y * feature_weights
                ).item()
            if dist.get_rank(group=None) == 0:
                if bayesian:
                    avg_df, avg_pm = df_pm(loss_fn)
                    log_metrics(
                        {
                            "train:loss": loss if not torch.isnan(loss) else -1,
                            "train:fbo_duration": (start_fwd.elapsed_time(end_opt))
                            / 1e3,
                            "train:fwd_duration": (start_fwd.elapsed_time(end_fwd))
                            / 1e3,
                            "train:bwd_duration": (start_bwd.elapsed_time(end_bwd))
                            / 1e3,
                            "train:opt_duration": (start_opt.elapsed_time(end_opt))
                            / 1e3,
                            "train:dl_duration": (dl_end - dl_start) / 1e9,
                            "train:h2d_duration": (gpu_time - dl_end) / 1e9,
                            "train:mse": mse,
                            "train:data_fitting": avg_df,
                            "train:prior_matching": avg_pm,
                            "train:cum_batch_time": cum_batch_time_s,
                        },
                        step=epoch * (dataloader_size + start_step) + i,
                    )

                    real_time_end = time.perf_counter_ns()
                    time_with_overhead = (real_time_end - real_time_start) / 1e9
                    print(
                        f"E{epoch}-S{i}/{len(dataloader) + start_step} - loss: {float(loss.detach())}, mse: {mse}. data fitting {avg_df}, prior matching {avg_pm}, Average time per batch {cum_batch_time_s / log_interval} s, Real time:  {time_with_overhead / log_interval} s"
                    )

                else:
                    log_metrics(
                        {
                            "train:loss": loss if not torch.isnan(loss) else -1,
                            "train:fbo_duration": (start_fwd.elapsed_time(end_opt))
                            / 1e3,
                            "train:fwd_duration": (start_fwd.elapsed_time(end_fwd))
                            / 1e3,
                            "train:bwd_duration": (start_bwd.elapsed_time(end_bwd))
                            / 1e3,
                            "train:opt_duration": (start_opt.elapsed_time(end_opt))
                            / 1e3,
                            "train:dl_duration": (dl_end - dl_start) / 1e9,
                            "train:h2d_duration": (gpu_time - dl_end) / 1e9,
                            "train:clipped_norm": clipped_norm,
                            "train:cum_batch_time": cum_batch_time_s,
                        },
                        step=epoch * (dataloader_size + start_step) + i,
                    )
                    real_time_end = time.perf_counter_ns()
                    time_with_overhead = (real_time_end - real_time_start) / 1e9
                    print(
                        f"E{epoch}-S{i}/{len(dataloader) + start_step} - loss: {float(loss.detach())}, Average time per batch {cum_batch_time_s / log_interval} s, Real time: {time_with_overhead / log_interval} s, Clipped norm {clipped_norm}"
                    )

            time.sleep(data_access_rank * 0.001)
            cum_batch_time_s = 0
            real_time_start = time.perf_counter_ns()

    if bayesian:
        dist.all_reduce(loss_sum, group=dtp_group)
    average_loss = loss_sum.item() / max(1, dataloader_size)
    return average_loss
