import os
import pickle
import time

import torch
import torch.distributed as dist
from beast.comm import (
    execute_in_waves,
    get_process_group,
)

# beast imports
from beast.evaluation import (
    crps,
    gather_along_dimension,
    latitude_weighted_average,
    make_variable_names,
    mse,
    rmse,
    spread_skill_ratio,
)
from beast.utils import set_all_seeds

from gb.globals import SST_NORMALIZED_ZERO
from gb.metrics import df_pm

# from evaluation import validate, validate_deterministic
# Import model
from setup import setup_model

# Name of the single pickle file used to make the evaluation loop resumable.
# It always lives in the run's root directory and is overwritten in place.
EVAL_CHECKPOINT_NAME = "eval_checkpoint.pkl"

DEFAULT_SURFACE_VARIABLES = ["Pmsl", "U10", "V10", "T2m", "sst", "tp"]
DEFAULT_PRESSURE_VARIABLES = ["Z", "Q", "T", "U", "V", "W"]
DEFAULT_PRESSURE_LEVELS = [
    1000,
    925,
    850,
    700,
    600,
    500,
    400,
    300,
    250,
    200,
    150,
    100,
    50,
]


def mean_var_agg(samples, up_process_group):
    up_num = up_process_group.size()
    total_samples = samples.shape[0] * up_num

    # Pass 1: exact global mean. One field, blocking -- we need the result
    # before we can form deviations.
    s1 = samples.sum(dim=0)
    dist.all_reduce(s1, op=dist.ReduceOp.SUM, group=up_process_group)
    mean = s1 / total_samples

    # Pass 2: sum of squared deviations about the shared mean. Because the
    # deviations are centred, there is no cancellation between two large
    # nearly-equal quantities the way there is in E[x^2] - E[x]^2.
    dev = samples - mean
    m2 = (dev * dev).sum(dim=0)
    dist.all_reduce(
        m2,
        op=dist.ReduceOp.SUM,
        group=up_process_group,
        async_op=False,
    )

    var = m2 / (total_samples - 1)
    var = var.clamp_min(1e-10)

    return mean, var


def _eval_checkpoint_path(config_dict) -> str:
    """Return the path of the evaluation checkpoint file.

    A user-provided path (``config_dict["eval_checkpoint"]``) always takes
    precedence. Otherwise the checkpoint defaults to ``eval_checkpoint.pkl`` in
    the run's root directory (``config_dict["resume_path"]``).

    Parameters
    ----------
    config_dict : dict
        The configuration dictionary.

    Returns
    -------
    str
        Path to the evaluation checkpoint file.
    """
    user_path = config_dict.get("eval_checkpoint")
    if user_path:
        return str(user_path)
    return os.path.join(str(config_dict["resume_path"]), EVAL_CHECKPOINT_NAME)


def save_eval_checkpoint(
    checkpoint_file: str,
    *,
    next_step: int,
    total_batches: int,
    metric_dict: dict,
    loss_sum: torch.Tensor,
    epoch: int,
) -> None:
    """Persist the evaluation state to a single pickle file (overwriting it).

    Only ``metric_dict`` (moved to CPU) together with the minimal bookkeeping
    required to resume the loop is stored. The write is atomic: the state is
    written to a temporary file that is then renamed over the target, so an
    interrupted write cannot corrupt an existing checkpoint.

    Parameters
    ----------
    checkpoint_file : str
        Absolute path of the pickle file to write.
    next_step : int
        Index of the *next* batch to process, i.e. the sampler ``resume_idx``
        to use on resume.
    total_batches : int
        Total number of batches of the full (non-resumed) dataloader. Stored so
        that metric normalization stays consistent across resumes.
    metric_dict : dict
        The accumulated per-lead-time metrics to persist.
    loss_sum : torch.Tensor
        The accumulated (normalized) loss sum.
    epoch : int
        The epoch this evaluation belongs to.
    """
    state = {
        "next_step": int(next_step),
        "total_batches": int(total_batches),
        "epoch": int(epoch),
        "loss_sum": loss_sum.detach().cpu(),
        "metric_dict": {
            dt_i: {name: tensor.detach().cpu() for name, tensor in metrics.items()}
            for dt_i, metrics in metric_dict.items()
        },
    }

    tmp_file = f"{checkpoint_file}.tmp"
    with open(tmp_file, "wb") as fh:
        pickle.dump(state, fh)
    os.replace(tmp_file, checkpoint_file)


def load_eval_checkpoint(
    checkpoint_file: str,
    device: torch.device,
) -> dict | None:
    """Load the evaluation checkpoint if it exists.

    Parameters
    ----------
    checkpoint_file : str
        Absolute path of the pickle file to load.
    device : torch.device
        Device onto which the restored tensors are moved.

    Returns
    -------
    dict or None
        The restored state (with tensors moved to ``device``), or ``None`` when
        no checkpoint file is present.
    """
    if not os.path.exists(checkpoint_file):
        return None

    with open(checkpoint_file, "rb") as fh:
        state = pickle.load(fh)

    state["loss_sum"] = state["loss_sum"].to(device)
    state["metric_dict"] = {
        dt_i: {name: tensor.to(device) for name, tensor in metrics.items()}
        for dt_i, metrics in state["metric_dict"].items()
    }
    return state


def validation(config_dict, smoothed_eval: bool = False, **kwargs: dict):

    # Early exit if rollout, in_ts and out_ts does not match the requirements of our shitty code.
    if kwargs["rollout"] > 1 and (
        config_dict["data"]["n_in_timesteps"] > config_dict["data"]["n_out_timesteps"]
    ):
        print(
            f"Invalid rollout ({kwargs['rollout']}), in_timesteps ({config_dict['data']['n_in_timesteps']}), out_timesteps ({config_dict['data']['n_out_timesteps']}). Exiting. "
        )
        return

    (
        model,
        valid_dataloader,
        loss_fn,
        up_group,
        dtp_group,
        device,
        n_samples,
        bayesian,
        epoch,
        start_step,
        sqrt_spatial_weights,
        autocast,
        feature_weights,
    ) = setup_model(config_dict, mode="validation", **kwargs)

    torch.cuda.empty_cache()
    set_all_seeds(17 + up_group.rank())

    # get pointer to the dataset, depending on whether we have a "subset" or not
    dataset = (
        valid_dataloader.dataset.dataset
        if config_dict["data"].get("valid_subset", -1) > 0
        else valid_dataloader.dataset
    )

    validation_start = time.perf_counter_ns()
    fully_aggregated_metrics, val_loss = validate(
        config_dict=config_dict,
        dataloader=valid_dataloader,
        dataset=dataset,
        model=model,
        n_samples=n_samples,
        spatial_weights=sqrt_spatial_weights,
        loss_fn=loss_fn,
        feature_weights=feature_weights,
        epoch=epoch,
        device=device,
        autocast=autocast,
        bayesian=bayesian,
        smoothed_eval=smoothed_eval,
    )
    validation_end = time.perf_counter_ns()
    validation_time = (validation_end - validation_start) / 1e9

    all_names = make_variable_names(
        all_surface_variables=DEFAULT_SURFACE_VARIABLES,
        all_pressure_variables=DEFAULT_PRESSURE_VARIABLES,
        all_pressure_levels=DEFAULT_PRESSURE_LEVELS,
    )

    if dist.get_rank(group=None) == 0:
        for dt_i in range(
            dataset.dt,
            dataset.dt * dataset.out_timesteps * dataset.rollout + 1,
            dataset.dt,
        ):
            for key in fully_aggregated_metrics[dt_i].keys():
                print(f"{key}_{dt_i}h:")
                for name, value in zip(all_names, fully_aggregated_metrics[dt_i][key]):
                    metric_value = value.item()
                    print(f"{name}_{dt_i}h: {metric_value}")

    if dist.get_rank(group=None) == 0:
        print(
            f"Epoch {epoch} Validation: loss {val_loss}, Total time: {validation_time} s"
        )


@torch.no_grad()
def validate(
    *,
    # Config
    config_dict,
    # Data
    dataloader,
    dataset,
    # Model
    model,
    n_samples,
    # Loss and Optimizer
    spatial_weights,
    loss_fn,
    feature_weights,
    # Training
    epoch,
    device,
    autocast,
    bayesian,
    smoothed_eval=False,
):
    spatial_group = get_process_group("JSpatial")
    channel_group = get_process_group("JChannel")
    dtp_group = get_process_group("DTP")
    up_group = get_process_group("SP")
    dp_group = get_process_group("DP")

    model.eval()

    downsampling = config_dict["data"].get("downsampling", 1)

    # Get size of local variables
    nlat = dataset.xlat // downsampling
    nlon = dataset.xlon // downsampling // spatial_group.size()

    nvars = int(config_dict["data"]["n_variables"]) // channel_group.size()

    # initialize tensor for storing loss
    loss_sum = torch.tensor([0.0], device=device, requires_grad=False)
    metric_names = ["mse", "crps", "variance"] if bayesian else ["mse"]
    metric_scaling = {"mse": 2, "spread_skill_ratio": 0, "crps": 1, "variance": 2}

    # load normalization
    norm_value_stds = torch.as_tensor(dataset.norm_values["std"]).to(
        device=device, dtype=torch.float32
    )[:, 0, 0]

    # initialize dictionaries for metrics
    # , fully_aggregated_metrics = {}, {}, {}
    fully_aggregated_metrics = {}
    average_loss = []

    # initialize one empty dictionary for
    metric_dict, averaged_metrics = {}, {}
    for i_out in range(dataset.out_timesteps * dataset.rollout):
        dt_i = (i_out + 1) * dataset.dt
        fully_aggregated_metrics[dt_i] = {}
        metric_dict[dt_i] = {}
        averaged_metrics[dt_i] = {}
        for name in metric_names:
            metric_dict[dt_i][name] = torch.zeros(
                (nvars, nlat, nlon), device=device, requires_grad=False
            )

    log_interval = config_dict["training"]["log_interval"]
    dsync_interval = config_dict["training"]["dsync_interval"]
    checkpoint_interval = config_dict["training"]["checkpoint_interval"]

    # --- Resume support -------------------------------------------------------
    # The evaluation loop can be resumed from a single pickle checkpoint. Its
    # path is a user-provided ``--eval_checkpoint`` or a default in the run's
    # root directory. ``resume_idx`` is the number of batches already processed
    # and folded into ``metric_dict``/``loss_sum``. ``total_batches`` is the
    # length of the *full* dataloader and is used as the normalization
    # denominator so that partial results remain consistent regardless of where
    # the run was interrupted.
    eval_checkpoint_file = _eval_checkpoint_path(config_dict)
    total_batches = len(dataloader)
    resume_idx = 0

    resumed_state = load_eval_checkpoint(eval_checkpoint_file, device)
    if resumed_state is not None and resumed_state.get("epoch") == epoch:
        resume_idx = resumed_state["next_step"]
        total_batches = resumed_state["total_batches"]
        loss_sum = resumed_state["loss_sum"]
        metric_dict = resumed_state["metric_dict"]
        if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "resume_idx"):
            dataloader.sampler.resume_idx = resume_idx
        if dist.get_rank(group=None) == 0:
            print(
                f"Resuming evaluation from checkpoint '{eval_checkpoint_file}' "
                f"at batch {resume_idx}/{total_batches}."
            )

    data_access_rank = int(dist.get_rank(group=None) // 8)
    if dist.get_rank(group=None) == 0:
        print("Let's go... ")
        print(metric_dict.keys())

    dl_iter = iter(dataloader)
    n_local_vars = dataset.in_var[1] - dataset.in_var[0]

    start_fwd = torch.cuda.Event(enable_timing=True)
    end_fwd = torch.cuda.Event(enable_timing=True)

    start_loss = torch.cuda.Event(enable_timing=True)
    end_loss = torch.cuda.Event(enable_timing=True)

    cum_fwd_time_s = 0.0
    cum_loss_time_s = 0.0
    cum_step_time_s = 0.0

    remaining_batches = len(dataloader)
    for i in range(remaining_batches):
        # ``global_step`` is the batch index into the *full* dataset (accounting
        # for any batches skipped on resume). It is used for checkpointing so
        # that the sampler's ``resume_idx`` matches the accumulated state.
        global_step = resume_idx + i

        # Desync on first iteration

        step_start = time.perf_counter_ns()
        if i % dsync_interval == 0:
            initial_x, rollout_y, _, _ = execute_in_waves(lambda: next(dl_iter))
        else:
            initial_x, rollout_y, _, _ = next(dl_iter)

        # Get first elements
        current_x = initial_x
        current_y = rollout_y[:, : n_local_vars * dataset.out_timesteps]
        rollout_step = False

        for ri in range(dataset.rollout):
            split_timesteps = []
            x = current_x.to(device)
            y = current_y.to(device)

            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=autocast
            ):
                start_fwd.record()
                if bayesian:
                    if not rollout_step:
                        pred = model(x, samples=n_samples)
                    else:
                        x = x.reshape(n_samples, x.shape[0] // n_samples, *x.shape[1:])
                        pred = model(x, sampled_input=True)
                else:
                    pred = model(x)
                end_fwd.record()

                if ri > 0 and bayesian:
                    # After the first rollout step, the samples get feed into the model as batches, and n_samples is set to 1. The reshape fixes hopefully avoids any issues downstream for the evaluation.
                    batch_dim = pred.shape[1] // n_samples
                    pred = pred.reshape(n_samples, batch_dim, *pred.shape[2:])

                start_loss.record()
                loss = loss_fn(pred * spatial_weights, y * spatial_weights)
                end_loss.record()
                # pred.shape before slicing = [local samples, batch, local variables, lat, local space]

            # The loss covers the whole out_timesteps block of this rollout
            # step, so it is accumulated once per step here rather than once per
            # output timestep in the loop below. Averaging over the rollout
            # steps keeps the reported value on the same scale as the training
            # loss (for rollout=1 the two are identical).
            loss_sum = loss_sum + loss.detach() / (total_batches * dataset.rollout)

            # remask the sea surface temperature
            pred = torch.where(
                feature_weights == 0, SST_NORMALIZED_ZERO, pred
            ).contiguous()

            # END_DT is one dt further than the last predicted timestep in this rollout step.
            # We need this for the range.
            start_dt = dataset.dt + (dataset.out_timesteps * dataset.dt) * ri
            end_dt = start_dt + (dataset.out_timesteps * dataset.dt)

            for j, dt_i in enumerate(range(start_dt, end_dt, dataset.dt)):
                pred_slice = slice(j * n_local_vars, (j + 1) * n_local_vars)
                pred_sliced = (
                    pred[:, :, pred_slice] if bayesian else pred[:, pred_slice]
                )
                # start_f = j + (ri * dataset.out_timesteps) * n_local_vars
                # target_slice = slice(start_f, start_f + n_local_vars)
                y_sliced = y[:, pred_slice]
                split_timesteps.append(
                    pred_sliced.flatten(0, 1).cpu() if bayesian else pred_sliced.cpu()
                )

                if smoothed_eval:
                    shape = pred_sliced.shape
                    pred_sliced = pred_sliced.flatten(0, 1)
                    pred_sliced = torch.nn.functional.pad(
                        pred_sliced, (0, 1, 0, 1), mode="replicate"
                    )
                    pred_sliced = torch.nn.functional.avg_pool2d(
                        pred_sliced,
                        kernel_size=2,
                        stride=1,
                    )

                    pred_sliced = pred_sliced.reshape(*shape)

                # calculate crps and mse if bayesian
                if bayesian:
                    mean, var = mean_var_agg(pred_sliced, up_group)
                    crps_val = crps(pred_sliced, y_sliced, up_group)
                    mse_val = mse(
                        mean * feature_weights[pred_slice].unsqueeze(0),
                        y_sliced * feature_weights[pred_slice].unsqueeze(0),
                    )
                else:
                    # only calculate mse if non-bayesian
                    mse_val = mse(
                        pred_sliced * feature_weights[pred_slice].unsqueeze(0),
                        y_sliced * feature_weights[pred_slice].unsqueeze(0),
                    )

                metric_dict[dt_i]["mse"] = (
                    metric_dict[dt_i]["mse"] + mse_val / total_batches
                )

                if bayesian:
                    metric_dict[dt_i]["variance"] = (
                        metric_dict[dt_i]["variance"] + var.mean(dim=0) / total_batches
                    )
                    metric_dict[dt_i]["crps"] = (
                        metric_dict[dt_i]["crps"] + crps_val / total_batches
                    )

            rollout_step = True

            if ri < dataset.rollout - 1:
                current_x = dataset.input_from_list(split_timesteps)
                start = n_local_vars * dataset.out_timesteps * (ri + 1)
                end = start + n_local_vars * dataset.out_timesteps
                current_y = rollout_y[:, start:end]

            cum_fwd_time_s += start_fwd.elapsed_time(end_fwd) / 1e3
            cum_loss_time_s += start_loss.elapsed_time(end_loss) / 1e3

        step_end = time.perf_counter_ns()
        cum_step_time_s += (step_end - step_start) / 1e9

        # We only print once per data sample;
        # mse and losses should be averaged over the out_timesteps to match training.
        if i % log_interval == 0:
            # Average loss over all four processes for exactness
            dist.all_reduce(loss, group=dtp_group)
            if dist.get_rank(group=None) == 0:
                if bayesian:
                    avg_df, avg_pm = df_pm(loss_fn)
                    print(
                        f"{epoch}, after Step {i}/{len(dataloader)}. loss: {float(loss.detach())}, mse {torch.nn.functional.mse_loss(pred * feature_weights.unsqueeze(0), y * feature_weights.unsqueeze(0))} data fitting {avg_df} prior matching {avg_pm} Average time per batch {cum_step_time_s / log_interval} s, Average time per fwd {cum_fwd_time_s / log_interval / dataset.rollout}, Average time per loss {cum_loss_time_s / log_interval / dataset.rollout}"
                    )
                else:
                    print(
                        f"{epoch}, after Step {i}/{len(dataloader)}. loss: {float(loss.detach())} Average time per batch {cum_step_time_s / log_interval} s,  Average time per fwd {cum_fwd_time_s / log_interval / dataset.rollout}, Average time per loss {cum_loss_time_s / log_interval / dataset.rollout}"
                    )

            time.sleep(data_access_rank * 0.001)
            cum_step_time_s = 0.0
            cum_fwd_time_s = 0.0
            cum_loss_time_s = 0.0

        # Overwrite the single resumable checkpoint every ``checkpoint_interval``
        # batches. ``global_step + 1`` is the number of batches processed so far
        # and therefore the ``resume_idx`` to continue from. Only global rank 0
        # writes the file so that a single checkpoint is produced.
        if (global_step + 1) % checkpoint_interval == 0 and dist.get_rank(
            group=None
        ) == 0:
            save_eval_checkpoint(
                eval_checkpoint_file,
                next_step=global_step + 1,
                total_batches=total_batches,
                metric_dict=metric_dict,
                loss_sum=loss_sum,
                epoch=epoch,
            )

    dist.all_reduce(loss_sum, group=dtp_group, op=dist.ReduceOp.AVG)

    for dt_i in range(
        dataset.dt,
        dataset.dt * dataset.out_timesteps * dataset.rollout + 1,
        dataset.dt,
    ):
        for name in metric_dict[dt_i].keys():
            dist.all_reduce(
                metric_dict[dt_i][name], group=dp_group, op=dist.ReduceOp.AVG
            )

            averaged_metrics[dt_i][name] = latitude_weighted_average(
                metric_dict[dt_i][name]
            )
            dist.all_reduce(
                averaged_metrics[dt_i][name], group=spatial_group, op=dist.ReduceOp.AVG
            )
            fully_aggregated_metrics[dt_i][name] = gather_along_dimension(
                averaged_metrics[dt_i][name], channel_group, dim=0
            )
            if name in metric_scaling.keys() and metric_scaling[name] != 0:
                fully_aggregated_metrics[dt_i][name] = fully_aggregated_metrics[dt_i][
                    name
                ] * (norm_value_stds ** metric_scaling[name])

        if bayesian:
            fully_aggregated_metrics[dt_i]["spread_skill_ratio"] = spread_skill_ratio(
                fully_aggregated_metrics[dt_i]
            )
            del fully_aggregated_metrics[dt_i]["variance"]

        # Calculate RMSE for both bayesian and non-bayesian
        fully_aggregated_metrics[dt_i]["rmse"] = rmse(fully_aggregated_metrics[dt_i])

    average_loss = loss_sum.item()
    return fully_aggregated_metrics, average_loss
