import os

import beast.data.transforms as trf
import torch
import torch.distributed as dist
import yaml
from beast.comm import get_process_group, wrap_ddp
from beast.data.dataloader import get_dataloader
from beast.data.era5 import EWBEra5
from beast.losses.up_loss import UPAllGaussLoss
from beast.model import get_model

# beast imports
from beast.model.utils import (
    convert_jigsaw_vi,
    count_model_parameters,
)
from beast.utils import get_spatial_weights, set_all_seeds, set_dtype
from jigsaw import launch as jigsaw_launch

from gb.checkpoint import (
    resume_training,
)
from gb.ewb import load_ewb_events
from gb.metrics import mse_loss_fn


def setup_optimizer(config_dict, model):
    """Setup the optimizer."""
    low_lr_params, normal_lr_params = [], []
    low_key = ["patch_embedding", "downsample", "upsample", "patch_recovery"]
    for name, param in model.named_parameters():
        if (
            low_key[0] in name
            or low_key[1] in name
            or low_key[2] in name
            or low_key[3] in name
        ):
            low_lr_params.append(param)
        else:
            normal_lr_params.append(param)

    # This needs to happen AFTER THE MODEL IS WRAPPED!!!!!!!!
    optimizer = torch.optim.AdamW(
        [
            {
                "params": low_lr_params,
                "lr": config_dict["training"]["low_lr"],
            },
            {
                "params": normal_lr_params,
                "lr": config_dict["training"]["normal_lr"],
            },
        ],
        betas=(0.9, 0.90),
        eps=1e-6,
        weight_decay=0,
    )
    return optimizer


def setup_scheduler(optimizer, config_dict):
    """Setup the cosine annealing LR scheduler."""
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config_dict["training"]["n_epochs"], eta_min=1e-5
    )
    return scheduler


def setup_model(config_dict, mode="training", **kwargs):
    """
    Set up the model.

    The train and valid_dataloader will be setup where necessary.
    Optimizer and scheduler is only returned for training.

    mode=testing is not currently implemented, but that should be equivalent
    to mode=validation.
    """
    spatial_group = get_process_group("JSpatial")
    channel_group = get_process_group("JChannel")
    dtp_group = get_process_group("DTP")
    up_group = get_process_group("SP")
    dp_group = get_process_group("DP")

    downsampling = config_dict["data"].get("downsampling", 1)

    n_samples = config_dict["training"]["n_samples"]
    dtype = set_dtype(config_dict["training"]["dtype"])
    autocast = config_dict["training"]["autocast"]
    device_id = torch.cuda.current_device()
    device = torch.device("cuda", device_id)

    initialization_rank = (
        3 + channel_group.rank() + channel_group.size() + up_group.size()
    )

    if dist.get_rank(group=None) == 0:
        print("Getting Dataloader")

    transforms = (
        None
        if downsampling == 1
        else trf.Downsample(factor=downsampling, dims=(-1, -2))
    )

    main_dl = None
    alt_dl = None

    match mode:
        case "training":
            main_dl = get_dataloader(
                config_dict,
                dtp_rank=dist.get_rank(dtp_group),
                dtp_size=dist.get_world_size(dtp_group),
                dp_rank=dist.get_rank(dp_group),
                dp_size=dist.get_world_size(dp_group),
                mode="train",
                transform=transforms,
                **kwargs,
            )
            dataset = (
                main_dl.dataset.dataset
                if config_dict["data"]["train_subset"] > 0
                else main_dl.dataset
            )
            if not config_dict["skip_validation"]:
                alt_dl = get_dataloader(
                    config_dict,
                    dtp_rank=dist.get_rank(dtp_group),
                    dtp_size=dist.get_world_size(dtp_group),
                    dp_rank=dist.get_rank(dp_group),
                    dp_size=dist.get_world_size(dp_group),
                    mode="validation",
                    transform=transforms,
                )
        case "validation":
            main_dl = get_dataloader(
                config_dict,
                dtp_rank=dist.get_rank(dtp_group),
                dtp_size=dist.get_world_size(dtp_group),
                dp_rank=dist.get_rank(dp_group),
                dp_size=dist.get_world_size(dp_group),
                mode="validation",
                transform=transforms,
                **kwargs,
            )
            dataset = (
                main_dl.dataset.dataset
                if config_dict["data"]["valid_subset"] > 0
                else main_dl.dataset
            )
        case "inference":
            events_path = config_dict["data"].get("ewb_events_path")
            if not events_path:
                raise ValueError(
                    "EWB inference requires data.ewb_events_path or the "
                    "EWB_EVENTS_PATH environment variable."
                )
            event_list = load_ewb_events(
                events_path,
                case_ids=config_dict["data"].get("ewb_case_ids"),
                event_types=config_dict["data"].get("ewb_event_types"),
            )
            main_dl = EWBEra5(
                event_list=event_list,
                lead_times=config_dict["data"].get("lead_times"),
                pg_rank=dist.get_rank(group=dtp_group),
                pg_size=dist.get_world_size(group=dtp_group),
                data_path=config_dict["data"]["data_path"],
                constant_masks_path=config_dict["data"]["constant_masks_path"],
                constant_masks=config_dict["data"]["constant_masks"],
                normalization_dir=config_dict["data"]["normalization_dir"],
                time_range=None,
                dt=config_dict["data"]["dt"],
                n_in_timesteps=config_dict["data"]["n_in_timesteps"],
                n_out_timesteps=config_dict["data"]["n_out_timesteps"],
                dtype=dtype,
                transform=transforms,
                target_transform=transforms,
            )
            dataset = main_dl

    if dist.get_rank(group=None) == 0:
        print(
            f"Dataloader ready: {len(main_dl)} {mode} batches, {len(alt_dl) if alt_dl is not None else 'no'} validation batches"
        )

    # Define the Model
    if dist.get_rank(group=None) == 0:
        print("Initializing Model")

    set_all_seeds(initialization_rank)

    # get the xlat and xlon from the datasets
    xlat, xlon = dataset.xlat // downsampling, dataset.xlon // downsampling

    sqrt_spatial_weights = torch.sqrt(get_spatial_weights(xlat, device))

    if dist.get_rank(group=None) == 0:
        print("Getting Model")

    model = get_model(
        **config_dict["model"],
        height=xlat,
        width=xlon,
        device=device,
        dtype=dtype,
        channel_group=channel_group,
        spatial_group=spatial_group,
        dtp_group=dtp_group,
    )
    if dist.get_rank(group=None) == 0:
        print("Moving model to device")

    model = model.to(dtype).to(device)
    bayesian = model.bayesian

    # Collective over DTP: every rank has to call it.
    n_params = count_model_parameters(
        model, {"DTP": dtp_group}, pending_bayesian_conversion=bayesian
    )
    if dist.get_rank(group=None) == 0:
        print(f"Parameters: {n_params}")

    if dist.get_rank(group=None) == 0 and bayesian:
        print("Make model Bayesian")

    if bayesian:
        if dist.get_world_size(group=dtp_group) > 1:
            if dist.get_rank(group=None) == 0:
                print("Jigsaw beastling")
            convert_jigsaw_vi(model, channel_group, spatial_group)
        else:
            if dist.get_rank(group=None) == 0:
                print("Non-Jigsaw beastling")
            import torch_blue.vi as vi
            import torch_blue.vi.distributions as vidist
            from beast.layers.nice_norm import NiceNorm

            vi.utils.convert.ban_convert(NiceNorm, ban_mode="submodule")
            vi.convert_to_vimodule(
                model,
                variational_distribution=vidist.Normal(std=0.15),
                prior=vidist.Normal(),
                rescale_prior=False,
                kaiming_initialization=True,
                prior_initialization=False,
                return_log_probs=True,
                keep_weights=False,
            )
        model.return_log_probs = False

    if dist.get_rank(group=None) == 0:
        print("Make model data parallel")

    # DDP averages over DP x SP; JSpatial is reduced by jigsaw. A Bayesian
    # model's SP ranks hold different ensemble members: the UP loss scales
    # their data-fitting gradient by the SP size, so DDP's average sums it.
    # A deterministic model's SP ranks are plain copies, averaged like DP.
    ddp_group = get_process_group("DDP")
    model = wrap_ddp(model, device_id=device_id, ddp_group=ddp_group)

    if dist.get_rank(group=None) == 0:
        print("Model is data parallel")

    # set up optimizer and scheduler if training
    optimizer = setup_optimizer(config_dict, model) if mode == "training" else None
    scheduler = setup_scheduler(optimizer, config_dict) if mode == "training" else None

    # Load checkpoints. Resume is triggered by the ``--resume <path>`` CLI flag
    # (stored in config_dict["resume_path"]). ``start_step`` follows the
    # convention encoded in the checkpoint: a mid-epoch checkpoint (step is not
    # None) resumes within the same epoch, an end-of-epoch checkpoint (step is
    # None) resumes at the start of the next epoch.
    if config_dict["resume_path"] is None:
        start_epoch, start_step = 0, 0
    else:
        model, optimizer, start_epoch, start_step = resume_training(
            str(config_dict["resume_path"]),
            model,
            optimizer,
        )
        if mode == "inference":
            start_epoch, start_step = 0, 0

    if dist.get_rank(group=None) == 0:
        print("Setup loss")

    feature_weights = dataset.get_feature_weights(channel_group.rank()).to(device)
    if downsampling > 1:
        feature_weights = -torch.nn.functional.max_pool2d(
            -feature_weights, downsampling
        )
    # initialize feature_weights
    loss_fn = (
        UPAllGaussLoss(
            model,
            len(dataset),
            up_group=up_group,
            heat=0.1,
            track=True,
            feature_weights=feature_weights,
        )
        if bayesian
        else mse_loss_fn(dtp_group)
    )

    jigsaw_launch.setup_precision(config_dict["training"]["tf32"])

    if mode == "training":
        return (
            model,
            optimizer,
            main_dl,
            alt_dl,
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
        )
    else:
        return (
            model,
            main_dl,
            loss_fn,
            up_group,
            dtp_group,
            device,
            n_samples,
            bayesian,
            start_epoch,
            start_step,
            sqrt_spatial_weights,
            autocast,
            feature_weights,
        )


def setup_paths(config_dict):
    ## Validation, profiling and timings
    if config_dict["resume_path"] is not None:
        results_path = config_dict["resume_path"]
    else:
        results_path = os.path.join(
            os.getenv("OUTPUT_DIR"),
            os.getenv("SLURM_JOB_PARTITION"),
            os.getenv("SLURM_JOB_ID", "interactive"),
        )

    validation_results_path = os.path.join(results_path, "validation")
    checkpoint_path = os.path.join(results_path, "checkpoints")
    timings_path = os.path.join(results_path, "timings")
    if dist.get_rank(group=None) == 0:
        os.makedirs(validation_results_path, exist_ok=True)
        with open(f"{results_path}/config.yaml", "w") as file:
            yaml.dump(config_dict, file)

        os.makedirs(timings_path, exist_ok=True)
        os.makedirs(checkpoint_path, exist_ok=True)

    return results_path, validation_results_path, checkpoint_path
