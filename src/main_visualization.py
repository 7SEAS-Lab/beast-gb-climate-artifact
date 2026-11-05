import argparse
import os
import re

import torch
import torch.distributed as dist
from beast.comm import create_process_mesh_dict, get_process_group
from beast.evaluation import (
    gather_along_dimension,
    make_variable_names,
    pred_target_bias_plot,
    select_variables,
)
from beast.utils import set_all_seeds

from evaluation import (
    DEFAULT_PRESSURE_LEVELS,
    DEFAULT_PRESSURE_VARIABLES,
    DEFAULT_SURFACE_VARIABLES,
)
from gb.config import load_configuration_yaml, overwrite_config
from gb.utils import count_flops
from main import init_torch_distributed
from setup import setup_model


def visualize(config_dict, output_path):
    if dist.get_rank(group=None) == 0:
        print("Setting up model.")
    (
        model,
        dataloader,
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
    ) = setup_model(config_dict, mode="validation")

    # still haven't gotten rid of the count_flops problem
    _ = count_flops(
        model,
        dataloader,
        loss_fn,
        device,
        dtp_group,
        n_samples,
        bayesian,
    )

    torch.cuda.empty_cache()
    set_all_seeds(17 + up_group.rank())

    spatial_group = get_process_group("JSpatial")
    channel_group = get_process_group("JChannel")
    dtp_group = get_process_group("DTP")
    up_group = get_process_group("SP")
    # dp_group = get_process_group("DP")

    viz(
        dataloader=dataloader,
        model=model,
        n_samples=n_samples,
        device=device,
        autocast=autocast,
        bayesian=bayesian,
        channel_group=channel_group,
        spatial_group=spatial_group,
        output_path=output_path,
    )


@torch.no_grad()
def viz(
    *,
    # Data
    dataloader,
    # Model
    model,
    n_samples,
    # Training
    device,
    autocast,
    bayesian,
    channel_group,
    spatial_group,
    output_path,
):

    model.eval()

    if dist.get_rank(group=None) == 0:
        print("Let's go... ")

    dl_iter = iter(dataloader)
    x, y, _, _ = next(dl_iter)

    x = x.to(device)
    y = y.to(device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast):
        pred = model(x, samples=n_samples) if bayesian else model(x)

        pred = pred[:, 0, :42]  # hacky way to get the first half
        y = y[0, :42]

    pressure_levels_we_want = [850]
    all_names = make_variable_names(
        all_surface_variables=DEFAULT_SURFACE_VARIABLES,
        all_pressure_variables=DEFAULT_PRESSURE_VARIABLES,
        all_pressure_levels=pressure_levels_we_want,
    )

    filtered_pred = select_variables(
        pred,
        channel_rank=channel_group.rank(),
        channel_group_size=channel_group.size(),
        all_surface_variables=DEFAULT_SURFACE_VARIABLES,
        all_pressure_variables=DEFAULT_PRESSURE_VARIABLES,
        all_pressure_levels=DEFAULT_PRESSURE_LEVELS,
        eval_pressure_levels=pressure_levels_we_want,
        dim=-3,
    )

    filtered_y = select_variables(
        y,
        channel_rank=channel_group.rank(),
        channel_group_size=channel_group.size(),
        all_surface_variables=DEFAULT_SURFACE_VARIABLES,
        all_pressure_variables=DEFAULT_PRESSURE_VARIABLES,
        all_pressure_levels=DEFAULT_PRESSURE_LEVELS,
        eval_pressure_levels=pressure_levels_we_want,
        dim=-3,
    )

    filtered_pred = gather_along_dimension(
        filtered_pred, channel_group, dim=-3, variable_size=True
    )
    filtered_y = gather_along_dimension(
        filtered_y, channel_group, dim=-3, variable_size=True
    )

    gathered_pred = gather_along_dimension(filtered_pred, spatial_group, dim=-1)
    gathered_y = gather_along_dimension(filtered_y, spatial_group, dim=-1)

    num_leftover_vars = gathered_pred.shape[-3]

    if dist.get_rank(group=None):
        for i in range(num_leftover_vars):
            var_pred_mean = gathered_pred[..., i, :, :].mean(dim=0)
            var_y = gathered_y[i, :, :]

            fig = pred_target_bias_plot(var_pred_mean, var_y, all_names[i])
            full_path = os.path.join(output_path, f"{all_names[i]}.png")

            fig.savefig(full_path)

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    parser.add_argument("--resume")
    parser.add_argument("--output_path", default="/path/to/output/viz")
    parser.add_argument("configargs", nargs="*")

    args = parser.parse_args()

    valid_path = None if args.resume is None else f"{args.resume}"

    # if partial paths or path to the actual file is found, store the directory in the config
    # valid options include:
    # cp-epoch{num}-step{num}-jc{num}.pt
    # cp-epoch{num}-jc{num}.pt
    # cp-epoch{num}
    # cp-epoch{num}-step{num}
    if re.search(r"cp-epoch\d+(-step\d+)?(-jc\d+)?(\.pt)?$", valid_path):
        valid_dir = os.path.join(os.path.dirname(valid_path), "..")
    elif os.path.isdir(valid_path):
        valid_dir = valid_path
    else:
        raise ValueError(
            f"Resume path {valid_path} is neither a valid path to a directory nor a partial path to the checkpoints."
        )

    # Validation script will not overwrite the original config
    config_dict = load_configuration_yaml(valid_dir)
    config_dict = overwrite_config(config_dict, args.configargs)

    ### Distributed Stuff ###
    init_torch_distributed()
    mesh_dims = config_dict["mesh_dims"]
    create_process_mesh_dict(mesh_dims)
    dist.barrier(group=None)
    global_rank = dist.get_rank(group=None)

    if global_rank == 0:
        print(f"Data path: {config_dict['data']['data_path']}")
        print(f"Mask path: {config_dict['data']['constant_masks_path']}")
        print(f"Normalization dir: {config_dict['data']['normalization_dir']}")

    # set the config dict resume_path to either the directory or the specific checkpoint
    config_dict["resume_path"] = valid_path

    visualize(config_dict, args.output_path)

    dist.barrier(group=None)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
