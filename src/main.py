import argparse
import os

import torch
import torch.distributed as dist
import yaml
from beast.comm import (
    create_process_mesh_dict,
)

from gb.config import load_configuration_yaml, setup_config_dict


def init_torch_distributed(backend: str = "nccl") -> dict[str, dist.ProcessGroup]:
    """Initialize the global process group for beast."""
    rank = int(os.getenv("SLURM_PROCID"))  # Get individual process ID.
    world_size = int(os.getenv("SLURM_NTASKS"))  # Total number of processes.
    slurm_localid = int(os.getenv("SLURM_LOCALID", rank))  # Local GPU ID.

    # print(f"Rank {rank}: world_size {world_size}, slurm_localid {slurm_localid}")
    # # Initialize GPUs
    device = "cuda" if backend == "nccl" else "cpu"
    if backend == "nccl":
        device_id = slurm_localid if torch.cuda.device_count() != 1 else 0
        device = torch.device("cuda", device_id)
        torch.cuda.set_device(device)

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        # device_id=device,
        init_method="env://",
    )
    if not dist.is_initialized():
        raise RuntimeError("Process group initialization failed.")


def torch_dist_warmup():
    device_id = torch.cuda.current_device()
    device = torch.device("cuda", device_id)

    dummy_a = torch.tensor([0], device=device)
    dummy_a += 1
    dist.all_reduce(dummy_a, op=dist.ReduceOp.SUM)
    return dummy_a


def _find_config_dir(resume_path: str) -> str:
    """Locate the directory containing ``config.yaml`` for a resume path.

    ``--resume`` may point at a checkpoint file, at the ``checkpoints/``
    directory, or directly at the run's results directory. The persisted
    ``config.yaml`` lives in the results directory (the parent of
    ``checkpoints/``), so this searches upwards from ``resume_path`` for it.
    """
    # Normalise to a directory to start the search from.
    directory = (
        os.path.dirname(resume_path) if os.path.isfile(resume_path) else resume_path
    )

    # Walk up a couple of levels looking for config.yaml (handles both the
    # results dir and the checkpoints/ subdir being passed in).
    candidate = directory
    for _ in range(3):
        if os.path.isfile(os.path.join(candidate, "config.yaml")):
            return candidate
        parent = os.path.dirname(candidate.rstrip("/"))
        if not parent or parent == candidate:
            break
        candidate = parent

    raise FileNotFoundError(
        f"Could not locate config.yaml near resume path '{resume_path}'. "
        "Pass --config_file explicitly."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    parser.add_argument("--resume")
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--flops", action="store_true")
    parser.add_argument("--skip_validation", action="store_true")

    parser.add_argument("configargs", nargs="*")

    args = parser.parse_args()

    if args.config_file is not None:
        with open(f"{args.config_file}", "r") as f:
            config_dict = yaml.load(f, Loader=yaml.SafeLoader)
        config_dict = setup_config_dict(config_dict, args)
    else:
        if args.resume is None:
            raise RuntimeError(
                "You must provide either --config_file or --resume <checkpoint path>."
            )
        # When resuming without a config file, reload the config that was
        # persisted next to the checkpoints. --resume may point at a checkpoint
        # file or at the checkpoints/ directory; the config.yaml lives in the
        # run's results directory (the parent of checkpoints/).
        config_dir = _find_config_dir(args.resume)
        config_dict = load_configuration_yaml(config_dir)
        config_dict = setup_config_dict(config_dict, args, resume=True)

    ### Distributed Stuff ###
    init_torch_distributed()
    dist.barrier(group=None)
    global_rank = dist.get_rank(group=None)

    warmup_result = torch_dist_warmup()
    if global_rank == 0:
        print(f"torch.dist warm-up: allreduce = {warmup_result}")

    mesh_dims = config_dict["mesh_dims"]
    create_process_mesh_dict(mesh_dims)
    dist.barrier(group=None)

    if global_rank == 0:
        print(f"Data path: {config_dict['data']['data_path']}")
        print(f"Mask path: {config_dict['data']['constant_masks_path']}")
        print(f"Normalization dir: {config_dict['data']['normalization_dir']}")

    from training import training_loop

    training_loop(config_dict, measure_flops=args.flops, use_mlflow=args.mlflow)

    if dist.get_rank(group=None) == 0:
        print("FINITE!")

    dist.barrier(group=None)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
