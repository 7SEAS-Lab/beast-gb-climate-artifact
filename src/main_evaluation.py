import argparse
import pathlib
import re
import warnings

import torch.distributed as dist
from beast.comm import (
    create_process_mesh_dict,
)

from gb.config import load_configuration_yaml, setup_config_dict
from main import init_torch_distributed


# Testing path: /path/to/output/<partition>/<job_id>
def _parse_lead_times(text: str, parser: argparse.ArgumentParser) -> list[int]:
    """Parse the ``--lead-times`` value into sorted unique hours.

    Accepts either a comma-separated list of integers (e.g. ``"12,24"``) or a
    range in the form ``<start>:<stop>:<step>`` (``<step>`` is optional and
    defaults to 1). All parts may be negative, e.g. ``"-24:24:6"``. The range
    follows Python ``range`` semantics: ``<stop>`` is exclusive.

    Parameters
    ----------
    text : str
        Comma-separated lead times in hours, e.g. ``"12,24"``, or a range in
        the form ``<start>:<stop>:<step>``.
    parser : argparse.ArgumentParser
        Used to report a usage error and exit on malformed input.

    Returns
    -------
    list of int
        Sorted, de-duplicated lead times.
    """
    if ":" in text:
        parts = text.split(":")
        if len(parts) not in (2, 3):
            parser.error(
                "--lead-times range expects '<start>:<stop>' or "
                f"'<start>:<stop>:<step>', got {text!r}."
            )
        try:
            bounds = [int(part) for part in parts]
        except ValueError:
            parser.error(
                "--lead-times range expects integer '<start>:<stop>:<step>', "
                f"got {text!r}."
            )
        start, stop = bounds[0], bounds[1]
        step = bounds[2] if len(parts) == 3 else 1
        if step == 0:
            parser.error("--lead-times range <step> must not be zero.")
        leads = list(range(start, stop, step))
    else:
        try:
            leads = [int(part) for part in text.split(",") if part.strip()]
        except ValueError:
            parser.error(
                f"--lead-times expects comma-separated integers, got {text!r}."
            )
    if not leads:
        parser.error("--lead-times needs at least one value.")
    return sorted(set(leads))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Arguments shared by every subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("model_path", type=pathlib.Path)
    common.add_argument("configargs", nargs="*")

    common.add_argument(
        "--avg_pool_store",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply an average pooling on the model outputs before writing them to Zarr. Set to False automatically if avg_pool_rollout is True",
    )

    # ``eval`` runs the validation pipeline.
    eval = subparsers.add_parser("eval", parents=[common], help="Run validation.")
    eval.add_argument(
        "--rollout",
        type=int,
        default=1,
        metavar="STEPS",
        help="Number of forward passes per dataset input.",
    )
    eval.add_argument(
        "--eval_checkpoint",
        type=pathlib.Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to an evaluation checkpoint (.pkl) to resume from. If the "
            "path points to a valid checkpoint file it is loaded and the "
            "evaluation continues from where it stopped. If omitted, a default "
            "checkpoint in the run's root directory is used."
        ),
    )

    # ``ewb`` runs the extreme-weather-benchmark inference pipeline.
    ewb = subparsers.add_parser("ewb", parents=[common], help="Run EWB inference.")
    ewb.add_argument(
        "--lead_times",
        type=str,
        default=None,
        metavar="HOURS",
        help=(
            "Lead times in hours to run. Either comma-separated integers, "
            "e.g. '12,24', or a range '<start>:<stop>:<step>' (<stop> "
            "exclusive, <step> optional and defaulting to 1), e.g. '-24:24:6'. "
            "Negative values are allowed. Sets data.lead_times, overriding the "
            "value stored in the checkpoint's config.yaml. Omit to use that "
            "stored value unchanged."
        ),
    )
    ewb.add_argument(
        "--bounded_region",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Crop stored fields to each event's lat/lon bounding box.",
    )
    ewb.add_argument(
        "--bounded_features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store only the features scored for each event type.",
    )
    ewb.add_argument(
        "--denormalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Denormalize model outputs before writing them to Zarr.",
    )
    ewb.add_argument(
        "--avg_ensemble",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Average model outputs over the ensemble dimension before storing.",
    )
    ewb.add_argument(
        "--avg_pool_rollout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply an average pooling on the model outputs before output is fedback to rollout.",
    )
    ewb.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help=(
            "Overwrite the existing forecast store. By default an existing "
            "store is resumed: every (event, lead_time) group already present "
            "on disk is left untouched and skipped, so only the missing ones "
            "are computed. Pass --overwrite to wipe the store and recompute "
            "everything from scratch."
        ),
    )

    args = parser.parse_args()

    # Validate before any distributed setup so a typo fails immediately.
    lead_times = (
        None
        if getattr(args, "lead_times", None) is None
        else _parse_lead_times(args.lead_times, parser)
    )

    model_path: pathlib.Path = args.model_path
    if not model_path.exists():
        warnings.warn(f"{model_path} does not exists.")
        return 1

    cp_re = re.compile(r"cp-epoch\d+(-step\d+)?(-jc\d+)?(\.pt)?$")
    if model_path.is_file() and cp_re.match(model_path.name):
        base_path = model_path.parent.parent
    else:
        items = list(map(lambda p: str(p).split("/")[-1], model_path.iterdir()))
        if "config.yaml" in items and "checkpoints" in items:
            base_path = model_path
        else:
            warnings.warn(f"Invalid base directory {model_path}")
            return 1

    # Validation script will not overwrite the original config
    config_dict = load_configuration_yaml(base_path)
    # Pass the full args namespace: setup_config_dict reads args.configargs,
    # args.skip_validation and args.resume (not just the configargs list).
    config_dict = setup_config_dict(config_dict, args, resume=True)

    config_dict["data"]["lead_times"] = lead_times

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
    config_dict["resume_path"] = base_path

    # ---- fork: dispatch to the requested subcommand -----------------------
    if args.command == "ewb":
        _run_ewb(config_dict, args)
    else:
        _run_eval(config_dict, args)

    if dist.get_rank(group=None) == 0:
        print("FINITE!")

    dist.barrier(group=None)
    dist.destroy_process_group()


def _run_ewb(config_dict: dict, args: argparse.Namespace) -> None:
    """Run the extreme-weather-benchmark inference pipeline."""
    config_dict["run_type"] = "inference"
    # EWB output options; the flags default to True (see the ``ewb`` parser).
    config_dict["data"]["bounded_region"] = args.bounded_region
    config_dict["data"]["bounded_features"] = args.bounded_features
    config_dict["data"]["denormalize"] = args.denormalize
    config_dict["data"]["avg_ensemble"] = args.avg_ensemble
    config_dict["data"]["avg_pool_rollout"] = args.avg_pool_rollout
    config_dict["data"]["avg_pool_store"] = args.avg_pool_store
    config_dict["data"]["overwrite"] = args.overwrite

    from ewb_inference import inference

    if dist.get_rank(group=None) == 0:
        print("Running inference...")
    inference(config_dict)


def _run_eval(config_dict: dict, args: argparse.Namespace) -> None:
    """Run the validation pipeline."""
    config_dict["run_type"] = "validation"

    # Optional user-provided evaluation checkpoint to resume from. When None,
    # the evaluation falls back to a default file in the run's root directory.
    config_dict["eval_checkpoint"] = (
        str(args.eval_checkpoint) if args.eval_checkpoint is not None else None
    )

    from evaluation import validation

    if dist.get_rank(group=None) == 0:
        print("Running validation...")
    validation(config_dict, rollout=args.rollout, smoothed_eval=args.avg_pool_store)


if __name__ == "__main__":
    main()
