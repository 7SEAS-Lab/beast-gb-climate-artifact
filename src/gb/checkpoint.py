"""Checkpointing utilities for beast.

This module wraps the distributed / Bayesian-aware checkpointing primitives
from the ``beast`` package and adds project-specific behaviour required by
the GrandBellbeast training and evaluation pipelines:

* Checkpoints carry the MLflow ``run_id``, ``epoch``, ``step`` and ``loss`` so
  that a run can be resumed exactly, or later loaded by an inference /
  evaluation pipeline.
* Weights are stored redundantly on the *local filesystem* and as *MLflow
  artifacts*.

The heavy lifting (handling of distributed model shards, DDP prefix
(un)wrapping and Bayesian VI weights) is delegated to ``beast.checkpoint`` --
this module only orchestrates *which* checkpoints are written / kept and how
they are mirrored to MLflow.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

import torch
import torch.distributed as dist
from beast.checkpoint import (
    is_checkpoint_writer,
    parse_checkpoint_path,
)
from beast.checkpoint import (
    load_checkpoint as _beast_load_checkpoint,
)
from beast.checkpoint import (
    load_state_dict as load_state_dict,  # re-exported for the training loop
)
from beast.checkpoint import (
    save_checkpoint as _beast_save_checkpoint,
)
from beast.comm import get_process_group

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "load_latest_model_states",
    "load_state_dict",
    "resume_training",
]

# MLflow artifact sub-directory in which checkpoints are mirrored.
MLFLOW_CHECKPOINT_DIR = "checkpoints"


def _current_channel_rank() -> int:
    """Return the JChannel rank of the current process.

    Checkpoints are sharded along the JChannel dimension (one file per channel
    rank, written as ``...-jc{rank}.pt``). When resuming, each process must load
    the shard that matches *its own* channel rank. Returns ``-1`` when the model
    is not channel-distributed, matching the ``channel_group_rank`` convention
    used by :func:`beast.checkpoint.save_checkpoint`.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return -1

    try:
        channel_group = get_process_group("JChannel")
    except Exception:
        channel_group = None

    if channel_group is None:
        return -1

    return channel_group.rank()


def save_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    step: int | None,
    loss: float | torch.Tensor,
    *,
    run_id: str | None = None,
    log_to_mlflow: bool = False,
) -> str:
    """Save a checkpoint locally, mirror it to MLflow.

    The model weights (and, when supplied, the optimizer state) are written to
    ``checkpoint_path`` through :func:`beast.checkpoint.save_checkpoint`, which
    transparently handles distributed shards, DDP prefixing and Bayesian VI
    parameters. The resulting file additionally records the MLflow ``run_id``
    so it can be traced back to the experiment it belongs to.

    After writing, the checkpoint is uploaded as an MLflow artifact (for
    redundancy).

    Parameters
    ----------
    checkpoint_path:
        Directory in which checkpoints are stored.
    model:
        Model to checkpoint (may be wrapped in ``DistributedDataParallel``).
    optimizer:
        Optimizer whose state should be saved so training can resume. Pass
        ``None`` for inference-only checkpoints.
    epoch:
        Current epoch.
    step:
        Current step (batch index) within the epoch.
    loss:
        Loss associated with this checkpoint.
    run_id:
        MLflow run id to associate with the checkpoint. If ``None`` the id of
        the currently active MLflow run (if any) is used.
    log_to_mlflow:
        When ``True`` the checkpoint file is uploaded as an MLflow
        artifact in addition to being written locally.

    Returns
    -------
    str
        Path to the checkpoint file written by this rank.
    """
    # Delegate the actual (distributed / Bayesian) write to beast.
    full_path = _beast_save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        step=step,
        loss=loss,
    )

    # Only the ranks that actually wrote a file continue with bookkeeping.
    if not is_checkpoint_writer():
        return full_path

    # Mirror to MLflow for redundancy (rank-0 only to avoid duplicate uploads).
    if log_to_mlflow:
        import mlflow

        if run_id is None:
            active_run = mlflow.active_run()
            run_id = active_run.info.run_id if active_run is not None else None

        if run_id is not None:
            try:
                mlflow.log_artifact(
                    full_path,
                    artifact_path=MLFLOW_CHECKPOINT_DIR,
                    run_id=run_id,
                )
            except Exception as err:  # pragma: no cover - network / tracking issues
                print(f"[checkpoint] Failed to log checkpoint to MLflow: {err}")

    return full_path


def load_checkpoint(
    checkpoint_path: str,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a single checkpoint file.

    Thin wrapper around :func:`beast.checkpoint.load_checkpoint` kept here so
    the rest of the project imports every checkpoint primitive from
    ``gb.checkpoint``. The returned dictionary contains the model / optimizer
    state dicts together with ``epoch``, ``step``, ``loss`` and (when written
    by :func:`save_checkpoint`) ``run_id``.

    Parameters
    ----------
    checkpoint_path:
        Path to the checkpoint file.
    map_location:
        Passed through to :func:`torch.load` to control where storages are
        materialised.

    Returns
    -------
    dict[str, Any]
        The loaded checkpoint dictionary.
    """
    return _beast_load_checkpoint(checkpoint_path, map_location=map_location)


def _resolve_checkpoint_file(
    resume_path: str,
    channel_rank: int,
) -> str:
    """Return the checkpoint file this process should load from ``resume_path``.

    ``resume_path`` may point directly at a ``.pt`` file or at a directory that
    contains one or more checkpoints. When it is a directory, the *latest*
    checkpoint (highest ``epoch`` then ``step``) belonging to the current
    process' ``channel_rank`` is selected.

    Parameters
    ----------
    resume_path:
        Path to a checkpoint file, a directory of checkpoints, or
        a checkpoint file group, with the structure cp-epoch{num}-step{num}.
        This function will resolve the correct channel and return the full ckpt path.
    channel_rank:
        JChannel rank of the current process (``-1`` when not channel
        distributed), used to pick the matching shard.

    Returns
    -------
    str
        Absolute path to the checkpoint file to load.
    """
    if os.path.isfile(resume_path):
        return resume_path
    elif re.search(r"cp-epoch\d+(-step\d+)?", resume_path):
        return resume_path + f"-jc{channel_rank}.pt"
    else:
        resume_path = os.path.join(resume_path, "checkpoints")

    if not os.path.isdir(resume_path):
        raise FileNotFoundError(
            f"Resume path '{resume_path}' is neither a file nor a directory."
        )

    candidates: list[dict[str, Any]] = []
    for fname in os.listdir(resume_path):
        if not fname.endswith(".pt"):
            continue
        info = parse_checkpoint_path(fname)
        if info is None:
            continue
        # Keep only the shard matching this process' channel rank. beast encodes
        # a missing shard as ``math.inf`` for non-distributed checkpoints, which
        # we normalise to -1 to match ``channel_rank``.
        shard = info["jchannel_rank"]
        shard = -1 if shard is None else int(shard)
        if shard != channel_rank:
            continue
        candidates.append(info)

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint for channel rank {channel_rank} found in '{resume_path}'."
        )

    latest = max(candidates, key=lambda x: (x["epoch"], x["step"]))
    return os.path.join(resume_path, latest["fname"])


def load_latest_model_states(
    resume_path: str,
    map_location: str | torch.device = "cpu",
) -> tuple[dict, dict | None, int, int | None, float, int, str | None]:
    """Load the most recent checkpoint from ``resume_path``.

    ``resume_path`` may either point directly at a ``.pt`` checkpoint file or at
    a directory containing several checkpoints; in the latter case the latest
    one (highest ``epoch``, then ``step``) for the current process' JChannel
    shard is selected.

    The return signature is a superset of the values consumed by the training
    loop and is intentionally positional so it can be unpacked directly, e.g.::

        (model_sd, opt_sd, epoch, step, loss, jc_rank, run_id) = \
            load_latest_model_states(resume_path)

    Note that ``step`` is returned verbatim from the checkpoint: a value of
    ``None`` means the checkpoint was written *after* the epoch finished (see
    :func:`resume_training` for how this drives the resume point).

    Parameters
    ----------
    resume_path:
        Path to a checkpoint file or a directory of checkpoints.
    map_location:
        Passed through to :func:`torch.load`.

    Returns
    -------
    tuple
        ``(model_state_dict, optimizer_state_dict, epoch, step, loss,
        channel_group_rank, run_id)``.
    """
    checkpoint_file = _resolve_checkpoint_file(resume_path, _current_channel_rank())
    checkpoint = load_checkpoint(checkpoint_file, map_location=map_location)

    return (
        checkpoint["model_state_dict"],
        checkpoint.get("optimizer_state_dict"),
        checkpoint["epoch"],
        checkpoint.get("step"),
        checkpoint.get("loss", float("inf")),
        checkpoint.get("channel_group_rank", -1),
        checkpoint.get("run_id"),
    )


def resume_training(
    resume_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, torch.optim.Optimizer, int, int]:
    """Restore ``model`` and ``optimizer`` from a checkpoint and compute where to resume.

    This is the single entry point used by the training script when the
    ``--resume`` command line argument is given. It loads the latest checkpoint
    found at ``resume_path`` (a file or a directory), restores the model and
    optimizer state via :func:`beast.checkpoint.load_state_dict` (which handles
    DDP prefixing and Bayesian VI weights), and derives the epoch / step at
    which training should continue.

    Resume-point convention
    ------------------------
    * If the checkpoint's ``step`` is ``None`` the checkpoint was taken *after*
      the epoch finished, so training resumes at the **start of the next
      epoch**: ``start_epoch = epoch + 1``, ``start_step = 0``.
    * Otherwise the checkpoint was taken *mid-epoch* after completing ``step``,
      so training resumes **within the same epoch at the next step**:
      ``start_epoch = epoch``, ``start_step = step + 1``.

    Parameters
    ----------
    resume_path:
        Path to a checkpoint file or a directory of checkpoints.
    model:
        Model (possibly ``DistributedDataParallel``-wrapped) to load weights into.
    optimizer:
        Optimizer to restore.
    map_location:
        Passed through to :func:`torch.load`.

    Returns
    -------
    tuple
        ``(model, optimizer, start_epoch, start_step)``.
    """
    (
        model_state_dict,
        optimizer_state_dict,
        epoch,
        step,
        _loss,
        _channel_rank,
        _run_id,
    ) = load_latest_model_states(resume_path, map_location=map_location)

    # Restore weights / optimizer. beast returns ``last_epoch + 1`` but we
    # recompute the resume point below from the (epoch, step) convention, so the
    # third element is intentionally discarded.

    state_dict = load_state_dict(
        model,
        optimizer,
        model_state_dict,
        optimizer_state_dict,
        epoch,
    )
    if isinstance(state_dict, tuple) and len(state_dict) == 3:
        model, optimizer, _ = state_dict
    else:
        model = state_dict

    if step is None:
        # Checkpoint written after the epoch completed -> start the next epoch.
        start_epoch = epoch + 1
        start_step = 0
    else:
        # Mid-epoch checkpoint -> continue the same epoch after the saved step.
        start_epoch = epoch
        start_step = step + 1

    if (
        not (dist.is_available() and dist.is_initialized())
        or dist.get_rank(group=None) == 0
    ):
        print(
            f"[checkpoint] Resuming from '{resume_path}': "
            f"epoch={epoch}, step={step} -> start_epoch={start_epoch}, start_step={start_step}"
        )

    return model, optimizer, start_epoch, start_step
