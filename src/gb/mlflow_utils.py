import os

import git
import mlflow
import torch.distributed as dist


def setup_mlflow_tracking(config_dict: dict[str, any]) -> str:
    global_rank = dist.get_rank(group=None)
    if global_rank == 0:
        SLURM_JOB_ID = os.getenv("SLURM_JOB_ID", "interactive")
        USER = os.getenv("USER", "sailor")
        PARTITION = os.getenv("SLURM_JOB_PARTITION", "")
        MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

        run_name = f"{USER}_{PARTITION}_{SLURM_JOB_ID}"

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("beast")

        # When resuming (``--resume`` given and a run_id was persisted in the
        # reloaded config), continue logging to the existing MLflow run instead
        # of creating a new one, so all metrics live under a single run.
        resume_run_id = (
            config_dict.get("run_id")
            if config_dict.get("resume_path") is not None
            else None
        )
        run_type = config_dict.get("run_type", "training")

        if resume_run_id is not None:
            if run_type == "training":
                run_id = mlflow.start_run(run_id=resume_run_id).info.run_id
                print(f"Resumed MLflow run with ID: {run_id}")
                # Refresh metadata (new SLURM job id, git commit, etc.) but avoid
                # re-logging config params which are immutable for an existing run.
                track_metadata(config_dict)
            else:
                run_id = mlflow.start_run(
                    parent_run_id=run_id, nested=True, tags={"type": run_type}
                ).info_run_id
                print(f"Evaluating MLflow run with ID: {run_id}")

        else:
            run_id = mlflow.start_run(run_name=run_name).info.run_id
            print(f"Started MLflow run with ID: {run_id}")
            track_metadata(config_dict)
            track_config(config_dict)

        dist.broadcast_object_list([run_id], src=0)  # Broadcast run_id to all processes
    else:
        obj_list = [None]
        dist.broadcast_object_list(
            obj_list, src=0
        )  # Ensure all processes wait for the run_id broadcast
        run_id = obj_list[0] if obj_list else None
    return run_id


def track_config(config: dict[str, any], prefix: str = ""):
    for key, value in config.items():
        if isinstance(value, dict):
            track_config(value, prefix=key)
        else:
            full_key = f"{prefix}:{key}" if prefix else key
            mlflow.log_param(full_key, value, synchronous=False)


def track_metadata(config_dict: dict[str, any]):
    print("Tracking metadata...")
    # SLURM metadata
    mlflow.set_tag("slurm_job_id", os.getenv("SLURM_JOB_ID"), synchronous=False)
    mlflow.set_tag("slurm_nnodes", os.getenv("SLURM_NNODES"), synchronous=False)
    mlflow.set_tag(
        "slurm_job_partition", os.getenv("SLURM_JOB_PARTITION"), synchronous=False
    )
    mlflow.set_tag(
        "slurm_tasks_per_node", os.getenv("SLURM_TASKS_PER_NODE"), synchronous=False
    )
    mlflow.set_tag(
        "slurm_cpus_per_task", os.getenv("SLURM_CPUS_PER_TASK"), synchronous=False
    )
    mlflow.set_tag("user", os.getenv("USER"), synchronous=False)

    # Git metadata
    try:
        repo = git.Repo(search_parent_directories=True)
        mlflow.set_tag("git_commit_hash", repo.head.object.hexsha, synchronous=False)
        mlflow.set_tag("git_branch", repo.active_branch.name, synchronous=False)
    except git.exc.InvalidGitRepositoryError:
        mlflow.set_tag("git_commit_hash", "N/A", synchronous=False)
        mlflow.set_tag("git_branch", "N/A", synchronous=False)


def get_logging_functions(run_id: str) -> tuple[callable, callable]:

    global_rank = dist.get_rank()

    if global_rank == 0:

        def log_metric(key: str, value: any, step: int | None = None):
            mlflow.log_metric(
                key,
                value,
                step=step,
                synchronous=False,
                run_id=run_id,
            )

        def log_metrics(metrics: dict, step: int | None = None):
            mlflow.log_metrics(metrics, step=step, synchronous=False, run_id=run_id)

    else:

        def log_metric(key: str, value: any, step: int | None = None):
            pass  # No-op for non-DDP rank 0 processes

        def log_metrics(metrics: dict, step: int | None = None):
            pass  # No-op for non-DDP rank 0 processes

    return log_metric, log_metrics
