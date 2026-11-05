#!/usr/bin/env python3
"""
parse_scaling.py - Parse the newer weak-scaling FP8 log format and write one CSV
summary per log folder.

Usage:
    python parse_scaling.py --logs_dir /path/to/logs/ --flops 123.45
    python parse_scaling.py --logs_dir /path/to/logs/ --flops 123.45 --out_path results.csv
"""

import argparse
import glob
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator

from src.plotting.utils import init_matplotlib, set_size


def collect_paths(path: str) -> list[Path]:
    p = Path(path)
    if p.is_dir():
        return sorted(p.glob("*.out"))
    return sorted(Path(match) for match in glob.glob(path))


def summarize(values: list[float]) -> tuple[float, float]:
    series = pd.Series(values, dtype=float)
    return float(series.mean()), float(series.std())


def ratio_std(value: float, mean: float, std: float) -> float:
    if mean <= 0:
        return float("nan")
    return abs(value * std / mean)


def resolve_mesh_dims(mesh_dims: list[int], ntasks: int) -> list[int]:
    if mesh_dims.count(-1) > 1:
        raise ValueError("mesh_dims can only contain one -1 placeholder")

    if -1 not in mesh_dims:
        return mesh_dims

    known_product = math.prod(dim for dim in mesh_dims if dim != -1)
    if known_product <= 0:
        raise ValueError(f"invalid mesh_dims={mesh_dims}")

    if ntasks % known_product != 0:
        raise ValueError(f"ntasks={ntasks} does not match mesh_dims={mesh_dims}")

    missing_dim = ntasks // known_product
    return [missing_dim if dim == -1 else dim for dim in mesh_dims]


def parse_log(path: Path, flops: float) -> dict[str, Any] | None:
    text = path.read_text()

    job_match = re.search(r"SLURM_JOB_ID:\s+(\d+)", text)
    nodes_match = re.search(r"SLURM_NNODES:\s+(\d+)", text)
    tasks_match = re.search(r"SLURM_NTASKS:\s+(\d+)", text)
    mesh_match = re.search(r"mesh_dims=([\d,-]+)", text)

    if not (job_match and nodes_match and tasks_match and mesh_match):
        print(f"  [skip] missing Slurm or mesh metadata in {path.name}")
        return None

    job_id = int(job_match.group(1))
    nodes = int(nodes_match.group(1))
    gpus = int(tasks_match.group(1))
    mesh_dims = [int(dim) for dim in mesh_match.group(1).split(",")]

    try:
        mesh_dims = resolve_mesh_dims(mesh_dims, gpus)
    except ValueError as exc:
        print(f"  [skip] {path.name}: {exc}")
        return None

    average_time_per_batch = [
        float(value)
        for value in re.findall(r"Average time per fwd\s*([\d.eE+\-]+),", text)
    ]
    if len(average_time_per_batch) == 0:
        average_time_per_batch = [
            float(value)
            for value in re.findall(r"Average time per batch\s*([\d.eE+\-]+)\s*s", text)
        ]

    real_time = [
        float(value) for value in re.findall(r"Real time:\s*([\d.eE+\-]+)\s*s", text)
    ]
    epoch_time = [
        float(value) * 60.0
        for value in re.findall(
            r"Epoch\s+\d+\s+average loss[^\n]*time \[min\]:\s*([\d.eE+\-]+)",
            text,
        )
    ]
    try:
        steps = int(re.search(r"0: E\d+-S\d+/(\d+) -", text)[1])
        step_print_outs = [int(x) for x in re.findall(r"0: E\d+-S(\d+)/\d+ -", text)]
        assert steps > 0
    except Exception as _:
        steps = int(re.search(r"0: \d+, after Step \d+/(\d+)\.", text)[1])
        step_print_outs = [
            int(step) for step in re.findall(r"0: \d+, after Step (\d+)/\d+\.", text)
        ]

    if not average_time_per_batch:
        print(f"  [skip] missing timing metrics in {path.name}")
        return None

    filtered_avg_time_per_batch = [
        value
        for key, value in zip(step_print_outs, average_time_per_batch)
        if key != min(step_print_outs)
    ]
    filtered_real_time_per_batch = None
    if real_time:
        filtered_real_time_per_batch = [
            value
            for key, value in zip(step_print_outs, real_time)
            if key != min(step_print_outs)
        ]

    avg_batch_mean, avg_batch_std = summarize(filtered_avg_time_per_batch)
    real_time_mean, real_time_std = (
        summarize(filtered_real_time_per_batch)
        if filtered_real_time_per_batch
        else (None, None)
    )
    epoch_time_mean, epoch_time_std = (
        summarize(epoch_time) if epoch_time else (None, None)
    )

    dtp_size = mesh_dims[2] * mesh_dims[3]
    peak_tflops_gpu_base = flops / dtp_size

    peak_tflops_gpu = peak_tflops_gpu_base / avg_batch_mean
    peak_tflops_gpu_std = ratio_std(peak_tflops_gpu, avg_batch_mean, avg_batch_std)
    peak_tflops_node = peak_tflops_gpu * (gpus // nodes)
    peak_tflops_node_std = ratio_std(peak_tflops_node, avg_batch_mean, avg_batch_std)
    peak_tflops = peak_tflops_gpu * gpus
    peak_tflops_std = ratio_std(peak_tflops, avg_batch_mean, avg_batch_std)

    if real_time_mean:
        real_tflops_gpu = flops / real_time_mean / dtp_size
        real_tflops = real_tflops_gpu * gpus
        real_tflops_std = ratio_std(real_tflops, real_time_mean, real_time_std)
    else:
        real_tflops_gpu = None
        real_tflops = None
        real_tflops_std = None

    if epoch_time_mean:
        sustained_tflops_gpu = flops / (epoch_time_mean / steps) / dtp_size
        sustained_tflops = sustained_tflops_gpu * gpus
        sustained_tflops_std = ratio_std(
            sustained_tflops, epoch_time_mean, epoch_time_std
        )
    else:
        sustained_tflops_gpu = None
        sustained_tflops = None
        sustained_tflops_std = None

    images_s = mesh_dims[0] / avg_batch_mean
    samples_s = mesh_dims[0] * mesh_dims[1] * 2 / avg_batch_mean
    images_s_std = ratio_std(images_s, avg_batch_mean, avg_batch_std)
    samples_s_std = ratio_std(samples_s, avg_batch_mean, avg_batch_std)

    record: dict[str, Any] = {
        "log_file": path.name,
        "job_id": job_id,
        "nodes": nodes,
        "gpus": gpus,
        "flops": flops,
        "steps": steps,
        "average_time_per_batch_mean_s": avg_batch_mean,
        "average_time_per_batch_std_s": avg_batch_std,
        "real_time_mean_s": real_time_mean,
        "real_time_std_s": real_time_std,
        "epoch_time_mean_s": epoch_time_mean,
        "epoch_time_std_s": epoch_time_std,
        "peak_tflops_gpu": peak_tflops_gpu,
        "peak_tflops_gpu_std": peak_tflops_gpu_std,
        "peak_tflops_node": peak_tflops_node,
        "peak_tflops_node_std": peak_tflops_node_std,
        "peak_tflops": peak_tflops,
        "peak_tflops_std": peak_tflops_std,
        "real_tflops_gpu": real_tflops_gpu,
        "real_tflops": real_tflops,
        "real_tflops_std": real_tflops_std,
        "sustained_tflops_gpu": sustained_tflops_gpu,
        "sustained_tflops": sustained_tflops,
        "sustained_tflops_std": sustained_tflops_std,
        "images_s": images_s,
        "images_s_std": images_s_std,
        "samples_s": samples_s,
        "samples_s_std": samples_s_std,
    }

    for index, dim in enumerate(mesh_dims):
        record[f"mesh_dim_{index}"] = dim

    return record


def default_csv_path(logs_dir: str) -> Path:
    path = Path(logs_dir)
    if path.is_dir():
        return path.parent / f"{path.name}.csv"
    return path.parent / "results.csv"


def hide_top_right_spines(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def format_tick(value: float, _position: int) -> str:
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{value:g}"


def set_integer_log_ticks(ax: plt.Axes, x_values: pd.Series, *, base: int) -> None:
    ticks = sorted({float(value) for value in x_values.tolist() if float(value) > 0})
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(format_tick))


def add_shaded_band(
    ax: plt.Axes,
    x: pd.Series,
    y: pd.Series,
    yerr: pd.Series,
    *,
    color,
    label: str | None = None,
) -> None:
    ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)
    ax.plot(x, y, "-o", color=color, linewidth=1.5, markersize=4, label=label)


def plot_series(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    yerr_col: str,
    out_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    reference_y: float | None = None,
) -> None:
    init_matplotlib()
    plt.rcParams.update({"text.usetex": False})
    fig, ax = plt.subplots(figsize=set_size(width="thesis", fraction=0.9))
    ordered = df.sort_values(x_col)

    add_shaded_band(
        ax,
        ordered[x_col],
        ordered[y_col],
        ordered[yerr_col],
        color="C0",
    )
    if reference_y is not None:
        ax.axhline(
            reference_y,
            color="0.4",
            linestyle="--",
            linewidth=1.2,
            label="Perfect scaling",
        )
    ax.set_xscale("log", base=2)
    set_integer_log_ticks(ax, ordered[x_col], base=2)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    hide_top_right_spines(ax)
    ax.grid(True, axis="both", alpha=0.25)
    if reference_y is not None:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def plot_comparison(
    df: pd.DataFrame,
    x_col: str,
    metrics: list[tuple[str, str, str]],
    out_path: Path,
    title: str,
    x_label: str,
    reference_x: pd.Series | None = None,
    reference_y: pd.Series | None = None,
    y_scale: float = 1.0,
    y_label: str = "TFLOPs/s",
    y_base: float | None = None,
) -> None:
    init_matplotlib()
    plt.rcParams.update({"text.usetex": False})
    fig, ax = plt.subplots(figsize=set_size(width="thesis", fraction=0.9))
    ordered = df.sort_values(x_col)

    for index, (y_col, yerr_col, label) in enumerate(metrics):
        add_shaded_band(
            ax,
            ordered[x_col],
            ordered[y_col] / y_scale,
            ordered[yerr_col] / y_scale,
            color=f"C{index}",
            label=label,
        )

    if reference_x is not None and reference_y is not None:
        ax.plot(
            reference_x,
            reference_y / y_scale,
            linestyle="--",
            color="0.4",
            linewidth=1.2,
            label="Perfect scaling",
        )

    ax.set_xscale("log", base=2)
    set_integer_log_ticks(ax, ordered[x_col], base=2)
    if y_base is not None:
        ax.set_yscale("log", base=y_base)
        ax.yaxis.set_major_locator(LogLocator(base=y_base))
        ax.yaxis.set_major_formatter(FuncFormatter(format_tick))
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    hide_top_right_spines(ax)
    ax.grid(True, axis="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def write_plots(df: pd.DataFrame, csv_out: Path) -> None:
    output_dir = csv_out.parent
    weak_ordered = df.sort_values("gpus")
    weak_reference = float(weak_ordered["peak_tflops_gpu"].iloc[0])
    plot_series(
        df,
        x_col="gpus",
        y_col="peak_tflops_gpu",
        yerr_col="peak_tflops_gpu_std",
        out_path=output_dir / "up_weak_scaling.pdf",
        title="Up Weak Scaling",
        x_label="GPUs",
        y_label="Peak TFLOPs / GPU",
        reference_y=weak_reference,
    )
    strong_ordered = df.sort_values("nodes")
    strong_x = strong_ordered["nodes"]
    strong_reference = (
        strong_ordered["peak_tflops"].iloc[0] * strong_x / strong_x.iloc[0]
    )
    plot_comparison(
        df,
        x_col="nodes",
        metrics=[
            ("peak_tflops", "peak_tflops_std", "peak_tflops"),
            ("real_tflops", "real_tflops_std", "real_tflops"),
            ("sustained_tflops", "sustained_tflops_std", "sustained_tflops"),
        ],
        out_path=output_dir / "dp_strong_scaling.pdf",
        title="DP Strong Scaling",
        x_label="Nodes",
        reference_x=strong_x,
        reference_y=strong_reference,
        y_scale=1000.0,
        y_label="Peta FLOP/s",
        y_base=10,
    )
    plot_series(
        df,
        x_col="mesh_dim_0",
        y_col="images_s",
        yerr_col="images_s_std",
        out_path=output_dir / "images_s.pdf",
        title="Images per Second",
        x_label="mesh_dim0",
        y_label="images/s",
    )
    plot_series(
        df,
        x_col="mesh_dim_1",
        y_col="samples_s",
        yerr_col="samples_s_std",
        out_path=output_dir / "samples_s.pdf",
        title="Samples per Second",
        x_label="mesh_dim1",
        y_label="samples/s",
    )


def main(
    logs_dir: str,
    flops: float,
    out_path: str | None = None,
):
    paths = collect_paths(logs_dir)
    if not paths:
        print("No .out files found.")
        sys.exit(1)

    print(f"Parsing {len(paths)} log file(s)...\n")
    records = []
    for path in paths:
        record = parse_log(path, flops)
        if record is not None:
            records.append(record)

    if not records:
        print("No valid records parsed.")
        sys.exit(1)

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(by="nodes")
    print(df)

    csv_out = Path(out_path) if out_path else default_csv_path(logs_dir)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)
    write_plots(df, csv_out)

    print(f"Wrote {len(df)} row(s) to {csv_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse weak-scaling .out logs and write a CSV summary."
    )
    parser.add_argument(
        "--logs_dir",
        required=True,
        help="Directory of *.out log files, or a glob pattern.",
    )
    parser.add_argument(
        "--flops",
        required=True,
        type=float,
        help="Flops value used to populate the summary table.",
    )
    parser.add_argument(
        "--out_path",
        default=None,
        help="Optional CSV output path. Defaults to one CSV next to the log folder.",
    )
    args = parser.parse_args()
    main(args.logs_dir, args.flops, out_path=args.out_path)
