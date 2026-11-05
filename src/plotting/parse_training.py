#!/usr/bin/env python3
"""
parse_training.py - Parse training ``.out`` logs and plot the training metrics
(loss, mse, data fitting, prior matching) over the global training step.

Each metric line in the logs looks like::

    0: E4-S69010/89835 - loss: -1.57e+14, mse: VIReturn(0.0201, device='cuda:0', ...).
       data fitting -2.36e+13 prior matching 5.87e+08 Average time per batch ...

From this we extract:
    * epoch          (E4)
    * step-in-epoch  (S69010) and steps-per-epoch (89835)
    * loss
    * mse            (the float inside ``VIReturn(...)``)
    * data fitting
    * prior matching

A single continuous global step is reconstructed as ``epoch * steps_per_epoch +
step_in_epoch`` so that runs which restart from checkpoints line up. Vertical
lines mark the epoch boundaries.

Usage:
    python -m src.plotting.parse_training --logs_dir /path/to/logs/training_cont
    python -m src.plotting.parse_training --logs_dir <dir> --out_path metrics.png --csv metrics.csv
"""

import argparse
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------------------------------- #
# Regex for a single metric line.
# Example:
#   0: E4-S69010/89835 - loss: -157431657136128.0, mse: VIReturn(0.0201, device=
#   'cuda:0', grad_fn=<AliasBackward0>). data fitting -23699694551040.0 prior
#   matching 587848960.0 Average time per batch 2.085 s, Real time: 2.29 s
# --------------------------------------------------------------------------- #
_FLOAT = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

LINE_RE = re.compile(
    rf"E(?P<epoch>\d+)-S(?P<step>\d+)/(?P<steps_per_epoch>\d+)"
    rf".*?loss:\s*(?P<loss>{_FLOAT})"
    rf".*?VIReturn\(\s*(?P<mse>{_FLOAT})"
    rf".*?data\s+fitting\s*(?P<data_fitting>{_FLOAT})"
    rf".*?prior\s+matching\s*(?P<prior_matching>{_FLOAT})",
    re.IGNORECASE,
)

METRICS = ["loss", "mse", "data_fitting", "prior_matching"]
METRIC_LABELS = {
    "loss": "Loss",
    "mse": "MSE",
    "data_fitting": "Data fitting",
    "prior_matching": "Prior matching",
}


def collect_paths(path: str) -> list[Path]:
    """Return the sorted list of ``.out`` files for a directory or glob."""
    p = Path(path)
    if p.is_dir():
        return sorted(p.glob("*.out"))
    return sorted(Path(match) for match in glob.glob(path))


def parse_file(path: Path) -> list[dict]:
    """Parse a single ``.out`` file into a list of per-step metric records."""
    records: list[dict] = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if "VIReturn(" not in line:
                continue
            m = LINE_RE.search(line)
            if not m:
                continue
            epoch = int(m.group("epoch"))
            step = int(m.group("step"))
            steps_per_epoch = int(m.group("steps_per_epoch"))
            records.append(
                {
                    "file": path.name,
                    "epoch": epoch,
                    "step_in_epoch": step,
                    "steps_per_epoch": steps_per_epoch,
                    # global, continuous step across epochs
                    "global_step": epoch * steps_per_epoch + step,
                    "loss": float(m.group("loss")),
                    "mse": float(m.group("mse")),
                    "data_fitting": float(m.group("data_fitting")),
                    "prior_matching": float(m.group("prior_matching")),
                }
            )
    return records


def parse_logs(paths: list[Path]) -> pd.DataFrame:
    """Parse all files and return a combined, step-sorted, de-duplicated frame."""
    records: list[dict] = []
    for path in paths:
        recs = parse_file(path)
        print(f"  {path.name}: {len(recs)} metric lines")
        records.extend(recs)

    if not records:
        raise SystemExit("No metric lines found in the provided logs.")

    df = pd.DataFrame.from_records(records)
    # Multiple restarts can log the same global step; keep the last occurrence.
    df = df.sort_values("global_step")
    df = df.drop_duplicates(subset="global_step", keep="last")
    df = df.reset_index(drop=True)
    return df


def epoch_boundaries(df: pd.DataFrame) -> list[int]:
    """Global-step positions where a new epoch begins."""
    spe = int(df["steps_per_epoch"].mode().iloc[0])
    first = int(df["epoch"].min())
    last = int(df["epoch"].max())
    return [e * spe for e in range(first, last + 2)]


def plot_metrics(df: pd.DataFrame, out_path: Path, logscale: bool) -> None:
    """Plot every metric in its own subplot with epoch boundary lines."""
    boundaries = epoch_boundaries(df)

    fig, axes = plt.subplots(
        len(METRICS), 1, figsize=(11, 3 * len(METRICS)), sharex=True
    )
    x = df["global_step"].to_numpy()

    for ax, metric in zip(axes, METRICS):
        ax.plot(x, df[metric].to_numpy(), lw=0.8, color="C0")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(True, alpha=0.3)
        if logscale and (df[metric] > 0).all():
            ax.set_yscale("log")

        # Epoch boundary markers.
        for i, b in enumerate(boundaries):
            if x.min() <= b <= x.max():
                ax.axvline(b, color="0.6", ls="--", lw=0.7)
                if ax is axes[0]:
                    ax.text(
                        b,
                        1.01,
                        f"E{boundaries.index(b) + int(df['epoch'].min())}",
                        transform=ax.get_xaxis_transform(),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="0.4",
                    )

    axes[-1].set_xlabel("Global training step")
    axes[0].set_title("Training metrics over time")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs_dir",
        required=True,
        help="Directory containing .out files (or a glob pattern).",
    )
    parser.add_argument(
        "--out_path",
        default=None,
        help="Output image path. Defaults to <logs_dir>/training_metrics.png.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional path to also dump the parsed metrics as CSV.",
    )
    parser.add_argument(
        "--logscale",
        action="store_true",
        help="Use a log y-axis for metrics that are strictly positive.",
    )
    args = parser.parse_args()

    paths = collect_paths(args.logs_dir)
    if not paths:
        raise SystemExit(f"No .out files found under {args.logs_dir}")
    print(f"Parsing {len(paths)} log file(s)...")

    df = parse_logs(paths)
    print(
        f"Parsed {len(df)} unique steps "
        f"(epochs {int(df['epoch'].min())}-{int(df['epoch'].max())})."
    )

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Wrote CSV to {args.csv}")

    logs_dir = Path(args.logs_dir)
    default_dir = logs_dir if logs_dir.is_dir() else logs_dir.parent
    out_path = (
        Path(args.out_path) if args.out_path else default_dir / "training_metrics.png"
    )
    plot_metrics(df, out_path, args.logscale)


if __name__ == "__main__":
    main()
