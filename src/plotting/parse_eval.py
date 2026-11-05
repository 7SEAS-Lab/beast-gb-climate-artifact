#!/usr/bin/env python3
"""
parse_eval.py - Parse evaluation run log files and generate comparison plots.

Usage:
    python -m src.plotting.parse_eval --logs_dir logs/eval/ --out_dir results/eval_plots

What it does
------------
1. Parses every ``*.out`` log in ``--logs_dir``. Each log contains, near the top,
   the mesh dimensions (``mesh_dims=a,b,c,d``) and the total number of nodes /
   GPUs (``SLURM_NNODES`` / ``SLURM_NTASKS``). Near the bottom it contains blocks
   of per-feature metrics laid out as::

       <metric>_<time>h:                # header line, e.g. "rmse_204h:"
       <feature>_<time>h: <value>       # e.g. "W_200_204h: 0.708"
       ...

   Supported metrics: mse, rmse, crps, spread_skill_ratio.

2. Generates comparison plots so you can:
   * see rollout stability (how fast metrics deteriorate with lead time),
   * spot the best-performing run (each run == one checkpoint + config),
   * spot which features a run is best/worst at.

   Three families of plots are produced:
   * Overall score vs lead time (one figure per metric), aggregating over
     features. For ``spread_skill_ratio`` the aggregate is the mean distance
     from the ideal value 1.0; for error metrics it is the mean of the
     per-feature values normalised by the cross-run mean (so features on very
     different scales, e.g. Z vs Q, contribute comparably).
   * Per-feature score vs lead time (one figure per metric+feature), plotting
     the raw values with one line per run.
   * A per-feature ranking heatmap (one per metric) at the final lead time.
"""

import argparse
import glob
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for cluster / CI
import matplotlib.pyplot as plt
import pandas as pd

try:  # optional shared styling; degrade gracefully if unavailable
    from src.plotting.utils import KIT_PALETTE
except Exception:  # pragma: no cover - fallback when run standalone
    KIT_PALETTE = None


# Metrics that appear as their own header line before a block of features.
KNOWN_METRICS = ("mse", "rmse", "crps", "spread_skill_ratio")

# Ideal target value per metric, used when aggregating an "overall" score.
METRIC_IDEAL: dict[str, float] = {
    "mse": 0.0,
    "rmse": 0.0,
    "crps": 0.0,
    "spread_skill_ratio": 1.0,
}

# Strip the leading rank prefix that srun adds, e.g. "   0: " or " 695: ".
_RANK_PREFIX_RE = re.compile(r"^\s*\d+:\s*")

# A metric header line, e.g. "rmse_204h:" or "spread_skill_ratio_12h:".
_METRIC_HEADER_RE = re.compile(
    r"^(?P<metric>" + "|".join(KNOWN_METRICS) + r")_(?P<time>\d+)h:\s*$"
)

# A feature value line, e.g. "W_200_204h: 0.708" or "Q_50_12h: 1.2e-07".
_FEATURE_RE = re.compile(
    r"^(?P<feature>[A-Za-z][A-Za-z0-9_]*?)_(?P<time>\d+)h:\s*"
    r"(?P<value>[-+]?(?:inf|nan|\d+\.?\d*(?:[eE][-+]?\d+)?))\s*$"
)


def collect_paths(path: str) -> list[Path]:
    """Return the list of log files for a directory or glob pattern."""
    p = Path(path)
    if p.is_dir():
        return sorted(p.glob("*.out"))
    return sorted(Path(match) for match in glob.glob(path))


def _run_label(path: Path) -> str:
    """Human-friendly, unique-ish label for a run derived from its file name.

    File names encode the run identity, e.g.
    ``eval-2022-N128-r10_subs6_gss32_e7s68k_1379677.out``.
    We drop the ``eval-`` prefix, the trailing job id and the extension.
    """
    stem = path.stem
    stem = re.sub(r"^eval-", "", stem)
    stem = re.sub(r"_\d+$", "", stem)  # trailing slurm job id
    return stem or path.stem


def parse_eval_log(path: Path) -> list[dict[str, Any]]:
    """Parse a single evaluation log into a list of metric records."""
    text = path.read_text(errors="replace")

    # ── metadata (top of file) ──────────────────────────────────────────────
    job_match = re.search(r"SLURM_JOB_ID:\s*(\d+)", text)
    nodes_match = re.search(r"SLURM_NNODES:\s*(\d+)", text)
    tasks_match = re.search(r"SLURM_NTASKS:\s*(\d+)", text)
    mesh_match = re.search(r"mesh_dims=([\d,\-]+)", text)
    rollout_match = re.search(r"--rollout\s+(\d+)", text)

    job_id = int(job_match.group(1)) if job_match else None
    nodes = int(nodes_match.group(1)) if nodes_match else None
    gpus = int(tasks_match.group(1)) if tasks_match else None
    mesh_dims = [int(d) for d in mesh_match.group(1).split(",")] if mesh_match else None
    mesh_str = ",".join(map(str, mesh_dims)) if mesh_dims else None
    rollout = int(rollout_match.group(1)) if rollout_match else None

    meta = {
        "log_file": path.name,
        "run": _run_label(path),
        "job_id": job_id,
        "nodes": nodes,
        "gpus": gpus,
        "mesh_dims": mesh_str,
        "rollout": rollout,
    }

    # ── metric blocks (bottom of file) ──────────────────────────────────────
    records: list[dict[str, Any]] = []
    current_metric: str | None = None
    current_time: int | None = None

    for raw_line in text.splitlines():
        line = _RANK_PREFIX_RE.sub("", raw_line).strip()
        if not line:
            continue

        header = _METRIC_HEADER_RE.match(line)
        if header:
            current_metric = header.group("metric")
            current_time = int(header.group("time"))
            continue

        if current_metric is None:
            continue

        m = _FEATURE_RE.match(line)
        if not m:
            # Any non-feature line ends the current block.
            current_metric = None
            current_time = None
            continue

        feature = m.group("feature")
        # Skip lines where a metric name is (re)used as a feature token.
        if feature in KNOWN_METRICS:
            continue

        feat_time = int(m.group("time"))
        value = float(m.group("value"))

        rec = dict(meta)
        rec.update(
            {
                "metric": current_metric,
                "time_h": current_time if current_time is not None else feat_time,
                "feature": feature,
                "value": value,
            }
        )
        records.append(rec)

    return records


def aggregate_overall(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-feature values into one overall score per (run, metric, time).

    Aggregation rules
    -----------------
    * ``spread_skill_ratio``: mean over features of ``|value - 1|`` (distance
      from the ideal calibration of 1.0). Lower is better.
    * error metrics (mse, rmse, crps): features live on wildly different scales
      (Z ~ 1e2, Q ~ 1e-7), so a raw sum would be dominated by a few features.
      We normalise each (metric, feature, time) value by the mean of that same
      quantity across all runs, then average over features. Lower is better and
      1.0 == "average run". This keeps every feature comparable.
    """
    work = df.copy()
    # Drop non-finite values (e.g. inf in spread_skill_ratio for sst).
    work = work[work["value"].apply(math.isfinite)]

    # Normalisation baseline for error metrics: cross-run mean per cell.
    baseline = work.groupby(["metric", "feature", "time_h"])["value"].transform("mean")

    def _score(row_value: float, metric: str, base: float) -> float:
        if metric == "spread_skill_ratio":
            return abs(row_value - METRIC_IDEAL[metric])
        if base and math.isfinite(base):
            return row_value / base
        return float("nan")

    work = work.assign(baseline=baseline)
    work["score"] = [
        _score(v, m, b)
        for v, m, b in zip(work["value"], work["metric"], work["baseline"])
    ]

    grouped = (
        work.groupby(["run", "log_file", "metric", "time_h"], dropna=False)["score"]
        .mean()
        .reset_index(name="overall_score")
    )
    return grouped


def _colors(n: int) -> list:
    if KIT_PALETTE:
        base = list(KIT_PALETTE)
    else:
        base = list(plt.get_cmap("tab10").colors)
    if n <= len(base):
        return base[:n]
    cmap = plt.get_cmap("turbo")
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))


def plot_overall_scores(agg: pd.DataFrame, out_dir: Path) -> None:
    """One figure per metric: overall score vs lead time, one line per run."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in sorted(agg["metric"].unique()):
        by_metric = agg[agg["metric"] == metric]
        pivot = by_metric.pivot_table(
            index="time_h", columns="run", values="overall_score"
        ).sort_index()

        runs = list(pivot.columns)
        colors = _colors(len(runs))

        fig, ax = plt.subplots(figsize=(8, 5))
        for run, color in zip(runs, colors):
            ax.plot(pivot.index, pivot[run], marker="o", ms=4, label=run, color=color)

        ax.set_xlabel("Lead time (h)")
        if metric == "spread_skill_ratio":
            ax.set_ylabel("Mean |spread/skill - 1|  (0 = ideal)")
        else:
            ax.set_ylabel(f"Overall {metric} (normalised, 1 = avg run)")
        ax.set_title(f"Overall score by lead time — {metric}")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"overall_{metric}.png", dpi=200)
        fig.savefig(out_dir / f"overall_{metric}.pdf")
        plt.close(fig)


def plot_per_feature(df: pd.DataFrame, out_dir: Path) -> None:
    """One figure per (metric, feature): raw value vs lead time, one line per run."""
    out_dir.mkdir(parents=True, exist_ok=True)

    finite = df[df["value"].apply(math.isfinite)]
    for metric in sorted(finite["metric"].unique()):
        mdf = finite[finite["metric"] == metric]
        metric_dir = out_dir / metric
        metric_dir.mkdir(parents=True, exist_ok=True)

        for feature in sorted(mdf["feature"].unique()):
            sub = mdf[mdf["feature"] == feature]
            if sub.empty:
                continue
            pivot = sub.pivot_table(
                index="time_h", columns="run", values="value"
            ).sort_index()

            runs = list(pivot.columns)
            colors = _colors(len(runs))

            fig, ax = plt.subplots(figsize=(7, 4.5))
            for run, color in zip(runs, colors):
                ax.plot(
                    pivot.index, pivot[run], marker="o", ms=4, label=run, color=color
                )

            ax.set_xlabel("Lead time (h)")
            ax.set_ylabel(f"{metric} — {feature}")
            if metric == "spread_skill_ratio":
                ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7)
            ax.set_title(f"{metric} — {feature}")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False, fontsize=7, loc="best")
            fig.tight_layout()
            fname = f"{metric}_{_sanitize(feature)}"
            fig.savefig(metric_dir / f"{fname}.png", dpi=150)
            plt.close(fig)


def plot_feature_ranking(df: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap per metric: normalised score per (run, feature) at the final lead time.

    Helps answer "on which features do we perform best?". Values are normalised
    per feature by the cross-run mean so features are comparable; lower is better
    (or closer to 0 for spread_skill_ratio, shown as |x-1|).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    finite = df[df["value"].apply(math.isfinite)]

    for metric in sorted(finite["metric"].unique()):
        mdf = finite[finite["metric"] == metric]
        final_t = mdf["time_h"].max()  # largest available lead time
        cell = mdf[mdf["time_h"] == final_t]
        if cell.empty:
            continue

        pivot = cell.pivot_table(index="run", columns="feature", values="value")
        if metric == "spread_skill_ratio":
            norm = (pivot - 1.0).abs()
            cbar_label = f"|{metric} - 1| @ {final_t}h (0 = ideal)"
        else:
            norm = pivot / pivot.mean(axis=0)
            cbar_label = f"{metric} @ {final_t}h (norm, 1 = avg run)"

        fig, ax = plt.subplots(
            figsize=(max(8, 0.28 * norm.shape[1]), max(3, 0.5 * norm.shape[0] + 1))
        )
        im = ax.imshow(norm.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(norm.shape[1]))
        ax.set_xticklabels(norm.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(norm.shape[0]))
        ax.set_yticklabels(norm.index, fontsize=7)
        ax.set_title(f"Per-feature ranking — {metric} @ {final_t}h")
        fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.02, pad=0.01)
        fig.tight_layout()
        fig.savefig(out_dir / f"ranking_{metric}.png", dpi=150)
        plt.close(fig)


def main(logs_dir: str, out_dir: str | None = None) -> None:
    paths = collect_paths(logs_dir)
    if not paths:
        print(f"No .out files found in {logs_dir!r}.")
        return

    all_records: list[dict[str, Any]] = []
    for path in paths:
        recs = parse_eval_log(path)
        if recs:
            all_records.extend(recs)
            print(f"[ok]   {path.name}: {len(recs)} metric values")
        else:
            print(f"[skip] {path.name}: no metrics parsed")

    if not all_records:
        print("No metrics found in any log files.")
        return

    df = pd.DataFrame.from_records(all_records)

    out = Path(out_dir) if out_dir else Path("results/eval_plots")
    out.mkdir(parents=True, exist_ok=True)

    df.to_csv(out / "parsed_metrics.csv", index=False)

    agg = aggregate_overall(df)
    agg.to_csv(out / "overall_scores.csv", index=False)

    plot_overall_scores(agg, out / "overall")
    plot_per_feature(df, out / "per_feature")
    plot_feature_ranking(df, out / "ranking")

    n_runs = df["run"].nunique()
    print(f"\nParsed {n_runs} runs, {len(df)} values. Wrote CSVs and plots to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse evaluation logs and generate comparison plots."
    )
    parser.add_argument(
        "--logs_dir",
        required=True,
        help="Directory of *.out log files, or a glob pattern.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for CSVs and plots (default: results/eval_plots).",
    )
    args = parser.parse_args()
    main(args.logs_dir, args.out_dir)
