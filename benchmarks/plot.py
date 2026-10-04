#!/usr/bin/env python3
"""Строит графики из CSV, сгенерированного benchmarks/bench.c (см. `make benchmark`).

Три больших грида (small multiples, 3x5 = 15 фильтров):
- ускорение параллельных стратегий относительно буквально-последовательной версии;
- время vs число воркеров;
- время vs размер очереди.
"""

import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STRATEGIES = [
    "Pixelwise", "By Rows", "By Cols", "Blocks 32x32", "Blocks 64x64",
    "Blocks 128x128",
]

BASELINE = "Baseline"

FILTERS = [
    "blur3x3", "blur5x5", "gaussian3x3", "gaussian5x5", "motionblur",
    "findedges1", "findedges2", "findedges3", "findedges4",
    "sharpen1", "sharpen2", "sharpen3", "emboss1", "emboss2", "identity",
]

WORKERS = [1, 4, 8, 16]
QUEUE_CAPACITIES = [1, 2, 4, 8, 16, 32, 64]
DEFAULT_CAPACITY = 10

METRIC = "min_ms"

NROWS = 3
NCOLS = 5


def load(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["workers"] = int(row["workers"])
            row["queue_capacity"] = int(row["queue_capacity"])
            row["num_images"] = int(row["num_images"])
            row["min_ms"] = float(row["min_ms"])
            row["mean_ms"] = float(row["mean_ms"])
            row["median_ms"] = float(row["median_ms"])
            rows.append(row)
    return rows


def _baseline_time(rows, filt):
    for r in rows:
        if r["filter"] == filt and r["strategy"] == BASELINE:
            return r[METRIC]
    return None


def _workers_data(rows, filt):
    out = defaultdict(dict)
    for r in rows:
        if (r["filter"] == filt and r["strategy"] != BASELINE and
                r["queue_capacity"] == DEFAULT_CAPACITY):
            out[r["strategy"]][r["workers"]] = r[METRIC]
    return out


def _queue_data(rows, filt):
    out = defaultdict(dict)
    for r in rows:
        if (r["filter"] == filt and r["strategy"] != BASELINE and
                r["queue_capacity"] in QUEUE_CAPACITIES):
            out[r["strategy"]][r["queue_capacity"]] = r[METRIC]
    return out


def _speedup_data(rows, filt):
    base = _baseline_time(rows, filt)
    data = _workers_data(rows, filt)
    if not base:
        return [], []

    strategies, speedups = [], []
    for s in STRATEGIES:
        vals = [v for v in data.get(s, {}).values() if v > 0]
        if not vals:
            continue
        strategies.append(s)
        speedups.append(base / (sum(vals) / len(vals)))
    return strategies, speedups


def _new_grid(suptitle):
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(22, 11), squeeze=False)
    fig.suptitle(suptitle, fontsize=14)
    return fig, axes


def _shared_legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   fontsize=8)


def _label_edges(axes, ylabel, xlabel):
    nrows, ncols = axes.shape
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == nrows - 1:
                ax.set_xlabel(xlabel)
            if r != nrows - 1:
                ax.tick_params(axis="x", labelbottom=False)


def _save(fig, outdir, base):
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(os.path.join(outdir, base + ".png"), dpi=200)
    fig.savefig(os.path.join(outdir, base + ".svg"))
    plt.close(fig)


def plot_speedup_grid(rows, outdir):
    fig, axes = _new_grid("Ускорение vs последовательная")

    for ax, filt in zip(axes.flat, FILTERS):
        strategies, speedups = _speedup_data(rows, filt)
        if not strategies:
            ax.set_visible(False)
            continue

        bars = ax.bar(strategies, speedups, color="#4C72B0")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(filt, fontsize=9)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        for b, v in zip(bars, speedups):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}x",
                    ha="center", va="bottom", fontsize=6)

    _label_edges(axes, "Ускорение, x", "Стратегия")
    _save(fig, outdir, "speedup_all")


def plot_time_vs_workers_grid(rows, outdir):
    fig, axes = _new_grid("Время vs воркеры")

    for ax, filt in zip(axes.flat, FILTERS):
        data = _workers_data(rows, filt)
        if not data:
            ax.set_visible(False)
            continue

        for s in STRATEGIES:
            by_workers = data.get(s, {})
            if not by_workers:
                continue
            xs = [w for w in WORKERS if w in by_workers]
            ax.plot(xs, [by_workers[w] for w in xs], marker="o", markersize=2,
                    label=s)

        ax.set_xticks(WORKERS)
        ax.set_title(filt, fontsize=9)
        ax.tick_params(labelsize=7)

    _shared_legend(fig, axes[0, 0])
    _label_edges(axes, "Время, мс", "Воркеры")
    _save(fig, outdir, "time_vs_workers_all")


def plot_time_vs_queue_grid(rows, outdir):
    fig, axes = _new_grid("Время vs очередь")

    for ax, filt in zip(axes.flat, FILTERS):
        data = _queue_data(rows, filt)
        if not data:
            ax.set_visible(False)
            continue

        for s in STRATEGIES:
            by_capacity = data.get(s, {})
            if not by_capacity:
                continue
            xs = [c for c in QUEUE_CAPACITIES if c in by_capacity]
            ax.plot(xs, [by_capacity[c] for c in xs], marker="o", markersize=2,
                    label=s)

        ax.set_xticks(QUEUE_CAPACITIES)
        ax.set_title(filt, fontsize=9)
        ax.tick_params(labelsize=7)

    _shared_legend(fig, axes[0, 0])
    _label_edges(axes, "Время, мс", "Размер очереди")
    _save(fig, outdir, "time_vs_queue_all")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/generated/bench_results.csv"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "benchmarks/generated"

    os.makedirs(outdir, exist_ok=True)
    rows = load(csv_path)
    if not rows:
        print(f"No data in {csv_path}", file=sys.stderr)
        return 1

    plot_speedup_grid(rows, outdir)
    plot_time_vs_workers_grid(rows, outdir)
    plot_time_vs_queue_grid(rows, outdir)

    print(f"Wrote plots to {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
