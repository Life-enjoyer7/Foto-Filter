#!/usr/bin/env python3
"""Строит графики из CSV, сгенерированного benchmarks/bench.c (см. `make benchmark`).
"""

import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STRATEGIES = ["sequential", "pixelwise", "rows", "cols", "blocks32", "blocks64", "blocks128"]

THREAD_COUNTS = [1, 2, 4, 8, 16]

FILTERS = [
    "blur3x3", "blur5x5", "gaussian3x3", "gaussian5x5", "motionblur",
    "findedges1", "findedges2", "findedges3", "findedges4",
    "sharpen1", "sharpen2", "sharpen3", "emboss1", "emboss2", "identity",
]

NROWS = 3
NCOLS = 5


def load(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["width"] = int(row["width"])
            row["height"] = int(row["height"])
            row["filter_w"] = int(row["filter_w"])
            row["filter_h"] = int(row["filter_h"])
            row["min_ms"] = float(row["min_ms"])
            row["mean_ms"] = float(row["mean_ms"])
            row["median_ms"] = float(row["median_ms"])
            row["threads"] = int(row.get("threads") or 0)
            rows.append(row)
    return rows


def _rows_for(rows, filt):
    return [r for r in rows if r["filter"] == filt]


def _speedup_data(rows, filt):
    by_strategy = defaultdict(list)
    for r in _rows_for(rows, filt):
        by_strategy[r["strategy"]].append((r["image"], r["median_ms"]))

    seq = dict(by_strategy.get("sequential", []))
    if not seq:
        return [], []

    strategies, speedups = [], []
    for s in STRATEGIES:
        if s == "sequential" or s not in by_strategy:
            continue
        ratios = [seq[img] / t for img, t in by_strategy[s] if img in seq and t > 0]
        if not ratios:
            continue
        strategies.append(s)
        speedups.append(sum(ratios) / len(ratios))
    return strategies, speedups


def _time_vs_size_data(rows, filt):
    by_strategy = defaultdict(lambda: defaultdict(list))
    for r in _rows_for(rows, filt):
        by_strategy[r["strategy"]][r["width"] * r["height"]].append(r["median_ms"])

    out = {}
    for s in STRATEGIES:
        sizes = sorted(by_strategy.get(s, {}))
        if not sizes:
            continue
        ys = [sum(by_strategy[s][sz]) / len(by_strategy[s][sz]) for sz in sizes]
        out[s] = (sizes, ys)
    return out


def _time_vs_threads_data(rows, filt):
    by_strategy = defaultdict(lambda: defaultdict(list))
    for r in _rows_for(rows, filt):
        if r["strategy"] != "sequential":
            by_strategy[r["strategy"]][r["threads"]].append(r["median_ms"])

    out = {}
    for s in STRATEGIES:
        if s == "sequential":
            continue
        by_threads = by_strategy.get(s, {})
        ts = [t for t in THREAD_COUNTS if t in by_threads]
        if not ts:
            continue
        ys = [sum(by_threads[t]) / len(by_threads[t]) for t in ts]
        out[s] = (ts, ys)
    return out


def _new_grid(suptitle):
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(22, 11), squeeze=False)
    fig.suptitle(suptitle, fontsize=14)
    return fig, axes


def _shared_legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8)


def _label_edges(axes, ylabel, xlabel):
    nrows, ncols = axes.shape
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == nrows - 1:
                ax.set_xlabel(xlabel)
            if c != 0:
                ax.tick_params(axis="y", labelleft=False)
            if r != nrows - 1:
                ax.tick_params(axis="x", labelbottom=False)


def _save(fig, outdir, base):
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(os.path.join(outdir, base + ".png"), dpi=200)
    fig.savefig(os.path.join(outdir, base + ".svg"))
    plt.close(fig)


def plot_speedup_grid(rows, outdir):
    fig, axes = _new_grid("Ускорение")

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


def plot_time_vs_size_grid(rows, outdir):
    fig, axes = _new_grid("Время vs размер")

    for ax, filt in zip(axes.flat, FILTERS):
        data = _time_vs_size_data(rows, filt)
        if not data:
            ax.set_visible(False)
            continue

        for s in STRATEGIES:
            if s not in data:
                continue
            sizes, ys = data[s]
            ax.plot(sizes, ys, marker="o", markersize=2, label=s)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(filt, fontsize=9)
        ax.tick_params(labelsize=7)

    _shared_legend(fig, axes[0, 0])
    _label_edges(axes, "Время, мс", "Пиксели")
    _save(fig, outdir, "time_vs_size_all")


def plot_time_vs_threads_grid(rows, outdir):
    fig, axes = _new_grid("Время vs потоки")

    for ax, filt in zip(axes.flat, FILTERS):
        data = _time_vs_threads_data(rows, filt)
        if not data:
            ax.set_visible(False)
            continue

        for s in STRATEGIES:
            if s not in data:
                continue
            ts, ys = data[s]
            ax.plot(ts, ys, marker="o", markersize=2, label=s)

        ax.set_xticks(THREAD_COUNTS)
        ax.set_title(filt, fontsize=9)
        ax.tick_params(labelsize=7)

    _shared_legend(fig, axes[0, 0])
    _label_edges(axes, "Время, мс", "Потоки")
    _save(fig, outdir, "time_vs_threads_all")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/generated/bench_results.csv"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "benchmarks/generated"

    os.makedirs(outdir, exist_ok=True)
    rows = load(csv_path)
    if not rows:
        print(f"No data in {csv_path}", file=sys.stderr)
        return 1

    plot_speedup_grid(rows, outdir)
    plot_time_vs_size_grid(rows, outdir)
    plot_time_vs_threads_grid(rows, outdir)

    print(f"Wrote plots to {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
