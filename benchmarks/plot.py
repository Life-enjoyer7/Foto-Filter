#!/usr/bin/env python3
"""Строит графики из CSV, сгенерированного benchmarks/bench.c (см. `make benchmark`)."""

import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STRATEGIES = ["Sequential", "Pixelwise", "By Rows", "By Cols", "Blocks 32x32", "Blocks 64x64", "Blocks 128x128"]


def load(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["workers"] = int(row["workers"])
            row["num_images"] = int(row["num_images"])
            row["min_ms"] = float(row["min_ms"])
            row["mean_ms"] = float(row["mean_ms"])
            row["median_ms"] = float(row["median_ms"])
            rows.append(row)
    return rows


def plot_speedup_per_filter(rows, outdir):
    """Для каждого фильтра - среднее (по всем конфигурациям потоков) ускорение
    каждой параллельной стратегии относительно Sequential."""
    by_filter = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        by_filter[r["filter"]][r["strategy"]][r["workers"]] = r["median_ms"]

    for filt, by_strategy in by_filter.items():
        seq = by_strategy.get("Sequential")
        if not seq:
            continue

        strategies, speedups = [], []
        for s in STRATEGIES:
            if s == "Sequential" or s not in by_strategy:
                continue
            ratios = [seq[w] / t for w, t in by_strategy[s].items() if w in seq and t > 0]
            if not ratios:
                continue
            strategies.append(s)
            speedups.append(sum(ratios) / len(ratios))

        if not strategies:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(strategies, speedups, color="#4C72B0")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel("Среднее ускорение относительно Sequential")
        ax.set_title(f"Ускорение по стратегиям — {filt}")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        for b, v in zip(bars, speedups):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}x", ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{filt}_speedup.png"), dpi=150)
        plt.close(fig)


def plot_time_vs_workers(rows, outdir, filt):
    """Время (медиана) в зависимости от числа воркеров для каждой стратегии,
    на фиксированном фильтре - показывает, окупается ли добавление потоков."""
    subset = [r for r in rows if r["filter"] == filt]
    if not subset:
        return

    by_strategy = defaultdict(list)
    for r in subset:
        by_strategy[r["strategy"]].append((r["workers"], r["median_ms"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    for s in STRATEGIES:
        pts = sorted(set(by_strategy.get(s, [])))
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", label=s)

    ax.set_xlabel("Число воркеров")
    ax.set_ylabel("Время, мс (медиана, весь images/)")
    ax.set_title(f"Время выполнения pipeline vs число воркеров — {filt}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{filt}_time_vs_workers.png"), dpi=150)
    plt.close(fig)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/generated/bench_results.csv"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "benchmarks/generated"

    os.makedirs(outdir, exist_ok=True)
    rows = load(csv_path)
    if not rows:
        print(f"No data in {csv_path}", file=sys.stderr)
        return 1

    # 1. Графики ускорения для всех фильтров
    plot_speedup_per_filter(rows, outdir)

    # 2. Графики время vs воркеры для всех 15 фильтров
    filters_to_plot = [
        "blur3x3",
        "blur5x5",
        "gaussian3x3",
        "gaussian5x5",
        "motionblur",
        "findedges1",
        "findedges2",
        "findedges3",
        "findedges4",
        "sharpen1",
        "sharpen2",
        "sharpen3",
        "emboss1",
        "emboss2",
        "identity"
    ]

    for filt in filters_to_plot:
        plot_time_vs_workers(rows, outdir, filt=filt)

    print(f"Wrote plots to {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
