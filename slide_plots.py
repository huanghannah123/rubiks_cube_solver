"""
Slide-sized versions of the two MCTS figures: fewer panels, bigger text,
so every label is readable when the plot is shrunk onto a slide.

    big_plots/mcts_plots_slide.png              4 panels from mcts_results.csv
    big_plots/mcts_heuristic_slide.png          2 panels from mcts_heuristic_results/

Run after make_big_plots.py and compare_mcts_heuristics.py:
    python slide_plots.py
"""
import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
OUT = BASE / "big_plots"
plt.rcParams.update({
    "font.size": 22, "axes.titlesize": 26, "axes.titleweight": "bold",
    "axes.labelsize": 23, "xtick.labelsize": 21, "ytick.labelsize": 21,
    "legend.fontsize": 21, "lines.linewidth": 4, "lines.markersize": 13,
})
BLUE = "#2e75b6"
COLORS = {"IDA*": "#1f4e79", "MCTS (sticker)": "#c0504d",
          "MCTS (PDB)": "#f0a030", "MCTS (PDB + guided)": "#3a9d5d"}


def mcts_slide():
    rows = list(csv.DictReader(open(BASE / "mcts_results.csv", newline="")))
    depths = sorted({int(r["scramble_depth"]) for r in rows})

    def per_depth(fn):
        return [fn([r for r in rows if int(r["scramble_depth"]) == d]) for d in depths]

    panels = [
        ("Success rate", "Solved (%)",
         per_depth(lambda g: 100 * sum(r["solved"] == "1" for r in g) / len(g))),
        ("Run time", "Seconds",
         per_depth(lambda g: statistics.mean(float(r["elapsed_seconds"]) for r in g))),
        ("Solution length (solved)", "Moves",
         per_depth(lambda g: statistics.mean(int(r["solution_length"]) for r in g if r["solved"] == "1"))),
        ("Tree nodes expanded", "Nodes",
         per_depth(lambda g: statistics.mean(int(r["expanded_nodes"]) for r in g))),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(17, 10.5))
    for ax, (title, ylabel, vals) in zip(axes.flat, panels):
        ax.plot(depths, vals, marker="o", color=BLUE)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Scramble length (moves)")
        ax.set_xticks(depths)
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(OUT / "mcts_plots_slide.png", dpi=150)
    plt.close(fig)


def heuristic_slide():
    rows = list(csv.DictReader(open(BASE / "mcts_heuristic_results" / "mcts_heuristic_comparison.csv", newline="")))
    depths = sorted({int(r["scramble_depth"]) for r in rows})
    solvers = list(COLORS)
    fig, axes = plt.subplots(1, 2, figsize=(17, 11))
    for ax, key, title in ((axes[0], "solved", "Cubes solved"),
                           (axes[1], "optimal", "Shortest answer found")):
        for s in solvers:
            vals = []
            for d in depths:
                g = [r for r in rows if r["solver"] == s and int(r["scramble_depth"]) == d]
                vals.append(100 * sum(r[key] == "True" for r in g) / len(g))
            ax.plot(depths, vals, marker="o", color=COLORS[s], label=s,
                    linestyle="--" if s == "IDA*" else "-", zorder=5 if s == "IDA*" else 2)
        ax.set_title(title)
        ax.set_ylabel("Percent of 5 scrambles")
        ax.set_xlabel("Scramble length (moves)")
        ax.set_xticks(depths)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(OUT / "mcts_heuristic_slide.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    mcts_slide()
    heuristic_slide()
    print("saved slide plots")


def pdb_histogram_slide():
    """One big histogram of the corner pattern database (slide 6)."""
    import contextlib, importlib.util, io, sys
    import numpy as np
    spec = importlib.util.spec_from_file_location("rubiks_cube_analysis", BASE / "rubiks_cube_analysis.py")
    rca = importlib.util.module_from_spec(spec)
    sys.modules["rubiks_cube_analysis"] = rca
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(rca)
    d = np.asarray(list(rca.CORNER_DISTANCE.values()), dtype=int)
    counts = np.bincount(d)
    fig, ax = plt.subplots(figsize=(14, 8.5))
    ax.bar(np.arange(len(counts)), counts, edgecolor="black", color=BLUE, width=0.75)
    for x, c in enumerate(counts):
        ax.text(x, c + 250, f"{c:,}", ha="center", va="bottom", fontsize=21)
    ax.set_xticks(np.arange(len(counts)))
    ax.set_xlabel("Distance h(s): moves to put the corners in place")
    ax.set_ylabel("Number of corner arrangements")
    ax.set_title("Histogram of all 40,320 corner arrangements")
    ax.set_ylim(0, counts.max() * 1.12)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "pdb_histogram_slide.png", dpi=150)
    plt.close(fig)


def ida_slide():
    """Two-panel IDA* benchmark figure, wide enough to fill slide 7."""
    import numpy as np
    rows = list(csv.DictReader(open(BASE / "ida_plots" / "ida_benchmark_results.csv", newline="")))
    rows = [r for r in rows if r["solved"] == "True"]
    lengths = sorted({int(r["scramble_length"]) for r in rows})
    nodes = [np.median([int(r["nodes"]) for r in rows if int(r["scramble_length"]) == n]) for n in lengths]
    secs = [np.mean([float(r["seconds"]) for r in rows if int(r["scramble_length"]) == n]) for n in lengths]
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    axes[0].plot(lengths, nodes, marker="o", color=BLUE)
    axes[0].set_yscale("log")
    axes[0].set_title("Nodes expanded")
    axes[0].set_ylabel("Nodes (log scale)")
    axes[1].plot(lengths, secs, marker="o", color="#f0a030")
    axes[1].set_title("Run time")
    axes[1].set_ylabel("Seconds")
    for ax in axes:
        ax.set_xlabel("Scramble length (moves)")
        ax.set_xticks(lengths)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "ida_benchmark_slide.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    pdb_histogram_slide()
    ida_slide()
    print("saved slide 6 and 7 plots")
