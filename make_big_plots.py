"""
Redraw every project plot with large, slide-readable labels.
(Mentor feedback: make all plot labels bigger.)

Uses your saved results, so nothing is re-benchmarked:
    mcts_results.csv                       -> mcts_plots.png
    ida_plots/ida_benchmark_results.csv    -> ida_plots/benchmark_performance.png
                                              ida_plots/benchmark_distributions.png
    corner pattern database                -> ida_plots/pdb_distance_distribution.png
    example IDA* solve                     -> ida_plots/example_threshold_profile.png
    comparison_results/comparison.csv      -> comparison_results/comparison_plots.png

New files go into big_plots/ so your originals stay untouched.

Run:
    python make_big_plots.py
"""

import contextlib
import csv
import importlib.util
import io
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "big_plots"
NAVY, BLUE, ORANGE, GREEN = "#1f4e79", "#2e75b6", "#f0a030", "#3a9d5d"

BIG = {
    "font.size": 17, "axes.titlesize": 20, "axes.titleweight": "bold",
    "axes.labelsize": 18, "xtick.labelsize": 16, "ytick.labelsize": 16,
    "legend.fontsize": 16, "figure.titlesize": 24, "lines.linewidth": 3,
    "lines.markersize": 10, "axes.prop_cycle": matplotlib.cycler(color=[BLUE, ORANGE, GREEN, NAVY]),
}
plt.rcParams.update(BIG)


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, BASE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(x):
    if x in ("", "None", None):
        return None
    if x in ("True", "False"):
        return x == "True"
    v = float(x)
    return int(v) if v.is_integer() else v


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


# ---------------------------------------------------------------- MCTS
def mcts_plots():
    rows = [{k: num(v) if k != "scramble" else v for k, v in r.items()}
            for r in read_csv(BASE / "mcts_results.csv")]
    depths = sorted({r["scramble_depth"] for r in rows})

    def mean_by(metric, only_solved=False):
        out = []
        for d in depths:
            g = [r[metric] for r in rows if r["scramble_depth"] == d
                 and (r["solved"] or not only_solved) and r[metric] is not None]
            out.append(statistics.mean(g) if g else float("nan"))
        return out

    success = [100 * statistics.mean(bool(r["solved"]) for r in rows if r["scramble_depth"] == d)
               for d in depths]
    panels = [
        (success, "Success rate", "Solved (%)"),
        (mean_by("elapsed_seconds"), "Run time", "Seconds"),
        (mean_by("solution_length", True), "Solution length (solved)", "Moves"),
        (mean_by("expanded_nodes"), "Tree nodes expanded", "Nodes"),
        (mean_by("rollout_steps"), "Rollout steps", "Steps"),
        (mean_by("iterations_to_solution", True), "Iterations to solution", "Iterations"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    for ax, (vals, title, ylabel) in zip(axes.flat, panels):
        ax.plot(depths, vals, marker="o", color=BLUE)
        ax.set_title(title)
        ax.set_xlabel("Scramble length (moves)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(depths)
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 105)
    fig.suptitle("2x2x2 Rubik's Cube: MCTS performance (5 scrambles per length)")
    save(fig, "mcts_plots.png")


# ---------------------------------------------------------------- IDA*
def ida_plots(rca):
    rows = [{k: num(v) if k != "scramble" else v for k, v in r.items()}
            for r in read_csv(BASE / "ida_plots" / "ida_benchmark_results.csv")]
    solved = [r for r in rows if r["solved"]]
    lengths = sorted({r["scramble_length"] for r in solved})
    mean_nodes = [np.mean([r["nodes"] for r in solved if r["scramble_length"] == n]) for n in lengths]
    med_nodes = [np.median([r["nodes"] for r in solved if r["scramble_length"] == n]) for n in lengths]
    mean_time = [np.mean([r["seconds"] for r in solved if r["scramble_length"] == n]) for n in lengths]

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    axes[0, 0].scatter([r["solution_length"] for r in solved], [r["nodes"] for r in solved],
                       s=140, alpha=0.8, color=BLUE)
    axes[0, 0].set(title="Search effort vs. solution length", xlabel="Solution length (moves)",
                   ylabel="Nodes expanded (log scale)", yscale="log")
    axes[0, 1].plot(lengths, mean_nodes, "o-", label="Mean nodes")
    axes[0, 1].plot(lengths, med_nodes, "s--", label="Median nodes")
    axes[0, 1].set(title="Search effort vs. scramble length", xlabel="Scramble length (moves)",
                   ylabel="Nodes expanded (log scale)", yscale="log")
    axes[0, 1].legend()
    axes[1, 0].plot(lengths, mean_time, "o-")
    axes[1, 0].set(title="Mean run time vs. scramble length", xlabel="Scramble length (moves)",
                   ylabel="Seconds")
    axes[1, 1].scatter([r["initial_heuristic"] for r in solved], [r["nodes"] for r in solved],
                       s=140, alpha=0.8, color=ORANGE)
    axes[1, 1].set(title="Search effort vs. starting heuristic", xlabel="Starting heuristic h(s)",
                   ylabel="Nodes expanded (log scale)", yscale="log")
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    fig.suptitle("2x2x2 Rubik's Cube: IDA* benchmark")
    save(fig, "benchmark_performance.png")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    nodes = [r["nodes"] for r in solved]
    bins = np.logspace(np.log10(max(1, min(nodes))), np.log10(max(nodes)), 12)
    axes[0].hist(nodes, bins=bins, edgecolor="black", color=BLUE)
    axes[0].set(title="Histogram of nodes expanded", xlabel="Nodes expanded (log scale)",
                ylabel="Number of scrambles", xscale="log")
    axes[1].hist([r["seconds"] for r in solved], bins=12, edgecolor="black", color=ORANGE)
    axes[1].set(title="Histogram of solve times", xlabel="Seconds", ylabel="Number of scrambles")
    save(fig, "benchmark_distributions.png")

    d = np.asarray(list(rca.CORNER_DISTANCE.values()), dtype=int)
    counts = np.bincount(d)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    axes[0].bar(np.arange(len(counts)), counts, edgecolor="black", color=BLUE)
    axes[0].set(title="Pattern database: histogram of distances", xlabel="Distance h(s) in moves",
                ylabel="Number of corner arrangements")
    axes[1].bar(np.arange(len(counts)), 100 * counts / len(d), edgecolor="black", color=ORANGE)
    axes[1].set(title="Share of arrangements by distance", xlabel="Distance h(s) in moves",
                ylabel="Percent of 40,320 arrangements")
    for ax in axes:
        ax.set_xticks(np.arange(len(counts)))
    save(fig, "pdb_distance_distribution.png")

    solver = rca.IDAStarSolver()
    with contextlib.redirect_stdout(io.StringIO()):
        solver.solve(rca.scramble_cube("R U R2 F D F2 U"), max_depth=11, verbose=False)
    h = solver.history
    th = [x["threshold"] for x in h]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    axes[0].bar(th, [x["nodes"] for x in h], edgecolor="black", color=BLUE)
    axes[0].set(title="IDA* nodes expanded per threshold", xlabel="f-cost threshold",
                ylabel="Nodes expanded")
    axes[1].plot(th, [x["seconds"] for x in h], marker="o", color=ORANGE)
    axes[1].set(title="Run time per threshold", xlabel="f-cost threshold", ylabel="Seconds")
    for ax in axes:
        ax.set_xticks(th)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Example solve: R U R2 F D F2 U")
    save(fig, "example_threshold_profile.png")


# ---------------------------------------------------------------- IDA* vs MCTS (Nov 11 run)
def comparison_plots():
    rows = [{k: num(v) if k not in ("scramble", "solver", "solution") else v
             for k, v in r.items()}
            for r in read_csv(BASE / "comparison_results" / "comparison.csv")]
    depths = sorted({r["scramble_depth"] for r in rows})
    panels = [
        ("success", "Success rate", "Solved (%)"),
        ("elapsed_seconds", "Mean run time", "Seconds (log scale)"),
        ("solution_length", "Mean solution length (solved)", "Moves"),
        ("nodes", "Mean nodes expanded", "Nodes (log scale)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    for ax, (metric, title, ylabel) in zip(axes.flat, panels):
        for solver, color in (("IDA*", NAVY), ("MCTS", ORANGE)):
            vals = []
            for d in depths:
                g = [r for r in rows if r["solver"] == solver and r["scramble_depth"] == d]
                if metric == "success":
                    vals.append(100 * sum(bool(r["solved"]) for r in g) / len(g))
                else:
                    e = [r[metric] for r in g if r[metric] is not None
                         and (metric != "solution_length" or r["solved"])]
                    vals.append(statistics.mean(e) if e else float("nan"))
            ax.plot(depths, vals, marker="o", color=color, label=solver)
        ax.set_title(title)
        ax.set_xlabel("Scramble length (moves)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(depths)
        ax.grid(True, alpha=0.3)
        ax.legend()
        if metric in ("elapsed_seconds", "nodes"):
            ax.set_yscale("log")
    axes[0, 0].set_ylim(0, 105)
    fig.suptitle("2x2x2 Rubik's Cube: IDA* vs MCTS (same scrambles)")
    save(fig, "comparison_plots.png")


# ---------------------------------------------------------------- BFS histogram
# Positions at each distance from solved (half-turn metric), as found by the
# full BFS in cube2x2_bfs.py / cube_simulator.py. These match Jaap
# Scherphuis's published 2x2x2 table at every distance.
BFS_COUNTS = [1, 9, 54, 321, 1847, 9992, 50136, 227536, 870072, 1887748, 623800, 2644]


def bfs_histogram():
    assert sum(BFS_COUNTS) == 3_674_160
    d = np.arange(len(BFS_COUNTS))
    fig, ax = plt.subplots(figsize=(18, 7.5))
    bars = ax.bar(d, BFS_COUNTS, edgecolor="black", color=BLUE)
    for x, c in zip(d, BFS_COUNTS):
        ax.text(x, c * 1.3, f"{c:,}", ha="center", va="bottom", fontsize=15)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 1e7)
    ax.set_xticks(d)
    ax.set_xlabel("Distance from solved (moves)")
    ax.set_ylabel("Number of positions (log scale)")
    ax.set_title("Histogram of all 3,674,160 positions by distance from solved (BFS)")
    ax.grid(True, alpha=0.3, axis="y")
    save(fig, "bfs_histogram.png")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    bfs_histogram()
    rca = load("rubiks_cube_analysis", "rubiks_cube_analysis.py")
    mcts_plots()
    ida_plots(rca)
    comparison_plots()
