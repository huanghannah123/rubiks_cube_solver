"""
IDA* vs three versions of MCTS on the SAME 2x2x2 scrambles.

    IDA*                 corner pattern database, always optimal
    MCTS (sticker)       your original MCTS heuristic (fraction of stickers solved)
    MCTS (PDB)           improved heuristic from mcts_improved.py
    MCTS (PDB + guided)  improved heuristic + rollouts guided by the PDB

Every solver gets the identical starting state for every case.
IDA* is optimal, so its solution length is the true shortest length, and
each MCTS answer is checked against it.

Run (takes about 10 minutes):
    python compare_mcts_heuristics.py

Output folder mcts_heuristic_results/:
    mcts_heuristic_comparison.csv   one row per (case, solver)
    mcts_heuristic_summary.csv      one row per solver
    mcts_heuristic_plots.png        4 panels, large labels for slides

To redraw the plot from the saved CSV without re-running:
    python compare_mcts_heuristics.py --plot-only

Note on "nodes": IDA* counts every state it expands. MCTS counts only tree
nodes; its random rollouts are extra work on top. For MCTS (PDB + guided),
each rollout step also looks up the table for every possible move. Compare
run time, not nodes, across the two families.
"""

import csv
import random
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mcts_improved import rca, make_solver

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "mcts_heuristic_results"

DEPTHS = range(1, 8)          # scramble lengths 1..7
PER_DEPTH = 5                 # 5 scrambles per length = 35 cases
SEED = 1234
MCTS_TIME_LIMIT = 5.0         # same limit as the MCTS benchmark in the slides
MCTS_ITERATIONS = 50_000
SOLVERS = ["IDA*", "MCTS (sticker)", "MCTS (PDB)", "MCTS (PDB + guided)"]
COLORS = {"IDA*": "#1f4e79", "MCTS (sticker)": "#c0504d",
          "MCTS (PDB)": "#f0a030", "MCTS (PDB + guided)": "#3a9d5d"}

# Big, readable text for slides (mentor feedback: make all labels bigger)
plt.rcParams.update({
    "font.size": 17, "axes.titlesize": 20, "axes.labelsize": 18,
    "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 15,
    "figure.titlesize": 24, "axes.titleweight": "bold", "lines.linewidth": 3,
    "lines.markersize": 10,
})


def make_cases():
    rng = random.Random(SEED)
    cases = []
    for depth in DEPTHS:
        for _ in range(PER_DEPTH):
            scramble = rca.make_random_scramble(depth, rng)
            cases.append({"case_id": len(cases) + 1, "scramble_depth": depth,
                          "scramble": scramble, "state": rca.scramble_cube(scramble)})
    return cases


def run_ida(state):
    solver = rca.IDAStarSolver()
    path = solver.solve(state, max_depth=11, verbose=False)
    ok = path is not None and rca.apply_solution(state, path) == rca.SOLVED
    return path if ok else None, solver.nodes, solver.elapsed_seconds


def run_mcts(kind, state, seed):
    solver = make_solver(kind, seed)
    path = solver.solve(state, iterations=MCTS_ITERATIONS, time_limit=MCTS_TIME_LIMIT)
    ok = path is not None and rca.apply_solution(state, path) == rca.SOLVED
    return path if ok else None, solver.stats["expanded_nodes"], solver.stats["elapsed_seconds"]


def summarize(rows):
    out = []
    for s in SOLVERS:
        sub = [r for r in rows if r["solver"] == s]
        solved = [r for r in sub if r["solved"]]
        out.append({
            "solver": s,
            "cases": len(sub),
            "solved": len(solved),
            "solved_pct": round(100 * len(solved) / len(sub), 1),
            "optimal": sum(r["optimal"] for r in sub),
            "mean_extra_moves_when_solved": round(statistics.mean(
                r["solution_length"] - r["optimal_length"] for r in solved), 2) if solved else "",
            "median_nodes": statistics.median(r["nodes"] for r in sub),
            "median_seconds": round(statistics.median(r["seconds"] for r in sub), 4),
            "max_seconds": round(max(r["seconds"] for r in sub), 3),
        })
    return out


def plot(rows, path):
    depths = list(DEPTHS)
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))

    def by_depth(solver, fn):
        vals = []
        for d in depths:
            g = [r for r in rows if r["solver"] == solver and r["scramble_depth"] == d]
            vals.append(fn(g))
        return vals

    panels = [
        ("Percent of cubes solved", "Solved (%)",
         lambda g: 100 * sum(r["solved"] for r in g) / len(g), False),
        ("Percent with the shortest solution", "Shortest (%)",
         lambda g: 100 * sum(r["optimal"] for r in g) / len(g), False),
        ("Median run time", "Seconds (log scale)",
         lambda g: statistics.median(r["seconds"] for r in g), True),
        ("Extra moves beyond the shortest (solved cubes)", "Mean extra moves",
         lambda g: statistics.mean(r["solution_length"] - r["optimal_length"]
                                   for r in g if r["solved"]) if any(r["solved"] for r in g)
         else float("nan"), None),
    ]
    for ax, (title, ylabel, fn, log) in zip(axes.flat, panels):
        for s in SOLVERS:
            ax.plot(depths, by_depth(s, fn), marker="o", color=COLORS[s], label=s)
        ax.set_title(title)
        ax.set_xlabel("Scramble length (moves)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(depths)
        ax.grid(True, alpha=0.3)
        if log:
            ax.set_yscale("log")
        elif log is False:
            ax.set_ylim(-5, 105)
        else:
            ax.set_ylim(bottom=-0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=18, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("2x2x2 cube: IDA* vs MCTS with three heuristics "
                 f"({PER_DEPTH} scrambles per length, same scrambles for every solver)")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def replot():
    """Redraw the plot from the saved CSV without re-running the solvers."""
    rows = []
    with open(OUT / "mcts_heuristic_comparison.csv", newline="") as f:
        for r in csv.DictReader(f):
            r["scramble_depth"] = int(r["scramble_depth"])
            r["solved"] = r["solved"] == "True"
            r["optimal"] = r["optimal"] == "True"
            r["optimal_length"] = int(r["optimal_length"])
            r["solution_length"] = int(r["solution_length"]) if r["solution_length"] else None
            r["nodes"] = int(r["nodes"])
            r["seconds"] = float(r["seconds"])
            rows.append(r)
    plot(rows, OUT / "mcts_heuristic_plots.png")


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    cases = make_cases()
    for case in cases:
        ida_path, ida_nodes, ida_sec = run_ida(case["state"])
        optimal_len = len(ida_path)
        print(f"Case {case['case_id']:2d}/{len(cases)} depth {case['scramble_depth']} "
              f"({case['scramble']}), shortest = {optimal_len}")
        results = {"IDA*": (ida_path, ida_nodes, ida_sec)}
        for kind in SOLVERS[1:]:
            results[kind] = run_mcts(kind, case["state"], SEED + case["case_id"])
        for s in SOLVERS:
            path, nodes, sec = results[s]
            rows.append({
                "case_id": case["case_id"], "scramble_depth": case["scramble_depth"],
                "scramble": case["scramble"], "solver": s,
                "solved": path is not None,
                "solution_length": len(path) if path is not None else "",
                "optimal_length": optimal_len,
                "optimal": path is not None and len(path) == optimal_len,
                "nodes": nodes, "seconds": sec,
                "solution": " ".join(path) if path else "",
            })
            print(f"    {s:20s} solved={path is not None!s:5s} "
                  f"len={len(path) if path else '-':>2} nodes={nodes:>6} time={sec:.3f}s")

    with open(OUT / "mcts_heuristic_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    with open(OUT / "mcts_heuristic_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    plot(rows, OUT / "mcts_heuristic_plots.png")

    print("\nSummary")
    for s in summary:
        print(s)


if __name__ == "__main__":
    import sys
    if "--plot-only" in sys.argv:
        replot()
    else:
        main()
