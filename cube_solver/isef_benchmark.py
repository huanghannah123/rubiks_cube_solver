"""
ISEF benchmark for the 2x2x2 Rubik's Cube: BFS vs IDA* vs MCTS.

Put this file in the same folder as:
    rubiks_cube_analysis.py                  (IDA* with corner-permutation PDB)
    rubiks_cube_mcts.py                      (MCTS, sticker heuristic)
    rubiks_cube_mcts_improved_heuristic.py   (MCTS, corner-aware heuristic)

Run from the VS Code terminal:
    python isef_benchmark.py              full run (about 45 minutes)
    python isef_benchmark.py --quick      small test run (about 3 minutes)
    python isef_benchmark.py --replot     redraw graphs from saved CSVs (seconds)

Everything is written to isef_results/:
    cases.csv          one row per (cube position, solver)
    memory.csv         peak memory measurements
    summary.csv        the numbers to quote, grouped by optimal distance
    summary.txt        the same numbers as readable text
    fig1_success_rate.png ... fig6_heuristic_accuracy.png, all_figures.png

How this differs from compare_idastar_mcts.py, and why:

1. Results are grouped by OPTIMAL distance (the true fewest moves, found by
   IDA*), not by scramble length. A 7-move scramble can be 4 moves from
   solved, so scramble length does not measure difficulty.
2. Each distance gets the same number of cube positions (--per-bin), so
   every point on a graph rests on the same sample size.
3. Success rates carry 95% Wilson confidence intervals.
4. Runtime is shown only for solved cases. A failed MCTS run always lasts
   exactly the time limit, so averaging it in would plot the limit.
5. Work is compared as "cube states generated": every state each solver
   produced, including MCTS rollout states, so the units match.
6. Peak memory is measured with tracemalloc in a separate, untimed rerun of
   the identical search (same seed, same iteration count), so measuring
   memory does not slow down the timed runs.
7. BFS (the baseline) runs for distances 1-5. Beyond that its visited set
   would need millions of states, which is the point it demonstrates.
   BFS also double-checks that every IDA* solution it covers is optimal.

Runtimes depend on the computer. Report the machine you ran it on.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import io
import math
import platform
import random
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

SOLVERS = ["BFS", "IDA*", "MCTS (sticker h)", "MCTS (corner h)"]
COLORS = {
    "IDA*": "#2a78d6",
    "MCTS (sticker h)": "#eb6834",
    "MCTS (corner h)": "#1baf7a",
    "BFS": "#52514e",
}
MARKERS = {"IDA*": "o", "MCTS (sticker h)": "s", "MCTS (corner h)": "^", "BFS": "D"}


# ---------------------------------------------------------------------------
# Loading the project modules
# ---------------------------------------------------------------------------
def load_modules():
    with contextlib.redirect_stdout(io.StringIO()):
        import rubiks_cube_analysis as ida
    import rubiks_cube_mcts as mcts_sticker
    import rubiks_cube_mcts_improved_heuristic as mcts_corner

    # The three files each build their own cube model. Make sure they agree.
    for mod in (mcts_sticker, mcts_corner):
        if mod.SOLVED != ida.SOLVED:
            raise RuntimeError(f"{mod.__name__} uses a different solved state")
        theirs = {a[0]: tuple(a[1]) for a in mod.ACTIONS}
        for name, perm, face in ida.ACTIONS:
            if theirs[name] != tuple(perm):
                raise RuntimeError(f"{mod.__name__} disagrees on move {name}")
    return ida, mcts_sticker, mcts_corner


# ---------------------------------------------------------------------------
# Building the test set: the same number of positions at each optimal distance
# ---------------------------------------------------------------------------
def build_cases(ida, per_bin, max_distance, seed):
    """
    Return cube positions grouped by optimal distance, with the IDA* result
    for each one (IDA* is how the optimal distance is known).

    For target distance d, scrambles of length d to d+2 are generated and
    solved with IDA* capped at depth d. A capped IDA* run behaves exactly
    like an uncapped run whenever a solution of length <= d exists, so its
    statistics are a valid IDA* measurement. Positions farther than d are
    skipped without a full search.
    """
    rng = random.Random(seed)
    bins = {d: [] for d in range(1, max_distance + 1)}
    seen = set()
    attempts = 0
    for target in range(1, max_distance + 1):
        # Only 18 positions are 1 move from solved (one per move), so small
        # distances may not have per_bin distinct positions. Stop a bin after
        # 3000 tries in a row that find nothing new.
        misses = 0
        while len(bins[target]) < per_bin and misses < 3000:
            attempts += 1
            misses += 1
            length = target + rng.choice((0, 1, 2))
            scramble = ida.make_random_scramble(length, rng)
            state = ida.scramble_cube(scramble)
            if state == ida.SOLVED or state in seen:
                continue
            solver = ida.IDAStarSolver()
            solution = solver.solve(state, max_depth=target, verbose=False)
            if solution is None:
                continue                      # farther than target: skip
            d = len(solution)
            if d == 0 or len(bins[d]) >= per_bin:
                continue
            if ida.apply_solution(state, solution) != ida.SOLVED:
                raise RuntimeError("IDA* returned a wrong solution")
            seen.add(state)
            if d == target:
                misses = 0
            bins[d].append({
                "scramble": scramble,
                "state": state,
                "optimal": d,
                "initial_h": ida.heuristic(state),
                "ida": {
                    "solved": True,
                    "solution": solution,
                    "seconds": solver.elapsed_seconds,
                    "states_generated": solver.generated,
                    "nodes_expanded": solver.nodes,
                    "iterations": len(solver.history),
                },
            })
            print(f"  distance {d}: {len(bins[d]):>2}/{per_bin}   "
                  f"IDA* {solver.elapsed_seconds:7.3f}s   {scramble}")
    for d, found in bins.items():
        if len(found) < per_bin:
            print(f"  note: only {len(found)} distinct positions found at distance {d}")
    cases = [c for d in sorted(bins) for c in bins[d]]
    for i, c in enumerate(cases, 1):
        c["case_id"] = i
    return cases, attempts


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------
def bfs_solve(ida, start):
    """Plain breadth-first search from the start state to SOLVED."""
    t0 = time.perf_counter()
    if start == ida.SOLVED:
        return [], 0, 1, time.perf_counter() - t0
    parent = {start: None}
    queue = collections.deque([start])
    generated = 0
    while queue:
        state = queue.popleft()
        for name, perm, face in ida.ACTIONS:
            nxt = ida.apply_move(state, perm)
            generated += 1
            if nxt in parent:
                continue
            parent[nxt] = (state, name)
            if nxt == ida.SOLVED:
                path = []
                node = nxt
                while parent[node] is not None:
                    node, move = parent[node]
                    path.append(move)
                return path[::-1], generated, len(parent), time.perf_counter() - t0
            queue.append(nxt)
    return None, generated, len(parent), time.perf_counter() - t0


def make_mcts(mod, rollout_depth, seed):
    return mod.MCTSCubeSolver(
        solved=mod.SOLVED, actions=mod.ACTIONS, apply_move=mod.apply_move,
        exploration=math.sqrt(2), rollout_depth=rollout_depth, seed=seed,
        history_interval=0,
    )


def run_mcts(mod, state, time_limit, max_iterations, rollout_depth, seed):
    solver = make_mcts(mod, rollout_depth, seed)
    solution = solver.solve(state, iterations=max_iterations, time_limit=time_limit)
    ok = solution is not None and solver.verify_solution(state, solution)
    s = solver.stats
    return {
        "solved": ok,
        "solution": solution if ok else None,
        "seconds": s["elapsed_seconds"],
        # tree nodes plus every state visited during random rollouts
        "states_generated": s["expanded_nodes"] + s["rollout_steps"],
        "nodes_expanded": s["expanded_nodes"],
        "iterations": s["iterations"],
    }


def peak_memory(fn):
    """Peak Python memory (bytes) allocated while fn() runs."""
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def wilson(successes, n, z=1.96):
    """95% Wilson score interval for a success rate."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def quartiles(values):
    values = sorted(values)
    if not values:
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0], values[0]
    q = statistics.quantiles(values, n=4, method="inclusive")
    return q[0], statistics.median(values), q[2]


def summarize(rows, mem_rows, distances):
    out = []
    for d in distances:
        for solver in SOLVERS:
            group = [r for r in rows if r["optimal"] == d and r["solver"] == solver]
            if not group:
                continue
            solved = [r for r in group if r["solved"]]
            rate, lo, hi = wilson(len(solved), len(group))
            t25, tmed, t75 = quartiles([r["seconds"] for r in solved])
            g25, gmed, g75 = quartiles([r["states_generated"] for r in group])
            mems = [m["peak_bytes"] for m in mem_rows
                    if m["optimal"] == d and m["solver"] == solver]
            out.append({
                "optimal_distance": d,
                "solver": solver,
                "cases": len(group),
                "solved": len(solved),
                "success_rate": round(rate, 4),
                "success_ci95_low": round(lo, 4),
                "success_ci95_high": round(hi, 4),
                "median_seconds_solved": tmed,
                "q1_seconds_solved": t25,
                "q3_seconds_solved": t75,
                "mean_extra_moves_solved": (
                    statistics.mean(r["extra_moves"] for r in solved)
                    if solved else float("nan")),
                "max_extra_moves_solved": (
                    max(r["extra_moves"] for r in solved)
                    if solved else float("nan")),
                "median_states_generated": gmed,
                "q1_states_generated": g25,
                "q3_states_generated": g75,
                "memory_cases": len(mems),
                "median_peak_memory_kb": (
                    statistics.median(mems) / 1024 if mems else float("nan")),
            })
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def style_axis(ax, xlabel, ylabel, distances):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(distances)
    ax.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#8a8984")
    ax.tick_params(colors="#52514e")


def series(summary, solver, key, distances):
    lookup = {(s["optimal_distance"], s["solver"]): s for s in summary}
    xs, ys = [], []
    for d in distances:
        s = lookup.get((d, solver))
        if s is not None and s[key] == s[key]:          # skip NaN
            xs.append(d)
            ys.append(s[key])
    return xs, ys


def _percent():
    from matplotlib.ticker import FuncFormatter
    return FuncFormatter(lambda v, _: f"{v:.0%}")


def draw_success(ax, summary, distances):
    lookup = {(s["optimal_distance"], s["solver"]): s for s in summary}
    offsets = {"BFS": -0.15, "IDA*": -0.05, "MCTS (sticker h)": 0.05, "MCTS (corner h)": 0.15}
    for solver in SOLVERS:
        pts = [(d, lookup[(d, solver)]) for d in distances if (d, solver) in lookup]
        if not pts:
            continue
        xs = [d + offsets[solver] for d, _ in pts]
        ys = [s["success_rate"] for _, s in pts]
        err = [[s["success_rate"] - s["success_ci95_low"] for _, s in pts],
               [s["success_ci95_high"] - s["success_rate"] for _, s in pts]]
        ax.errorbar(xs, ys, yerr=err, color=COLORS[solver], marker=MARKERS[solver],
                    markersize=7, linewidth=2, capsize=3, label=solver)
    ax.set_ylim(-0.03, 1.05)
    ax.yaxis.set_major_formatter(_percent())
    ax.set_title("Success rate (bars = 95% confidence interval)")
    style_axis(ax, "Optimal solution length (moves)", "Positions solved", distances)


def draw_runtime(ax, summary, distances):
    for solver in SOLVERS:
        xs, med = series(summary, solver, "median_seconds_solved", distances)
        _, q1 = series(summary, solver, "q1_seconds_solved", distances)
        _, q3 = series(summary, solver, "q3_seconds_solved", distances)
        if not xs:
            continue
        ax.plot(xs, med, color=COLORS[solver], marker=MARKERS[solver],
                markersize=7, linewidth=2, label=solver)
        ax.fill_between(xs, q1, q3, color=COLORS[solver], alpha=0.15, linewidth=0)
    ax.set_yscale("log")
    ax.set_title("Runtime on solved positions (median, band = middle 50%)")
    style_axis(ax, "Optimal solution length (moves)", "Seconds (log scale)", distances)


def draw_extra_moves(ax, summary, distances):
    for solver in SOLVERS:
        xs, ys = series(summary, solver, "mean_extra_moves_solved", distances)
        if xs:
            ax.plot(xs, ys, color=COLORS[solver], marker=MARKERS[solver],
                    markersize=7, linewidth=2, label=solver)
    ax.set_title("Moves above optimal on solved positions (0 = optimal)")
    style_axis(ax, "Optimal solution length (moves)", "Mean extra moves", distances)


def draw_states(ax, summary, distances):
    for solver in SOLVERS:
        xs, med = series(summary, solver, "median_states_generated", distances)
        _, q1 = series(summary, solver, "q1_states_generated", distances)
        _, q3 = series(summary, solver, "q3_states_generated", distances)
        if not xs:
            continue
        ax.plot(xs, med, color=COLORS[solver], marker=MARKERS[solver],
                markersize=7, linewidth=2, label=solver)
        ax.fill_between(xs, q1, q3, color=COLORS[solver], alpha=0.15, linewidth=0)
    ax.set_yscale("log")
    ax.set_title("Search work: cube states generated (median, all runs)")
    style_axis(ax, "Optimal solution length (moves)", "States (log scale)", distances)


def draw_memory(ax, summary, distances, pdb_kb):
    for solver in SOLVERS:
        xs, ys = series(summary, solver, "median_peak_memory_kb", distances)
        if xs:
            ax.plot(xs, ys, color=COLORS[solver], marker=MARKERS[solver],
                    markersize=7, linewidth=2, label=solver)
    ax.axhline(pdb_kb, color=COLORS["IDA*"], linestyle="--", linewidth=1.5)
    ax.annotate(f"IDA* pattern database, built once: {pdb_kb / 1024:.1f} MB",
                xy=(distances[-1], pdb_kb), xytext=(0, -13), textcoords="offset points",
                ha="right", color="#52514e", fontsize=9)
    ax.set_yscale("log")
    ax.set_title("Peak memory during the search (median)")
    style_axis(ax, "Optimal solution length (moves)", "Kilobytes (log scale)", distances)


def draw_heuristic(ax, cases, distances):
    counts = collections.Counter((c["optimal"], c["initial_h"]) for c in cases)
    biggest = max(counts.values())
    for (d, h), n in counts.items():
        ax.scatter(d, h, s=40 + 300 * n / biggest, color=COLORS["IDA*"],
                   alpha=0.8, edgecolors="white", linewidths=1.5, zorder=3)
    lim = max(distances) + 0.5
    ax.plot([0.5, lim], [0.5, lim], color="#8a8984", linestyle="--", linewidth=1.2)
    ax.annotate("perfect estimate (h = true distance)", xy=(lim - 0.2, lim - 0.2),
                xytext=(-5, -14), textcoords="offset points", ha="right",
                color="#52514e", fontsize=9)
    ax.set_xlim(0.5, lim)
    ax.set_ylim(-0.3, lim)
    ax.set_title("Pattern-database heuristic vs true distance (dot size = count)")
    style_axis(ax, "Optimal solution length (moves)",
               "Heuristic h(start) (moves)", distances)
    ax.set_yticks(range(0, max(distances) + 1))


def make_figures(summary, cases, distances, pdb_kb, out_dir, subtitle):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                         "axes.titleweight": "bold"})
    panels = [
        ("fig1_success_rate.png", lambda ax: draw_success(ax, summary, distances), True),
        ("fig2_runtime.png", lambda ax: draw_runtime(ax, summary, distances), True),
        ("fig3_extra_moves.png", lambda ax: draw_extra_moves(ax, summary, distances), True),
        ("fig4_states_generated.png", lambda ax: draw_states(ax, summary, distances), True),
        ("fig5_peak_memory.png", lambda ax: draw_memory(ax, summary, distances, pdb_kb), True),
        ("fig6_heuristic_accuracy.png", lambda ax: draw_heuristic(ax, cases, distances), False),
    ]
    for filename, draw, legend in panels:
        fig, ax = plt.subplots(figsize=(8, 5.2))
        draw(ax)
        if legend:
            ax.legend(frameon=False, fontsize=10)
        fig.text(0.01, 0.005, subtitle, fontsize=8, color="#8a8984")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(out_dir / filename, dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    for ax, (_, draw, legend) in zip(axes.flat, panels):
        draw(ax)
        if legend:
            ax.legend(frameon=False, fontsize=9)
    fig.suptitle("2x2x2 Rubik's Cube: BFS vs IDA* vs MCTS, grouped by optimal distance",
                 fontsize=15, fontweight="bold")
    fig.text(0.01, 0.005, subtitle, fontsize=9, color="#8a8984")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out_dir / "all_figures.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------
def write_csv(rows, path):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(x, digits=3):
    if isinstance(x, float) and x != x:
        return "-"
    if isinstance(x, float):
        return f"{x:.{digits}g}"
    return str(x)


def write_text_summary(summary, rows, args, pdb_kb, machine, attempts, path):
    lines = [
        "2x2x2 Rubik's Cube search benchmark",
        f"Machine: {machine}",
        f"Positions per optimal distance: {args.per_bin} (distance 1 has only 18)   distances: 1-{args.max_distance}",
        f"MCTS budget: {args.mcts_time_limit:g} s or {args.mcts_iterations} iterations, "
        f"rollout depth {args.rollout_depth}",
        f"BFS run for distances 1-{args.bfs_max_distance}",
        f"Seed: {args.seed}   scrambles tried to fill the bins: {attempts}",
        f"IDA* pattern database (40,320 corner permutations): {pdb_kb / 1024:.1f} MB, built once",
        "",
    ]
    header = (f"{'dist':>4}  {'solver':<17}{'solved':>9}  {'95% CI':>10}  "
              f"{'median s':>9}  {'extra mv':>8}  {'states':>10}  {'mem KB':>9}")
    lines.append(header)
    lines.append("-" * len(header))
    for s in summary:
        ci = f"{s['success_ci95_low']:.0%}-{s['success_ci95_high']:.0%}"
        lines.append(
            f"{s['optimal_distance']:>4}  {s['solver']:<17}"
            f"{s['solved']:>4}/{s['cases']:<4}  {ci:>10}  "
            f"{fmt(s['median_seconds_solved']):>9}  "
            f"{fmt(s['mean_extra_moves_solved'], 2):>8}  "
            f"{fmt(s['median_states_generated']):>10}  "
            f"{fmt(s['median_peak_memory_kb']):>9}")
    ida_rows = [r for r in rows if r["solver"] == "IDA*"]
    under = sum(r["seconds"] <= args.mcts_time_limit for r in ida_rows)
    lines += [
        "",
        f"IDA* solved {len(ida_rows)}/{len(ida_rows)} positions, all optimal. "
        f"{under}/{len(ida_rows)} finished within the MCTS time budget "
        f"of {args.mcts_time_limit:g} s.",
        f"Slowest IDA* solve: {max(r['seconds'] for r in ida_rows):.2f} s.",
        f"BFS agreed with IDA* on the optimal length of every position up to "
        f"distance {args.bfs_max_distance}.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def replot(args):
    """Redraw every figure from saved CSVs, without rerunning any search."""
    import matplotlib
    matplotlib.use("Agg")
    out_dir = BASE_DIR / args.output_dir
    with (out_dir / "summary.csv").open(encoding="utf-8") as f:
        summary = []
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                if k == "solver":
                    row[k] = v
                elif k in ("optimal_distance", "cases", "solved", "memory_cases"):
                    row[k] = int(v)
                else:
                    row[k] = float(v) if v not in ("", "nan") else float("nan")
            summary.append(row)
    with (out_dir / "cases.csv").open(encoding="utf-8") as f:
        cases = [{"optimal": int(r["optimal"]), "initial_h": int(r["initial_h"])}
                 for r in csv.DictReader(f) if r["solver"] == "IDA*"]
    info = (out_dir / "summary.txt").read_text(encoding="utf-8").splitlines()
    pdb_mb = float(next(l for l in info if "pattern database" in l)
                   .split(":")[1].split("MB")[0])
    per_bin = next(l for l in info if l.startswith("Positions")).split(":")[1].split()[0]
    budget = next(l for l in info if l.startswith("MCTS budget")).split(":")[1].split()[0]
    seed = next(l for l in info if l.startswith("Seed")).split()[1]
    machine = next(l for l in info if l.startswith("Machine")).split(": ", 1)[1]
    distances = sorted({s["optimal_distance"] for s in summary})
    subtitle = (f"{per_bin} positions per distance, seed {seed}, "
                f"MCTS budget {budget} s. {machine}")
    make_figures(summary, cases, distances, pdb_mb * 1024, out_dir, subtitle)
    print(f"Figures redrawn in {out_dir}")


def benchmark(args):
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")

    t_start = time.perf_counter()
    print("Loading solvers and building the IDA* pattern database...")
    ida, mcts_sticker, mcts_corner = load_modules()
    out_dir = BASE_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Memory cost of the pattern database, measured by rebuilding it once.
    pdb_kb = peak_memory(ida.build_corner_pattern_database) / 1024

    machine = (f"{platform.processor() or platform.machine()}, {platform.system()}, "
               f"Python {platform.python_version()}")
    distances = list(range(1, args.max_distance + 1))

    print(f"\nStep 1/3: finding {args.per_bin} positions at each optimal distance "
          f"1-{args.max_distance} (this is where IDA* runs)")
    cases, attempts = build_cases(ida, args.per_bin, args.max_distance, args.seed)

    print("\nStep 2/3: running BFS and both MCTS versions on every position")
    rows = []
    mcts_versions = [("MCTS (sticker h)", mcts_sticker), ("MCTS (corner h)", mcts_corner)]
    for c in cases:
        state, d = c["state"], c["optimal"]
        results = {"IDA*": c["ida"]}
        if d <= args.bfs_max_distance:
            path, generated, stored, secs = bfs_solve(ida, state)
            ok = path is not None and ida.apply_solution(state, path) == ida.SOLVED
            if ok and len(path) != d:
                raise RuntimeError(f"BFS found {len(path)} moves but IDA* found {d}")
            results["BFS"] = {
                "solved": ok, "solution": path, "seconds": secs,
                "states_generated": generated, "nodes_expanded": stored,
                "iterations": None,
            }
        for k, (name, mod) in enumerate(mcts_versions):
            seed = args.seed * 1000 + c["case_id"] * 10 + k
            r = run_mcts(mod, state, args.mcts_time_limit, args.mcts_iterations,
                         args.rollout_depth, seed)
            r["seed"] = seed
            results[name] = r
        c["results"] = results
        for solver in SOLVERS:
            if solver not in results:
                continue
            r = results[solver]
            rows.append({
                "case_id": c["case_id"],
                "optimal": d,
                "initial_h": c["initial_h"],
                "scramble": c["scramble"],
                "solver": solver,
                "solved": r["solved"],
                "solution_length": len(r["solution"]) if r["solved"] else None,
                "extra_moves": len(r["solution"]) - d if r["solved"] else None,
                "seconds": r["seconds"],
                "states_generated": r["states_generated"],
                "nodes_expanded": r["nodes_expanded"],
                "iterations": r["iterations"],
                "solution": " ".join(r["solution"]) if r["solved"] else None,
            })
        line = "  ".join(
            f"{s}={'ok' if results[s]['solved'] else 'FAIL'} {results[s]['seconds']:.2f}s"
            for s in SOLVERS if s in results)
        print(f"  case {c['case_id']:>3}/{len(cases)}  d={d}  {line}")

    print(f"\nStep 3/3: measuring peak memory on {args.memory_cases} positions "
          "per distance (untimed reruns)")
    mem_rows = []
    for d in distances:
        for c in [c for c in cases if c["optimal"] == d][: args.memory_cases]:
            state = c["state"]
            jobs = [("IDA*", lambda: ida.IDAStarSolver().solve(
                state, max_depth=d, verbose=False))]
            if d <= args.bfs_max_distance:
                jobs.append(("BFS", lambda: bfs_solve(ida, state)))
            for name, mod in mcts_versions:
                r = c["results"][name]
                jobs.append((name, lambda mod=mod, r=r: make_mcts(
                    mod, args.rollout_depth, r["seed"]).solve(
                        state, iterations=r["iterations"], time_limit=float("inf"))))
            for solver, job in jobs:
                mem_rows.append({"case_id": c["case_id"], "optimal": d,
                                 "solver": solver, "peak_bytes": peak_memory(job)})
            print(f"  memory measured for case {c['case_id']} (d={d})")

    summary = summarize(rows, mem_rows, distances)
    write_csv(rows, out_dir / "cases.csv")
    write_csv(mem_rows, out_dir / "memory.csv")
    write_csv(summary, out_dir / "summary.csv")
    text = write_text_summary(summary, rows, args, pdb_kb, machine, attempts,
                              out_dir / "summary.txt")
    subtitle = (f"{args.per_bin} positions per distance, seed {args.seed}, "
                f"MCTS budget {args.mcts_time_limit:g} s. {machine}")
    make_figures(summary, cases, distances, pdb_kb, out_dir, subtitle)
    print("\n" + text)
    print(f"\nTotal run time: {(time.perf_counter() - t_start) / 60:.1f} minutes")
    print(f"All results saved in {out_dir}")
    if args.show:
        import matplotlib.pyplot as plt
        img = plt.imread(out_dir / "all_figures.png")
        plt.figure(figsize=(16, 9))
        plt.imshow(img)
        plt.axis("off")
        plt.show()


def parse_args():
    p = argparse.ArgumentParser(
        description="2x2x2 Rubik's Cube benchmark: BFS vs IDA* vs MCTS.")
    p.add_argument("--per-bin", type=int, default=20,
                   help="positions per optimal distance (default 20)")
    p.add_argument("--max-distance", type=int, default=9,
                   help="largest optimal distance to test (default 9)")
    p.add_argument("--bfs-max-distance", type=int, default=5,
                   help="run BFS up to this distance (default 5)")
    p.add_argument("--mcts-time-limit", type=float, default=5.0,
                   help="seconds MCTS gets per position (default 5)")
    p.add_argument("--mcts-iterations", type=int, default=50_000)
    p.add_argument("--rollout-depth", type=int, default=20)
    p.add_argument("--memory-cases", type=int, default=3,
                   help="positions per distance used for memory (default 3)")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output-dir", default="isef_results")
    p.add_argument("--quick", action="store_true",
                   help="small test run: 4 per distance, distances 1-7, 2 s MCTS")
    p.add_argument("--replot", action="store_true",
                   help="only redraw the figures from an earlier run's CSV files")
    p.add_argument("--show", action="store_true",
                   help="open the combined figure in a window at the end")
    args = p.parse_args()
    if args.quick:
        args.per_bin, args.max_distance = 4, 7
        args.mcts_time_limit, args.memory_cases = 2.0, 1
        args.output_dir += "_quick"
    if args.per_bin < 1 or args.max_distance < 1:
        p.error("--per-bin and --max-distance must be at least 1")
    if args.bfs_max_distance > 6:
        print("Warning: BFS above distance 6 needs several GB of memory.")
    if args.max_distance > 10:
        print("Warning: distance 11 positions can take 20+ minutes each for IDA*.")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.replot:
        replot(arguments)
    else:
        benchmark(arguments)
