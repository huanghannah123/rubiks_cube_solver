# Solving the 2x2x2 Rubik's Cube with Search Algorithms

A school-year research project that solves the 2x2x2 Rubik's Cube (Pocket Cube) in Python with three search algorithms of increasing strength: **Breadth-First Search (BFS)**, **A\***, and **IDA\***. The goal is to find the shortest possible solution as efficiently as possible, and to measure how much a good heuristic speeds up the search.

## Files

| File | What it does |
|---|---|
| `cube2x2_bfs.py` | The `Cube` class (corner permutation + orientation), all 18 moves, and a BFS solver that always finds a shortest solution. Includes tests. |
| `cube_simulator.py` | A visual BFS simulator (tkinter). Scramble the cube, watch BFS search live, play back the solution, and map the entire state space of 3,674,160 positions. |
| `cube2x2_ida.py` | IDA* solver on a 24-sticker representation, with two pattern-database heuristics (corner permutation and corner orientation) combined with `max`. |
| `ida_simulator.py` | A visual IDA* simulator: shows f = g + h for every node, the rising bound, and the search tree as it grows. |
| `docs/research_sources.md` | Annotated bibliography of the papers and books behind the project. |

## Requirements

Python 3.8 or newer. No extra packages: everything uses the standard library. The simulators need `tkinter`, which comes with the normal python.org installer on Windows and macOS (on Linux: `sudo apt install python3-tk`).

## How to run

```bash
# BFS solver: runs the tests, then solves a sample scramble
python cube2x2_bfs.py

# IDA* solver: builds the heuristic tables, then solves a sample scramble
python cube2x2_ida.py

# Simulators (open a window)
python cube_simulator.py
python ida_simulator.py

# Simulator self-tests (no window)
python cube_simulator.py --test
python ida_simulator.py --test
```

## Results so far

- **State space:** BFS from the solved cube reaches all 3,674,160 positions. The number of positions at each distance matches the published table exactly, and the hardest positions need 11 moves ("God's number" for the 2x2x2 in the half-turn metric).
- **Optimized BFS:** storing each state as a single integer (0 to 3,674,159) with a bytearray as the visited list maps the whole state space in about 5 seconds, compared with roughly 1.5 GB of memory for the object-based version.
- **IDA\*:** with the combined pattern-database heuristic, IDA* finds optimal solutions while checking only a tiny fraction of the positions BFS would visit (for example, 70 nodes for the 7-move sample scramble in `cube2x2_ida.py`). The tests confirm its solutions are exactly as short as BFS's.

## Roadmap

- [x] BFS solver and state-space map
- [x] IDA* with pattern-database heuristics
- [ ] A* with a priority queue (`heapq`) and a node-count comparison against IDA*
- [ ] Benchmark table: nodes searched and run time for BFS vs A* vs IDA* on the same scrambles
- [ ] Learned heuristic (reinforcement learning), following DeepCubeA

## Key sources

See `docs/research_sources.md` for the full list. The core references are Hart, Nilsson & Raphael (1968) for A*, Korf (1985) for IDA*, Culberson & Schaeffer (1998) and Korf (1997) for pattern databases, and Agostinelli et al. (2019) for learned heuristics.
