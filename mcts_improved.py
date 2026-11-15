"""
Improved heuristic for the MCTS 2x2x2 solver.

Your original MCTS heuristic counts the fraction of stickers already in place.
That number barely changes when a move helps or hurts, so the search gets
almost no signal. This file keeps your MCTSCubeSolver exactly as it is and
adds two upgraded versions that reuse the corner pattern database (PDB)
from rubiks_cube_analysis.py, the same table IDA* uses.

    MCTS (sticker)      your original solver, unchanged
    MCTS (PDB)          heuristic = 1 - PDB distance / 8
                        rollout reward = closest state seen during the rollout
    MCTS (PDB + guided) same heuristic, and each rollout move picks the move
                        that lowers the PDB distance most (with some randomness),
                        then wasted moves are cut out of the answer it finds

Place this file next to rubiks_cube_analysis.py and rubiks_cube_mcts.py.
"""

import contextlib
import importlib.util
import io
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _load(name, filename):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, BASE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    with contextlib.redirect_stdout(io.StringIO()):     # hide startup prints
        spec.loader.exec_module(module)
    return module


rca = _load("rubiks_cube_analysis", "rubiks_cube_analysis.py")
rcm = _load("rubiks_cube_mcts", "rubiks_cube_mcts.py")

# Largest value in the corner PDB is 7, so 8 keeps the score above 0.
PDB_SCALE = max(rca.CORNER_DISTANCE.values()) + 1


class PDBMCTSSolver(rcm.MCTSCubeSolver):
    """MCTS with the corner pattern database as its heuristic."""

    def __init__(self, *args, guided_rollouts=False, greedy_prob=0.7, **kwargs):
        super().__init__(*args, **kwargs)
        self.guided_rollouts = guided_rollouts
        self.greedy_prob = greedy_prob

    def heuristic(self, state):
        # 1.0 at the solved cube, lower the farther the corners are from solved.
        return 1.0 - rca.heuristic(state) / PDB_SCALE

    def _pick_rollout_action(self, state, actions):
        if self.guided_rollouts and self.rng.random() < self.greedy_prob:
            best, best_h = [], None
            for action in actions:
                h = rca.heuristic(self.apply_move(state, action[1]))
                if best_h is None or h < best_h:
                    best, best_h = [action], h
                elif h == best_h:
                    best.append(action)
            return self.rng.choice(best)
        return self.rng.choice(actions)

    def rollout(self, state, last_face=None):
        current = state
        previous_face = last_face
        moves = []
        best_h = self.heuristic(current)

        for _ in range(self.rollout_depth):
            if self.is_solved(current):
                return 1.0, moves, True, best_h
            actions = self.legal_actions(previous_face)
            name, permutation, face = self._pick_rollout_action(current, actions)
            current = self.apply_move(current, permutation)
            moves.append(name)
            previous_face = face
            self.stats["rollout_steps"] += 1
            best_h = max(best_h, self.heuristic(current))

        if self.is_solved(current):
            return 1.0, moves, True, best_h
        # Reward the closest point the rollout reached, not where it ended up.
        # A random walk usually wanders away again, so the end state hides
        # whether this branch was promising.
        return 0.9 * best_h, moves, False, best_h

    def shorten(self, start_state, moves):
        """Cut wasted moves out of a found solution.

        Walk the states the solution passes through. If a later state can be
        reached from an earlier one in a single move (or is the same state),
        replace everything in between with that one move. Repeat until
        nothing changes. The result still solves the cube and is never longer.
        """
        by_name = {a[0]: a for a in self.actions}
        changed = True
        while changed:
            changed = False
            states = [start_state]
            for m in moves:
                states.append(self.apply_move(states[-1], by_name[m][1]))
            index = {}
            for i, s in enumerate(states):
                index.setdefault(s, []).append(i)
            for i in range(len(states)):
                # same state visited again later: drop the loop
                later = [j for j in index[states[i]] if j > i]
                if later:
                    moves = moves[:i] + moves[later[-1]:]
                    changed = True
                    break
                # a later state is one move away: jump straight there
                best = None
                for name, perm, _ in self.actions:
                    nxt = self.apply_move(states[i], perm)
                    for j in index.get(nxt, []):
                        if j > i + 1 and (best is None or j > best[0]):
                            best = (j, name)
                if best:
                    moves = moves[:i] + [best[1]] + moves[best[0]:]
                    changed = True
                    break
        return moves

    def solve(self, start_state, iterations=50_000, time_limit=10.0):
        solution = super().solve(start_state, iterations, time_limit)
        if solution and self.guided_rollouts:
            solution = self.shorten(start_state, solution)
            self.stats["solution_length"] = len(solution)
        return solution


def make_solver(kind, seed, rollout_depth=20):
    common = dict(solved=rcm.SOLVED, actions=rcm.ACTIONS,
                  apply_move=rcm.apply_move, rollout_depth=rollout_depth,
                  seed=seed, history_interval=0)
    if kind == "MCTS (sticker)":
        return rcm.MCTSCubeSolver(**common)
    if kind == "MCTS (PDB)":
        return PDBMCTSSolver(**common)
    if kind == "MCTS (PDB + guided)":
        return PDBMCTSSolver(guided_rollouts=True, **common)
    raise ValueError(kind)


if __name__ == "__main__":
    start = rca.scramble_cube("L2 U2 B L' B2 R2 U")
    for kind in ("MCTS (sticker)", "MCTS (PDB)", "MCTS (PDB + guided)"):
        solver = make_solver(kind, seed=1)
        path = solver.solve(start, iterations=50_000, time_limit=5.0)
        print(f"{kind:20s} solved={path is not None} "
              f"length={len(path) if path else None} "
              f"time={solver.stats['elapsed_seconds']:.2f}s "
              f"nodes={solver.stats['expanded_nodes']}")
