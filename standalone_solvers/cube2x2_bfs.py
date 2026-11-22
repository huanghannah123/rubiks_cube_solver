"""
2x2x2 Rubik's Cube solver with BFS  (Lesson 1 -> Lesson 2)
==========================================================

Contents
--------
Part 1: BFS on a simple graph (fixed for Python 3)
Part 2: Rubik's cube notation  (U D L R F B, prime ', and 2)
Part 3: The 2x2x2 Cube class   (permutation + orientation)
Part 4: All 18 moves           (U U' U2  D D' D2  L L' L2  R R' R2  F F' F2  B B' B2)
Part 5: BFS solver             (finds the SHORTEST solution)
Part 6: Tests + demo           (run:  python cube2x2_bfs.py)
"""

from collections import defaultdict, deque
import random


# =============================================================================
# Part 1: BFS on a simple graph
# =============================================================================
# Fixes compared to the lesson version:
#   * print s,          -> Python 2 syntax. Python 3 uses print(s, end=" ")
#   * visited = [False] * len(self.graph)
#                       -> node 3 has no outgoing edges, so it is not a key in
#                          the graph. len(graph) == 3 and visited[3] crashes
#                          with IndexError. A set() works for any node labels.
#   * queue.pop(0)      -> O(n) on a list. deque.popleft() is O(1).

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)   # {node: [neighbors]}

    def add_edge(self, u, v):
        self.graph[u].append(v)          # edge u ---> v

    def BFS(self, s):
        visited = {s}                    # set of visited nodes
        queue = deque([s])               # first in, first out
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)           # "process" the node
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return order


# =============================================================================
# Part 2: Notation
# =============================================================================
#   U = Up, D = Down, L = Left, R = Right, F = Front, B = Back
#   X   -> turn face X 90 degrees clockwise (as if you were looking at that face)
#   X'  -> counterclockwise   (same as doing X three times)
#   X2  -> 180 degrees        (same as doing X two times)
#
# Corner positions (each name lists the 3 faces the corner touches):
#
#        UBL(2) ------ URB(1)          top layer, seen from above
#          |             |
#        UFL(3) ------ UFR(0)          <- front
#
#        DBL(6) ------ DRB(5)          bottom layer, seen from above
#          |             |
#        DLF(7) ------ DFR(4)          <- front
#
#   0 = UFR   1 = URB   2 = UBL   3 = UFL
#   4 = DFR   5 = DRB   6 = DBL   7 = DLF
#   (Each bottom corner sits directly under the top corner with index - 4.)
#
# Orientation of a corner (0, 1 or 2):
#   Every corner has exactly one U or D colored sticker (white/yellow).
#   0 = that sticker faces U or D (correct)
#   1 = the corner is twisted one step clockwise
#   2 = twisted two steps clockwise (= one step counterclockwise)
#   Rule: the sum of all 8 orientations is always a multiple of 3.


# =============================================================================
# Part 3: Cube class
# =============================================================================
SOLVED_PERMUTATION = tuple(range(8))     # (0, 1, 2, 3, 4, 5, 6, 7)
SOLVED_ORIENTATION = (0,) * 8            # (0, 0, 0, 0, 0, 0, 0, 0)


class Cube:
    def __init__(self, permutation=None, orientation=None):
        if permutation is None:
            permutation = SOLVED_PERMUTATION
        if orientation is None:
            orientation = SOLVED_ORIENTATION
        # permutation[i] = which corner piece is sitting in position i
        # orientation[i] = twist (0/1/2) of the piece sitting in position i
        self.permutation = tuple(permutation)
        self.orientation = tuple(orientation)

    def state(self):
        return (self.permutation, self.orientation)

    def is_solved(self):
        return (self.permutation == SOLVED_PERMUTATION
                and self.orientation == SOLVED_ORIENTATION)

    def __hash__(self):              # lets a Cube be used in a set / dict key
        return hash(self.state())

    def __eq__(self, other):
        return isinstance(other, Cube) and self.state() == other.state()

    def __repr__(self):
        return f"Cube(permutation={self.permutation}, orientation={self.orientation})"

    # ---- moves -------------------------------------------------------------
    def apply_move(self, move):
        """Return a NEW cube after one move, e.g. 'R', "U'", 'F2'."""
        perm_table, twist_table = MOVES[move]
        new_perm = tuple(self.permutation[perm_table[i]] for i in range(8))
        new_ori = tuple((self.orientation[perm_table[i]] + twist_table[i]) % 3
                        for i in range(8))
        return Cube(new_perm, new_ori)

    def apply_sequence(self, sequence):
        """Apply a string like "R U R' U'" or a list ['R', 'U', ...]."""
        if isinstance(sequence, str):
            sequence = sequence.split()
        cube = self
        for move in sequence:
            cube = cube.apply_move(move)
        return cube


# =============================================================================
# Part 4: The moves
# =============================================================================
# Rule for every move:
#     new_permutation[i] = old_permutation[ perm[i] ]
#     new_orientation[i] = (old_orientation[ perm[i] ] + twist[i]) % 3
#
# Example U (clockwise from above):  UFR <- URB <- UBL <- UFL <- UFR
#     perm = (1, 2, 3, 0, 4, 5, 6, 7)
#     new position 0 (UFR) gets old position 1 (URB), and so on.
#   Note: the lesson had (3, 0, 1, 2, ...) for U. With this rule that is
#   actually U' (counterclockwise). Easy to mix up!
#
# U and D never twist corners (twist all 0).
# R, L, F, B twist the 4 corners they move: +1 on two of them, +2 on the other two.
# These tables were derived from a 3D model of the cube and are checked in tests.

BASE_MOVES = {
    #       perm                        twist
    'U': ((1, 2, 3, 0, 4, 5, 6, 7), (0, 0, 0, 0, 0, 0, 0, 0)),
    'D': ((0, 1, 2, 3, 7, 4, 5, 6), (0, 0, 0, 0, 0, 0, 0, 0)),
    'R': ((4, 0, 2, 3, 5, 1, 6, 7), (2, 1, 0, 0, 1, 2, 0, 0)),
    'L': ((0, 1, 6, 2, 4, 5, 7, 3), (0, 0, 2, 1, 0, 0, 1, 2)),
    'F': ((3, 1, 2, 7, 0, 5, 6, 4), (1, 0, 0, 2, 2, 0, 0, 1)),
    'B': ((0, 5, 1, 3, 4, 6, 2, 7), (0, 2, 1, 0, 0, 1, 2, 0)),
}


def compose(move_a, move_b):
    """Table for 'do move_a, then move_b'."""
    pa, ta = move_a
    pb, tb = move_b
    perm = tuple(pa[pb[i]] for i in range(8))
    twist = tuple((ta[pb[i]] + tb[i]) % 3 for i in range(8))
    return perm, twist


def build_all_moves():
    moves = {}
    for face, quarter in BASE_MOVES.items():
        half = compose(quarter, quarter)        # X2 = X X
        prime = compose(half, quarter)          # X' = X X X
        moves[face] = quarter
        moves[face + "2"] = half
        moves[face + "'"] = prime
    return moves


MOVES = build_all_moves()   # 18 moves: U U2 U' D D2 D' ... B B2 B'

ALL_MOVES = list(MOVES)                                   # 18 moves
URF_MOVES = ["U", "U'", "U2", "R", "R'", "R2", "F", "F'", "F2"]
# On a 2x2x2, U/R/F alone can solve anything that was scrambled with U/R/F
# (the DBL corner never moves). 9 moves instead of 18 -> much faster BFS.


def inverse_move(move):
    if move.endswith("'"):
        return move[0]
    if move.endswith("2"):
        return move
    return move + "'"


def inverse_sequence(sequence):
    if isinstance(sequence, str):
        sequence = sequence.split()
    return [inverse_move(m) for m in reversed(sequence)]


def random_scramble(length, moves=URF_MOVES, seed=None):
    """Random scramble that never turns the same face twice in a row."""
    rng = random.Random(seed)
    scramble = []
    while len(scramble) < length:
        m = rng.choice(moves)
        if scramble and scramble[-1][0] == m[0]:
            continue
        scramble.append(m)
    return scramble


# =============================================================================
# Part 5: BFS solver
# =============================================================================
# Same idea as the graph BFS:
#   node      = a cube state
#   neighbors = the states you get by doing one move
#   visited   = a dict  state -> (parent_state, move)  (also lets us rebuild the path)
# Because BFS explores all 1-move states, then all 2-move states, ...
# the first time we reach the solved state we have a SHORTEST solution.

def bfs_solve(start, moves=URF_MOVES, max_depth=14):
    if start.is_solved():
        return []

    parent = {start: (None, None)}          # visited + how we got here
    queue = deque([(start, 0)])             # (cube, depth)

    while queue:
        cube, depth = queue.popleft()
        if depth >= max_depth:
            continue
        last_move = parent[cube][1]
        for move in moves:
            # pruning: turning the same face twice in a row is never useful
            if last_move is not None and move[0] == last_move[0]:
                continue
            nxt = cube.apply_move(move)
            if nxt in parent:
                continue
            parent[nxt] = (cube, move)
            if nxt.is_solved():
                # walk back through parents to rebuild the move list
                path = []
                node = nxt
                while parent[node][0] is not None:
                    node, m = parent[node]
                    path.append(m)
                return list(reversed(path))
            queue.append((nxt, depth + 1))

    return None   # not found within max_depth


# =============================================================================
# Part 6: Tests + demo
# =============================================================================
def run_tests():
    solved = Cube()

    # every quarter turn done 4 times returns to solved; X X' is solved
    for face in "UDLRFB":
        assert solved.apply_sequence([face] * 4).is_solved(), face
        assert solved.apply_sequence([face, face + "'"]).is_solved(), face
        assert solved.apply_sequence([face + "2"] * 2).is_solved(), face
        assert not solved.apply_move(face).is_solved(), face

    # sum of orientations stays a multiple of 3, permutation stays valid
    cube = solved.apply_sequence(random_scramble(50, ALL_MOVES, seed=1))
    assert sum(cube.orientation) % 3 == 0
    assert sorted(cube.permutation) == list(range(8))

    # famous fact: (R U R' U') repeated 6 times returns to solved
    sexy = "R U R' U'"
    c = solved
    for k in range(1, 7):
        c = c.apply_sequence(sexy)
        assert c.is_solved() == (k == 6)

    # scramble followed by its inverse is solved
    s = random_scramble(20, ALL_MOVES, seed=2)
    assert solved.apply_sequence(s).apply_sequence(inverse_sequence(s)).is_solved()

    # BFS solutions actually solve the cube and are never longer than the scramble
    for seed in range(5):
        s = random_scramble(6, URF_MOVES, seed=seed)
        start = solved.apply_sequence(s)
        sol = bfs_solve(start, URF_MOVES)
        assert sol is not None and len(sol) <= len(s)
        assert start.apply_sequence(sol).is_solved()

    # graph BFS from the lesson
    g = Graph()
    for u, v in [(0, 1), (0, 2), (1, 2), (2, 0), (2, 3)]:
        g.add_edge(u, v)
    assert g.BFS(2) == [2, 0, 3, 1]

    print("All tests passed.\n")


if __name__ == "__main__":
    run_tests()

    # --- Part 1 demo ---
    g = Graph()
    for u, v in [(0, 1), (0, 2), (1, 2), (2, 0), (2, 3)]:
        g.add_edge(u, v)
    print("BFS starting from vertex 2:", *g.BFS(2))

    # --- Part 4 demo: one move ---
    print("\nAfter R:", Cube().apply_move("R"))

    # --- Part 5 demo: scramble and solve ---
    scramble = random_scramble(7, URF_MOVES, seed=42)
    cube = Cube().apply_sequence(scramble)
    print("\nScramble:", " ".join(scramble))
    print(cube)
    solution = bfs_solve(cube)
    print("BFS solution:", " ".join(solution), f"({len(solution)} moves)")
    print("Solved after solution?", cube.apply_sequence(solution).is_solved())
