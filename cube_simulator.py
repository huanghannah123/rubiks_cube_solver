"""
2x2x2 Rubik's Cube BFS Simulator
================================
A visual front end for cube2x2_bfs.py, built for a science-fair demo.

Run:      python cube_simulator.py          (opens the simulator window)
Tests:    python cube_simulator.py --test   (checks everything, no window)

Needs only standard Python 3 (tkinter is included with the normal
python.org installer on Windows and macOS). Keep this file in the same
folder as cube2x2_bfs.py.

What the judge can do
---------------------
* Watch a 3D cube (front view + back view) and a flat "net" of all 6 faces
* Scramble it, or turn it by hand with the U / R / F buttons (or keyboard)
* Press "Solve with BFS" and watch the search live: states discovered,
  depth, speed, and a bar chart of how many positions sit at each depth
* Play the shortest solution back move by move, with smooth turn animation
* "Map entire state space": BFS from the solved cube over all 3,674,160
  positions, reproducing the published distance table (God's number = 11)
* Benchmark the optimized BFS against the original bfs_solve()

How the BFS here relates to cube2x2_bfs.py
------------------------------------------
It is the SAME breadth-first search (same moves, same "don't turn the same
face twice" pruning, same shortest-path guarantee). The difference is how a
cube state is stored:

  original:   a Cube object (two tuples) in a dict   -> ~1.5 GB, ~90 s worst case
  optimized:  one integer 0 .. 3,674,159 per state, and a bytearray of
              3,674,160 bytes as the "visited" list   -> ~60 MB, a few seconds

The integer move tables are generated FROM Cube.apply_move, so every move
still comes from your own permutation/orientation tables.
"""

import math
import sys
import threading
import time
from collections import deque

from cube2x2_bfs import (Cube, URF_MOVES, SOLVED_ORIENTATION, bfs_solve,
                         inverse_move, random_scramble)

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:          # allows --test on machines without tkinter
    tk = None


# =============================================================================
# 1. Sticker geometry: turning (permutation, orientation) into colors
# =============================================================================
# Each corner position lists its 3 faces in clockwise order, starting with
# the U or D face. Position numbers match cube2x2_bfs.py.
FACELETS = [("U", "R", "F"), ("U", "B", "R"), ("U", "L", "B"), ("U", "F", "L"),
            ("D", "F", "R"), ("D", "R", "B"), ("D", "B", "L"), ("D", "L", "F")]

# 3D coordinates of each corner position (x = right, y = up, z = front)
POSITIONS = [(1, 1, 1), (1, 1, -1), (-1, 1, -1), (-1, 1, 1),
             (1, -1, 1), (1, -1, -1), (-1, -1, -1), (-1, -1, 1)]

NORMALS = {"U": (0, 1, 0), "D": (0, -1, 0), "R": (1, 0, 0),
           "L": (-1, 0, 0), "F": (0, 0, 1), "B": (0, 0, -1)}

COLORS = {"U": "#FFFFFF", "D": "#FFD500", "F": "#009B48",
          "B": "#0046AD", "R": "#C41E3A", "L": "#FF5800"}

COLOR_NAMES = {"U": "white", "D": "yellow", "F": "green",
               "B": "blue", "R": "red", "L": "orange"}


def sticker_colors(cube):
    """{(position, face): home face of the sticker shown there}.

    The piece in position i is cube.permutation[i], twisted by
    cube.orientation[i]. Its sticker that now faces FACELETS[i][k]
    originally faced FACELETS[piece][(k - twist) % 3].
    (Checked against an independent 3D model in run_tests.)
    """
    out = {}
    for i in range(8):
        piece = cube.permutation[i]
        twist = cube.orientation[i]
        for k in range(3):
            out[(i, FACELETS[i][k])] = FACELETS[piece][(k - twist) % 3]
    return out


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def rotate(v, axis, angle):
    """Rotate vector v around a unit axis by angle (radians), Rodrigues."""
    c, s = math.cos(angle), math.sin(angle)
    k = cross(axis, v)
    d = dot(axis, v)
    return tuple(v[i] * c + k[i] * s + axis[i] * d * (1 - c) for i in range(3))


def move_angle(move):
    """Clockwise (seen from the face) is -90 degrees around its normal."""
    if move.endswith("'"):
        return math.pi / 2
    if move.endswith("2"):
        return -math.pi
    return -math.pi / 2


# Flat net layout: where each face goes and which 3D directions are
# "right" and "down" when you look straight at that face.
NET_LAYOUT = {   # face: (grid col, grid row, right vector, down vector)
    "U": (2, 0, (1, 0, 0), (0, 0, 1)),
    "L": (0, 2, (0, 0, 1), (0, -1, 0)),
    "F": (2, 2, (1, 0, 0), (0, -1, 0)),
    "R": (4, 2, (0, 0, -1), (0, -1, 0)),
    "B": (6, 2, (-1, 0, 0), (0, -1, 0)),
    "D": (2, 4, (1, 0, 0), (0, 0, -1)),
}


def net_slots():
    """[(face, col, row, position)] for all 24 stickers on the flat net."""
    slots = []
    for face, (gc, gr, right, down) in NET_LAYOUT.items():
        for i in range(8):
            if face in FACELETS[i]:
                p = POSITIONS[i]
                col = gc + (0 if dot(p, right) < 0 else 1)
                row = gr + (0 if dot(p, down) < 0 else 1)
                slots.append((face, col, row, i))
    return slots


NET_SLOTS = net_slots()


# =============================================================================
# 2. Optimized BFS: every state becomes one integer
# =============================================================================
# With only U, R, F moves the DBL corner (position 6) never moves or twists,
# so a state is fully described by
#   * how the other 7 pieces are arranged  -> 7! = 5,040 possibilities
#   * the twist of 6 of them (the 7th is forced by "sum % 3 == 0") -> 3^6 = 729
# 5,040 x 729 = 3,674,160 = exactly the number of 2x2x2 positions.

MOVABLE = [0, 1, 2, 3, 4, 5, 7]          # every position except DBL (6)
N_PERM = 5040
N_ORI = 729
N_STATES = N_PERM * N_ORI                # 3,674,160

# Published distance table (Jaap's Puzzle Page), half-turn metric
KNOWN_LAYERS = [1, 9, 54, 321, 1847, 9992, 50136, 227536,
                870072, 1887748, 623800, 2644]


def encode_perm(perm):
    ranks = [MOVABLE.index(perm[p]) for p in MOVABLE]   # 7 numbers, 0..6
    index = 0
    for i in range(7):                                  # Lehmer code
        smaller = sum(1 for j in range(i + 1, 7) if ranks[j] < ranks[i])
        index = index * (7 - i) + smaller
    return index


def decode_perm(index):
    digits = []
    for base in range(1, 8):                            # bases 1..7
        digits.append(index % base)
        index //= base
    digits.reverse()
    pool = list(range(7))
    ranks = [pool.pop(d) for d in digits]
    perm = [6] * 8
    for p, r in zip(MOVABLE, ranks):
        perm[p] = MOVABLE[r]
    return tuple(perm)


def encode_ori(ori):
    index = 0
    for p in range(6):
        index = index * 3 + ori[p]
    return index


def decode_ori(index):
    ori = [0] * 8
    for p in range(5, -1, -1):
        ori[p] = index % 3
        index //= 3
    ori[7] = (-sum(ori)) % 3                            # sum must be 0 mod 3
    return tuple(ori)


def can_encode(cube):
    return cube.permutation[6] == 6 and cube.orientation[6] == 0


def encode(cube):
    return encode_perm(cube.permutation) * N_ORI + encode_ori(cube.orientation)


def decode(state):
    p, o = divmod(state, N_ORI)
    return Cube(decode_perm(p), decode_ori(o))


class MoveTables:
    """Integer move tables, generated from Cube.apply_move."""

    def __init__(self, moves=URF_MOVES):
        self.moves = list(moves)
        self.face = [m[0] for m in self.moves]
        self.inverse = [self.moves.index(inverse_move(m)) for m in self.moves]
        solved_perm = tuple(range(8))
        self.perm = []
        self.ori = []
        for m in self.moves:
            # A move changes the permutation using only the permutation,
            # and the orientation using only the orientation, so the two
            # parts get separate (small) tables.
            self.perm.append([
                encode_perm(Cube(decode_perm(i), SOLVED_ORIENTATION)
                            .apply_move(m).permutation)
                for i in range(N_PERM)])
            self.ori.append([
                encode_ori(Cube(solved_perm, decode_ori(i))
                           .apply_move(m).orientation)
                for i in range(N_ORI)])


class BFSWorker(threading.Thread):
    """Breadth-first search on integer states, run in the background.

    goal=0 (solved): stop at the first time the solved state is found.
    goal=None:       explore everything reachable (maps the state space).
    The GUI reads the progress attributes while it runs.
    """

    def __init__(self, tables, start, goal=0):
        super().__init__(daemon=True)
        self.t = tables
        self.start_state = start
        self.goal = goal
        # progress, read by the GUI
        self.discovered = 1
        self.depth = 0
        self.layers = [1]
        self.solution = None
        self.done = False
        self.cancelled = False
        self.elapsed = 0.0
        self.t0 = time.perf_counter()

    def cancel(self):
        self.cancelled = True

    def run(self):
        t = self.t
        pt, ot, face, n_moves = t.perm, t.ori, t.face, len(t.moves)
        START = 255
        # visited[s] = 0 if unseen, otherwise (index of the move that reached it) + 1
        visited = bytearray(N_STATES)
        visited[self.start_state] = START
        self.t0 = time.perf_counter()

        if self.start_state == self.goal:
            self.solution = []
            return self._finish()

        frontier = [self.start_state]
        discovered = 1
        found = None
        while frontier and found is None and not self.cancelled:
            self.depth += 1
            nxt = []
            append = nxt.append
            for count, s in enumerate(frontier):
                if count & 0x3FFF == 0:            # every 16,384 nodes
                    if self.cancelled:
                        break
                    self.discovered = discovered + len(nxt)
                    self.elapsed = time.perf_counter() - self.t0
                    self._partial = len(nxt)
                last = visited[s]
                last_face = face[last - 1] if last != START else None
                p, o = divmod(s, N_ORI)
                for mi in range(n_moves):
                    # pruning: never turn the same face twice in a row
                    if face[mi] == last_face:
                        continue
                    n = pt[mi][p] * N_ORI + ot[mi][o]
                    if visited[n]:
                        continue
                    visited[n] = mi + 1
                    append(n)
                    if n == self.goal:
                        found = n
                        break
                if found is not None:
                    break
            discovered += len(nxt)
            self.layers = self.layers + [len(nxt)]
            self.discovered = discovered
            frontier = nxt

        if found is not None:
            # walk back to the start by undoing the recorded moves
            path = []
            s = found
            while visited[s] != START:
                mi = visited[s] - 1
                path.append(t.moves[mi])
                inv = t.inverse[mi]
                p, o = divmod(s, N_ORI)
                s = pt[inv][p] * N_ORI + ot[inv][o]
            self.solution = list(reversed(path))
        self._finish()

    def _finish(self):
        self.elapsed = time.perf_counter() - self.t0
        self.done = True


# =============================================================================
# 3. 3D drawing
# =============================================================================
def make_camera(yaw_deg, pitch_deg):
    cy, sy = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    cp, sp = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))

    def view(v):
        x, y, z = v
        x1 = x * cy + z * sy
        z1 = -x * sy + z * cy
        return (x1, y * cp - z1 * sp, y * sp + z1 * cp)   # (x, y, depth)
    return view


FRONT_CAMERA = make_camera(-35, 28)     # shows U, F, R
BACK_CAMERA = make_camera(145, -28)     # shows D, B, L

FACE_CORNERS = [(-1, -1), (1, -1), (1, 1), (-1, 1)]


def tangent_axes(n):
    if n[0] != 0:
        return (0, 1, 0), (0, 0, 1)
    if n[1] != 0:
        return (1, 0, 0), (0, 0, 1)
    return (1, 0, 0), (0, 1, 0)


def draw_cube_3d(canvas, colors, camera, cx, cy, scale, anim=None, tag="cube"):
    """Draw 8 cubies. anim = (face, angle) rotates that layer mid-turn."""
    items = []
    for i in range(8):
        center = tuple(0.5 * c for c in POSITIONS[i])
        axis, angle = None, 0.0
        if anim and dot(POSITIONS[i], NORMALS[anim[0]]) > 0:
            axis, angle = NORMALS[anim[0]], anim[1]

        def world(v):
            return rotate(v, axis, angle) if axis else v

        quads = []
        for face, n in NORMALS.items():
            u, w = tangent_axes(n)
            for size, push, fill in ((0.5, 0.5, "#141414"), (0.43, 0.502, None)):
                outward = face in FACELETS[i]
                if fill is None:                  # sticker layer
                    if not outward:
                        continue
                    fill = COLORS[colors[(i, face)]]
                pts = [tuple(center[k] + push * n[k] + size * (a * u[k] + b * w[k])
                             for k in range(3)) for a, b in FACE_CORNERS]
                rn = camera(world(n))
                if rn[2] <= 1e-6:                 # facing away: hidden
                    continue
                quads.append([camera(world(p)) for p in pts] + [fill])
        depth = camera(world(center))[2]
        items.append((depth, quads))

    items.sort(key=lambda it: it[0])              # far cubies first
    for _, quads in items:
        for q in quads:
            fill = q[-1]
            flat = []
            for (x, y, _) in q[:-1]:
                flat += [cx + scale * x, cy - scale * y]
            canvas.create_polygon(flat, fill=fill, outline="#141414",
                                  width=1.5, tags=tag)


# =============================================================================
# 4. The simulator window
# =============================================================================
BG = "#1f2329"          # cube canvas background
PANEL = "#f4f5f7"       # control panel background
ACCENT = "#2563eb"
MUTED = "#6b7280"


class Simulator:
    def __init__(self, root):
        self.root = root
        root.title("2x2x2 Rubik's Cube - BFS Simulator")
        root.configure(bg=PANEL)
        root.minsize(1150, 700)
        root.geometry("1230x730")

        self.tables = MoveTables()
        self.cube = Cube()
        self.scramble = []
        self.history = []                # every move done since the last reset
        self.solution = None
        self.sol_index = 0
        self.playing = False
        self.animating = False
        self.queue = deque()             # moves waiting to be animated
        self.worker = None
        self.anim_label = ""
        self.last_layers = []
        self.mode = None                 # "solve" or "map"

        self._build_ui()
        self._bind_keys()
        self.redraw()

    # ---------------------------------------------------------------- layout
    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=PANEL)
        style.configure("TLabel", background=PANEL, font=("Helvetica", 11))
        style.configure("Head.TLabel", font=("Helvetica", 12, "bold"))
        style.configure("Title.TLabel", font=("Helvetica", 17, "bold"))
        style.configure("Muted.TLabel", foreground=MUTED, font=("Helvetica", 10))
        style.configure("Stat.TLabel", font=("Helvetica", 15, "bold"))
        style.configure("TButton", font=("Helvetica", 11), padding=(8, 4))
        style.configure("Big.TButton", font=("Helvetica", 12, "bold"),
                        padding=(10, 6), foreground="white", background=ACCENT)
        style.map("Big.TButton", background=[("disabled", "#9ca3af"),
                                             ("active", "#1d4ed8")])
        style.configure("Move.TButton", font=("Courier", 13, "bold"),
                        padding=(4, 3), width=4)
        style.configure("TScale", background=PANEL)
        style.configure("Horizontal.TProgressbar", troughcolor="#e5e7eb",
                        background=ACCENT)

        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        # left column: drawing canvas + playback bar underneath
        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(left, width=760, height=560, bg=BG,
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        play = ttk.Frame(left, padding=(0, 8, 0, 0))
        play.pack(fill="x")

        # right column: controls
        side = ttk.Frame(main, padding=(14, 0, 0, 0))
        side.pack(side="right", fill="y")

        ttk.Label(side, text="2x2x2 BFS Solver", style="Title.TLabel").pack(anchor="w")
        ttk.Label(side, text="Breadth-first search finds the SHORTEST solution",
                  style="Muted.TLabel").pack(anchor="w")

        # scramble
        box = self._section(side, "1. Scramble")
        row = ttk.Frame(box)
        row.pack(fill="x")
        self.scramble_btn = ttk.Button(row, text="Scramble (S)", command=self.do_scramble)
        self.scramble_btn.pack(side="left")
        ttk.Label(row, text="  length").pack(side="left")
        self.len_var = tk.IntVar(value=8)
        self.len_label = ttk.Label(row, text="8", width=3)
        len_scale = ttk.Scale(row, from_=1, to=14, orient="horizontal", length=95,
                              command=self._on_len)
        len_scale.pack(side="left", padx=4)
        self.len_label.pack(side="left")
        len_scale.set(8)
        self.reset_btn = ttk.Button(row, text="Reset", command=self.do_reset)
        self.reset_btn.pack(side="left", padx=(4, 0))

        # manual moves
        box = self._section(side, "2. Turn it yourself")
        grid = ttk.Frame(box)
        grid.pack(anchor="w")
        self.move_buttons = []
        for r, face in enumerate("URF"):
            for c, suffix in enumerate(["", "'", "2"]):
                m = face + suffix
                b = ttk.Button(grid, text=m, style="Move.TButton",
                               command=lambda m=m: self.user_move(m))
                b.grid(row=r, column=c, padx=2, pady=2)
                self.move_buttons.append(b)
        ttk.Label(grid, text="Keys: u r f = clockwise\nShift+U R F = counter-\nclockwise",
                  style="Muted.TLabel", justify="left").grid(
            row=0, column=3, rowspan=3, padx=(12, 0), sticky="w")

        # solve
        box = self._section(side, "3. Search")
        row = ttk.Frame(box)
        row.pack(fill="x")
        self.solve_btn = ttk.Button(row, text="Solve with BFS  (Enter)",
                                    style="Big.TButton", command=self.do_solve)
        self.solve_btn.pack(side="left")
        self.cancel_btn = ttk.Button(row, text="Cancel", command=self.do_cancel,
                                     state="disabled")
        self.cancel_btn.pack(side="left", padx=6)

        stats = ttk.Frame(box)
        stats.pack(fill="x", pady=(8, 0))
        self.stat_vars = {}
        for col, (key, label) in enumerate([("found", "States found"),
                                            ("depth", "Depth"),
                                            ("time", "Time"),
                                            ("speed", "States/sec")]):
            cell = ttk.Frame(stats)
            cell.grid(row=0, column=col, sticky="w", padx=(0, 14))
            v = tk.StringVar(value="-")
            self.stat_vars[key] = v
            ttk.Label(cell, textvariable=v, style="Stat.TLabel").pack(anchor="w")
            ttk.Label(cell, text=label, style="Muted.TLabel").pack(anchor="w")

        self.progress = ttk.Progressbar(box, maximum=N_STATES, length=340)
        self.progress.pack(fill="x", pady=(8, 0))
        self.progress_label = ttk.Label(box, text=f"0 of {N_STATES:,} positions explored",
                                        style="Muted.TLabel")
        self.progress_label.pack(anchor="w")

        self.chart = tk.Canvas(box, width=350, height=140, bg="white",
                               highlightthickness=1, highlightbackground="#d1d5db")
        self.chart.pack(pady=(6, 0))

        row = ttk.Frame(box)
        row.pack(fill="x", pady=(6, 0))
        self.map_btn = ttk.Button(row, text="Map entire state space",
                                  command=self.do_map)
        self.map_btn.pack(side="left")
        self.bench_btn = ttk.Button(row, text="Benchmark original",
                                    command=self.do_benchmark, state="disabled")
        self.bench_btn.pack(side="left", padx=6)

        # playback (under the cube, left column)
        row = ttk.Frame(play)
        row.pack(fill="x")
        ttk.Label(row, text="4. Watch the solution", style="Head.TLabel").pack(side="left")
        self.back_btn = ttk.Button(row, text="< Back", command=self.step_back)
        self.back_btn.pack(side="left", padx=(14, 0))
        self.play_btn = ttk.Button(row, text="Play  (Space)", command=self.toggle_play,
                                   width=13)
        self.play_btn.pack(side="left", padx=6)
        self.fwd_btn = ttk.Button(row, text="Step >", command=self.step_forward)
        self.fwd_btn.pack(side="left")
        ttk.Label(row, text="   Speed").pack(side="left")
        self.speed = tk.DoubleVar(value=0.55)
        ttk.Scale(row, from_=0.1, to=1.0, variable=self.speed, orient="horizontal",
                  length=140).pack(side="left", padx=6)

        self.sol_text = tk.Text(play, height=1, width=60, font=("Courier", 14, "bold"),
                                relief="flat", bg="white", wrap="none",
                                highlightthickness=1, highlightbackground="#d1d5db",
                                padx=6, pady=4)
        self.sol_text.tag_configure("done", foreground="#9ca3af")
        self.sol_text.tag_configure("current", foreground="white", background=ACCENT)
        self.sol_text.tag_configure("todo", foreground="#111827")
        self.sol_text.tag_configure("note", foreground="#9ca3af",
                                    font=("Helvetica", 11))
        self.sol_text.configure(state="disabled")
        self.sol_text.pack(fill="x", pady=(6, 0))

        self.status = ttk.Label(play, text="Ready. Scramble the cube to begin.",
                                style="Muted.TLabel", wraplength=740, justify="left")
        self.status.pack(anchor="w", pady=(6, 0))
        play.bind("<Configure>",
                  lambda e: self.status.configure(wraplength=max(300, e.width - 10)))

        self._update_buttons()
        self._draw_chart([], None)

    def _section(self, parent, title):
        ttk.Label(parent, text=title, style="Head.TLabel").pack(anchor="w", pady=(8, 2))
        f = ttk.Frame(parent)
        f.pack(fill="x")
        return f

    def _bind_keys(self):
        for face in "urf":
            self.root.bind(face, lambda e, f=face.upper(): self.user_move(f))
            self.root.bind(face.upper(), lambda e, f=face.upper(): self.user_move(f + "'"))
        self.root.bind("s", lambda e: self.do_scramble())
        self.root.bind("<Return>", lambda e: self.do_solve())
        self.root.bind("<space>", lambda e: self.toggle_play())
        self.root.bind("<Right>", lambda e: self.step_forward())
        self.root.bind("<Left>", lambda e: self.step_back())

    def _on_len(self, value):
        n = int(round(float(value)))
        self.len_var.set(n)
        self.len_label.configure(text=str(n))

    def set_status(self, text):
        self.status.configure(text=text)

    def busy(self):
        return self.worker is not None and not self.worker.done

    def _update_buttons(self):
        searching = self.busy()
        moving = self.animating or bool(self.queue)
        has_sol = self.solution is not None and len(self.solution) > 0
        normal = lambda ok: "normal" if ok else "disabled"
        self.scramble_btn.configure(state=normal(not searching and not moving))
        self.reset_btn.configure(state=normal(not searching and not moving))
        for b in self.move_buttons:
            b.configure(state=normal(not searching and not self.playing))
        self.solve_btn.configure(state=normal(not searching and not moving))
        self.map_btn.configure(state=normal(not searching))
        self.cancel_btn.configure(state=normal(searching))
        self.back_btn.configure(state=normal(has_sol and self.sol_index > 0
                                             and not moving and not self.playing))
        self.fwd_btn.configure(state=normal(has_sol and self.sol_index < len(self.solution)
                                            and not moving and not self.playing))
        self.play_btn.configure(state=normal(has_sol),
                                text="Pause  (Space)" if self.playing else "Play  (Space)")
        self.bench_btn.configure(state=normal(has_sol and not searching
                                              and self.mode == "solve"))

    # --------------------------------------------------------------- drawing
    def redraw(self, anim=None):
        c = self.canvas
        c.delete("all")
        w = max(c.winfo_width(), 700)
        h = max(c.winfo_height(), 540)
        colors = sticker_colors(self.cube)
        top = h - 200                      # where the bottom panels start

        c.create_text(18, 18, anchor="w", fill="#e5e7eb",
                      font=("Helvetica", 13, "bold"), text="Front view  (Up, Front, Right)")
        c.create_text(w * 0.75, 18, anchor="center", fill="#e5e7eb",
                      font=("Helvetica", 13, "bold"), text="Back view  (Down, Back, Left)")
        size = min(118, (top - 60) / 2.9)
        draw_cube_3d(c, colors, FRONT_CAMERA, w * 0.32, 38 + size * 1.5, size, anim)
        draw_cube_3d(c, colors, BACK_CAMERA, w * 0.75, 30 + size * 1.25, size * 0.62, anim)

        # current move banner (under the back view)
        banner_y = 30 + size * 1.25 + size * 0.62 * 1.6 + 26
        if self.anim_label:
            c.create_text(w * 0.75, banner_y, fill="#fbbf24",
                          font=("Helvetica", 19, "bold"), text=self.anim_label)
        elif self.cube.is_solved():
            c.create_text(w * 0.75, banner_y, fill="#34d399",
                          font=("Helvetica", 19, "bold"), text="SOLVED")

        self._draw_net(c, 22, top + 20, 26)
        self._draw_state_panel(c, 255, top + 20, w)

    def _draw_net(self, c, x0, y0, cell):
        c.create_text(x0, y0 - 14, anchor="w", fill="#e5e7eb",
                      font=("Helvetica", 12, "bold"), text="Unfolded net")
        colors = sticker_colors(self.cube)
        for face, col, row, i in NET_SLOTS:
            x = x0 + col * cell
            y = y0 + 8 + row * cell
            c.create_rectangle(x, y, x + cell - 2, y + cell - 2,
                               fill=COLORS[colors[(i, face)]], outline="#000000", width=1)
        for face, (gc, gr, _, _) in NET_LAYOUT.items():
            x = x0 + gc * cell + cell - 1
            y = y0 + 8 + gr * cell + cell - 1
            c.create_oval(x - 8, y - 8, x + 8, y + 8, fill=BG, outline="")
            c.create_text(x, y, text=face, fill="#ffffff",
                          font=("Helvetica", 10, "bold"))

    def _draw_state_panel(self, c, x0, y0, w):
        """Show the exact data the solver works with."""
        cube = self.cube
        c.create_text(x0, y0 - 14, anchor="w", fill="#e5e7eb",
                      font=("Helvetica", 12, "bold"), text="What the computer sees")
        names = ["UFR", "URB", "UBL", "UFL", "DFR", "DRB", "DBL", "DLF"]
        mono = ("Courier", 11)
        y = y0 + 8
        c.create_text(x0, y, anchor="w", fill="#9ca3af", font=mono,
                      text="position  " + " ".join(f"{n:>3}" for n in names))
        c.create_text(x0, y + 18, anchor="w", fill="#93c5fd", font=mono,
                      text="piece     " + " ".join(f"{names[p]:>3}" for p in cube.permutation))
        c.create_text(x0, y + 36, anchor="w", fill="#fca5a5", font=mono,
                      text="twist     " + " ".join(f"{o:>3}" for o in cube.orientation))
        state = encode(cube) if can_encode(cube) else None
        c.create_text(x0, y + 60, anchor="w", fill="#d1d5db", font=mono,
                      text=f"permutation = {cube.permutation}")
        c.create_text(x0, y + 77, anchor="w", fill="#d1d5db", font=mono,
                      text=f"orientation = {cube.orientation}")
        if state is not None:
            c.create_text(x0, y + 94, anchor="w", fill="#d1d5db", font=mono,
                          text=f"state number = {state:,}  (of {N_STATES:,})")
        c.create_text(x0, y + 120, anchor="w", fill="#9ca3af",
                      font=("Helvetica", 10), width=w - x0 - 16,
                      text=("Each column is one corner slot. 'piece' = which corner sits "
                            "there, 'twist' = 0/1/2 clockwise turns. Twists always add "
                            "up to a multiple of 3."))
        moves = " ".join(self.history[-14:])
        if moves:
            c.create_text(x0, y + 152, anchor="w", fill="#9ca3af", font=mono,
                          width=w - x0 - 16,
                          text=("Moves so far: " + ("... " if len(self.history) > 14 else "")
                                + moves))

    def _draw_chart(self, layers, highlight):
        ch = self.chart
        ch.delete("all")
        W, H = 350, 140
        left, bottom, top = 10, H - 22, 32
        ch.create_text(W / 2, 9, text="Positions found at each depth (log scale)",
                       font=("Helvetica", 9, "bold"), fill="#374151")
        n = max(12, len(layers))
        bw = (W - 2 * left) / n
        log_max = math.log10(2_000_000)
        for d in range(n):
            x = left + d * bw
            ch.create_text(x + bw / 2, H - 10, text=str(d), font=("Helvetica", 8),
                           fill=MUTED)
            if d < len(layers) and layers[d] > 0:
                hgt = (bottom - top) * math.log10(layers[d] + 1) / log_max
                color = "#f59e0b" if d == highlight else ACCENT
                ch.create_rectangle(x + 2, bottom - hgt, x + bw - 2, bottom,
                                    fill=color, outline="")
                ch.create_text(x + bw / 2, bottom - hgt - 6, text=short(layers[d]),
                               font=("Helvetica", 7), fill="#374151")
        ch.create_line(left, bottom, W - left, bottom, fill="#9ca3af")

    # --------------------------------------------------------------- moves
    def user_move(self, move):
        if self.busy() or self.playing:
            return
        self._invalidate_solution()
        self.queue.append((move, None, 1.0))
        self._pump()

    def _pump(self):
        """Animate the next queued move, if any."""
        self._update_buttons()
        if self.animating or not self.queue:
            return
        move, label, speed_mult = self.queue.popleft()
        self.anim_label = label or move
        self.animate(move, speed_mult, self._after_move)

    def _after_move(self):
        self.anim_label = ""
        self.redraw()
        self._update_buttons()
        if self.queue:
            self.root.after(30, self._pump)
        elif self.playing:
            self.root.after(250, self._play_next)

    def animate(self, move, speed_mult, done):
        self.animating = True
        face = move[0]
        total = move_angle(move)
        s = max(0.05, min(1.0, self.speed.get() * speed_mult))
        frames = int(4 + 26 * (1 - s))            # 4 (fast) .. 30 (slow)
        if move.endswith("2"):
            frames = int(frames * 1.5)

        def step(i):
            t = i / frames
            ease = t * t * (3 - 2 * t)            # smooth start and stop
            self.redraw(anim=(face, total * ease))
            if i < frames:
                self.root.after(16, step, i + 1)
            else:
                self.cube = self.cube.apply_move(move)
                self.history.append(move)
                self.animating = False
                done()
        step(1)

    # --------------------------------------------------------------- actions
    def do_scramble(self):
        if self.busy() or self.animating or self.queue:
            return
        self.playing = False
        self._invalidate_solution()
        n = self.len_var.get()
        self.scramble = random_scramble(n, URF_MOVES)
        self.set_status(f"Scramble ({n} moves): {' '.join(self.scramble)}")
        for k, m in enumerate(self.scramble):
            self.queue.append((m, f"Scramble {k + 1}/{n}:  {m}", 2.2))
        self._pump()

    def do_reset(self):
        if self.busy() or self.animating:
            return
        self.queue.clear()
        self.playing = False
        self.cube = Cube()
        self.history = []
        self.scramble = []
        self._invalidate_solution()
        self._reset_stats()
        self.set_status("Reset to the solved cube.")
        self.redraw()

    def _invalidate_solution(self):
        self.solution = None
        self.sol_index = 0
        self._show_solution()
        self._update_buttons()

    def _reset_stats(self):
        for v in self.stat_vars.values():
            v.set("-")
        self.progress["value"] = 0
        self.progress_label.configure(text=f"0 of {N_STATES:,} positions explored")
        self._draw_chart([], None)

    def do_solve(self):
        if self.busy() or self.animating or self.queue:
            return
        if self.cube.is_solved():
            self.set_status("The cube is already solved. Scramble it first!")
            return
        if not can_encode(self.cube):
            self.set_status("This simulator only uses U, R and F moves.")
            return
        self.playing = False
        self._invalidate_solution()
        self._reset_stats()
        self.mode = "solve"
        self.solve_start = self.cube
        self.worker = BFSWorker(self.tables, encode(self.cube), goal=0)
        self.worker.start()
        self.set_status("Searching: BFS checks every position 1 move away, then 2 "
                        "moves away, and so on. The first time it reaches the solved "
                        "cube, that path is guaranteed to be the shortest.")
        self._update_buttons()
        self._poll()

    def do_map(self):
        if self.busy():
            return
        self.playing = False
        self._reset_stats()
        self.mode = "map"
        self.worker = BFSWorker(self.tables, 0, goal=None)
        self.worker.start()
        self.set_status("Mapping: BFS from the solved cube to EVERY reachable "
                        "position. This counts how many positions need 0, 1, 2, ... "
                        "moves, and finds the hardest positions.")
        self._update_buttons()
        self._poll()

    def do_cancel(self):
        if self.worker:
            self.worker.cancel()

    def _poll(self):
        w = self.worker
        layers = list(w.layers)
        partial = getattr(w, "_partial", 0)
        shown = layers + ([partial] if not w.done and partial else [])
        self.stat_vars["found"].set(f"{w.discovered:,}")
        self.stat_vars["depth"].set(str(w.depth))
        elapsed = w.elapsed if w.done else time.perf_counter() - w.t0
        self.stat_vars["time"].set(f"{elapsed:.2f}s")
        if elapsed > 0:
            self.stat_vars["speed"].set(short(w.discovered / elapsed))
        self.progress["value"] = w.discovered
        pct = 100 * w.discovered / N_STATES
        self.progress_label.configure(
            text=f"{w.discovered:,} of {N_STATES:,} positions explored ({pct:.1f}%)")
        self._draw_chart(shown, None)
        if w.done:
            self._search_finished()
        else:
            self.root.after(100, self._poll)

    def _search_finished(self):
        w = self.worker
        self._update_buttons()
        if w.cancelled:
            self.set_status("Search cancelled.")
            return
        if self.mode == "map":
            layers = w.layers[:-1] if w.layers and w.layers[-1] == 0 else w.layers
            self.last_layers = layers
            self._draw_chart(layers, len(layers) - 1)
            match = layers == KNOWN_LAYERS
            self.set_status(
                f"Explored all {sum(layers):,} positions in {w.elapsed:.1f}s. "
                f"The farthest positions need {len(layers) - 1} moves "
                f"({layers[-1]:,} of them), so 'God's number' for the 2x2x2 is "
                f"{len(layers) - 1}. "
                + ("Every layer count matches the published distance table."
                   if match else "Layer counts differ from the published table!"))
            self.stat_vars["depth"].set(str(len(layers) - 1))
            return
        sol = w.solution
        if sol is None:
            self.set_status("No solution found (this should not happen).")
            return
        ok = self.solve_start.apply_sequence(sol).is_solved()
        self.solution = sol
        self.sol_index = 0
        self.stat_vars["depth"].set(str(len(sol)))
        self._draw_chart(w.layers, len(sol))
        self._show_solution()
        extra = ""
        if self.scramble and self.history[-len(self.scramble):] == self.scramble \
                and len(sol) < len(self.scramble):
            extra = f" That's shorter than the {len(self.scramble)}-move scramble!"
        self.set_status(
            f"Found a solution of {len(sol)} moves after looking at {w.discovered:,} "
            f"positions in {w.elapsed:.2f}s. "
            + ("Checked with the Cube class: it solves the cube." if ok
               else "WARNING: the check with the Cube class failed.")
            + extra + " Press Play to watch it.")
        self._update_buttons()

    # --------------------------------------------------------------- playback
    def _show_solution(self):
        t = self.sol_text
        t.configure(state="normal")
        t.delete("1.0", "end")
        if self.solution is None:
            t.insert("end", "Solve the cube and the shortest solution appears here.", "note")
        elif not self.solution:
            t.insert("end", "(already solved)", "note")
        else:
            for k, m in enumerate(self.solution):
                tag = "done" if k < self.sol_index else (
                    "current" if k == self.sol_index else "todo")
                t.insert("end", f" {m} ", tag)
                t.insert("end", " ")
            t.insert("end", f"   {self.sol_index}/{len(self.solution)} moves done", "note")
        t.configure(state="disabled")

    def step_forward(self):
        if not self.solution or self.animating or self.queue:
            return
        if self.sol_index >= len(self.solution):
            return
        k = self.sol_index
        m = self.solution[k]
        self.sol_index += 1
        self.queue.append((m, f"Move {k + 1}/{len(self.solution)}:  {m}", 1.0))
        self._show_solution()
        self._pump()

    def step_back(self):
        if not self.solution or self.animating or self.queue or self.sol_index == 0:
            return
        self.sol_index -= 1
        m = inverse_move(self.solution[self.sol_index])
        self.queue.append((m, f"Undo move {self.sol_index + 1}:  {m}", 1.0))
        self._show_solution()
        self._pump()

    def toggle_play(self):
        if not self.solution:
            return
        if self.playing:
            self.playing = False
            self._update_buttons()
            return
        if self.sol_index >= len(self.solution):      # replay from the start
            self.cube = self.solve_start
            self.sol_index = 0
            self._show_solution()
            self.redraw()
        self.playing = True
        self._update_buttons()
        if not self.animating:
            self._play_next()

    def _play_next(self):
        if not self.playing:
            return
        if self.sol_index >= len(self.solution):
            self.playing = False
            self._update_buttons()
            self.set_status(f"Solved in {len(self.solution)} moves, the fewest possible. "
                            "Press Play to watch again.")
            return
        self.step_forward()

    # --------------------------------------------------------------- benchmark
    def do_benchmark(self):
        if not self.solution or self.busy():
            return
        if len(self.solution) > 9:
            self.set_status(
                f"This position needs {len(self.solution)} moves. The original "
                "bfs_solve() would store millions of Cube objects (over 1 GB of "
                "memory, a minute or more), so the benchmark is limited to "
                "solutions of 9 moves or fewer. Try a shorter scramble.")
            return
        start = self.solve_start
        fast_time = self.worker.elapsed
        self.bench_btn.configure(state="disabled")
        self.set_status("Running the original bfs_solve() from cube2x2_bfs.py ...")
        result = {}

        def run():
            t0 = time.perf_counter()
            result["sol"] = bfs_solve(start, URF_MOVES)
            result["time"] = time.perf_counter() - t0

        th = threading.Thread(target=run, daemon=True)
        th.start()

        def wait():
            if th.is_alive():
                self.root.after(150, wait)
                return
            orig = result["sol"]
            t = result["time"]
            same = orig is not None and len(orig) == len(self.solution)
            speed = t / fast_time if fast_time > 0 else float("inf")
            self.set_status(
                f"Original bfs_solve(): {t:.2f}s, {len(orig)} moves "
                f"({' '.join(orig)}).  Optimized BFS: {fast_time:.2f}s, "
                f"{len(self.solution)} moves.  Speed-up: {speed:.1f}x. "
                + ("Both found the same length, as BFS guarantees."
                   if same else "Lengths differ - check this!"))
            self._update_buttons()
        wait()


def short(n):
    n = float(n)
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e4:
        return f"{n / 1e3:.0f}k"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return f"{n:.0f}"


# =============================================================================
# 5. Tests (python cube_simulator.py --test)
# =============================================================================
def run_tests():
    import random
    from cube2x2_bfs import ALL_MOVES

    # (a) sticker colors match an independent 3D sticker model for all 18 moves
    def model_solved():
        return {(POSITIONS[i], NORMALS[f]): f for i in range(8) for f in FACELETS[i]}

    def model_move(stickers, move):
        axis = NORMALS[move[0]]
        turns = {"": 1, "2": 2, "'": 3}[move[1:]]
        for _ in range(turns):
            new = {}
            for (p, n), col in stickers.items():
                if dot(p, axis) > 0:
                    p = tuple(round(x) for x in rotate(p, axis, -math.pi / 2))
                    n = tuple(round(x) for x in rotate(n, axis, -math.pi / 2))
                new[(p, n)] = col
            stickers = new
        return stickers

    rng = random.Random(0)
    for _ in range(200):
        seq = [rng.choice(ALL_MOVES) for _ in range(rng.randint(1, 25))]
        model = model_solved()
        for m in seq:
            model = model_move(model, m)
        colors = sticker_colors(Cube().apply_sequence(seq))
        mine = {(POSITIONS[i], NORMALS[f]): c for (i, f), c in colors.items()}
        assert mine == model, seq
    print("sticker colors match the 3D model")

    # (b) every face has 4 stickers of one color when solved, 24 net slots
    assert len(NET_SLOTS) == 24
    solved_colors = sticker_colors(Cube())
    assert all(c == f for (i, f), c in solved_colors.items())

    # (c) integer encoding round-trips
    for _ in range(500):
        c = Cube().apply_sequence(random_scramble(12, URF_MOVES, seed=rng.random()))
        assert decode(encode(c)) == c
    assert encode(Cube()) == 0
    print("integer encoding round-trips")

    tables = MoveTables()
    # (d) integer move tables agree with Cube.apply_move
    for _ in range(500):
        c = Cube().apply_sequence(random_scramble(10, URF_MOVES, seed=rng.random()))
        s = encode(c)
        for mi, m in enumerate(tables.moves):
            p, o = divmod(s, N_ORI)
            assert tables.perm[mi][p] * N_ORI + tables.ori[mi][o] == encode(c.apply_move(m))
    print("move tables agree with Cube.apply_move")

    # (e) optimized BFS gives the same length as the original bfs_solve
    for seed in range(15):
        scr = random_scramble(rng.randint(1, 7), URF_MOVES, seed=seed)
        c = Cube().apply_sequence(scr)
        w = BFSWorker(tables, encode(c))
        w.run()
        orig = bfs_solve(c, URF_MOVES)
        assert c.apply_sequence(w.solution).is_solved()
        assert len(w.solution) == len(orig), (scr, w.solution, orig)
    print("optimized BFS = original BFS on solution length")

    # (f) mapping the whole space reproduces the published table
    t0 = time.perf_counter()
    w = BFSWorker(tables, 0, goal=None)
    w.run()
    layers = [x for x in w.layers if x]
    assert layers == KNOWN_LAYERS, layers
    print(f"full state space: {sum(layers):,} positions, max depth "
          f"{len(layers) - 1}, matches published table ({time.perf_counter() - t0:.1f}s)")
    print("\nAll simulator tests passed.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_tests()
        sys.exit(0)
    if tk is None:
        print("tkinter is not installed. On Linux: sudo apt install python3-tk")
        sys.exit(1)
    root = tk.Tk()
    app = Simulator(root)
    root.mainloop()
