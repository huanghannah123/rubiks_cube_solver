"""
2x2x2 Rubik's Cube IDA* Simulator
=================================
An animated view of the IDA* solver in cube2x2_ida.py, for a science-fair demo.

Run:      python ida_simulator.py          (opens the window)
Tests:    python ida_simulator.py --test   (checks the animation matches your solver)

Only ONE other file is needed, in the same folder: your IDA* solver,
either  cube2x2_ida.py  (corner + orientation heuristic)
or      cube.py         (corner-permutation heuristic only)
Everything else (3D drawing, scrambles) is built into this file.

What you see while IDA* searches
--------------------------------
* The cube shows the exact position IDA* is checking. At slow speeds it
  physically turns into each new position and turns back when it backs up.
* The f = g + h gauge:  g = moves made so far (blue),
                        h = heuristic estimate of moves still needed (orange),
                        red line = the current bound.
  If f > bound the branch is cut off; otherwise IDA* goes one move deeper.
* The search tree fills in as positions are checked:
  each row is a depth, each block is a position,
  blue = explored, gray = cut off (f > bound), gold = the solution path.
  It restarts every time the bound goes up.
* The chart on the right shows how many positions each iteration checked.

The search here is your IDAStarSolver.search rewritten with an explicit
stack, so the window can pause after every node. It uses your heuristic,
your move pruning and your move order, and --test checks that it visits
exactly the same number of nodes and returns the same solution.
"""

import contextlib
import io
import math
import sys
import time

import random

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    tk = None

ida = None                       # cube2x2_ida, loaded at startup


# =============================================================================
# 0. Small helpers: moves, scrambles, colors, 3D drawing
# =============================================================================
FACES = ["U", "D", "R", "L", "F", "B"]
ALL_MOVES = [f + s for f in FACES for s in ("", "'", "2")]      # 18 moves

# Distances of all 3,674,160 positions (Jaap's Puzzle Page), for the BFS comparison
KNOWN_LAYERS = [1, 9, 54, 321, 1847, 9992, 50136, 227536,
                870072, 1887748, 623800, 2644]

BG = "#1f2329"          # cube canvas background
PANEL = "#f4f5f7"       # control panel background
ACCENT = "#2563eb"
MUTED = "#6b7280"

COLORS = {"U": "#FFFFFF", "D": "#FFD500", "F": "#009B48",
          "B": "#0046AD", "R": "#C41E3A", "L": "#FF5800"}

POSITIONS = [(1, 1, 1), (1, 1, -1), (-1, 1, -1), (-1, 1, 1),
             (1, -1, 1), (1, -1, -1), (-1, -1, -1), (-1, -1, 1)]

NORMALS = {"U": (0, 1, 0), "D": (0, -1, 0), "R": (1, 0, 0),
           "L": (-1, 0, 0), "F": (0, 0, 1), "B": (0, 0, -1)}


def inverse_move(move):
    if move.endswith("'"):
        return move[0]
    if move.endswith("2"):
        return move
    return move + "'"


def random_scramble(length, moves=ALL_MOVES, seed=None):
    """Random scramble that never turns the same face twice in a row."""
    rng = random.Random(seed)
    scramble = []
    while len(scramble) < length:
        m = rng.choice(moves)
        if scramble and scramble[-1][0] == m[0]:
            continue
        scramble.append(m)
    return scramble


def short(n):
    n = float(n)
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e4:
        return f"{n / 1e3:.0f}k"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return f"{n:.0f}"


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


def draw_cube_3d(canvas, colors, camera, cx, cy, scale, anim=None):
    """Draw the 8 small cubes. anim = (face, angle) turns that layer mid-move.
    colors[(position, face)] = which face's color the sticker there has."""
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
            outward = dot(POSITIONS[i], n) > 0
            for size, push, fill in ((0.5, 0.5, "#141414"), (0.43, 0.502, None)):
                if fill is None:                  # sticker layer
                    if not outward:
                        continue
                    fill = COLORS[colors[(i, face)]]
                if camera(world(n))[2] <= 1e-6:   # facing away: hidden
                    continue
                pts = [tuple(center[k] + push * n[k] + size * (a * u[k] + b * w[k])
                             for k in range(3)) for a, b in FACE_CORNERS]
                quads.append([camera(world(p)) for p in pts] + [fill])
        items.append((camera(world(center))[2], quads))

    items.sort(key=lambda it: it[0])              # far cubes first
    for _, quads in items:
        for q in quads:
            flat = []
            for (x, y, _) in q[:-1]:
                flat += [cx + scale * x, cy - scale * y]
            canvas.create_polygon(flat, fill=q[-1], outline="#141414", width=1.5)


IDA_MODULES = ["cube2x2_ida", "cube"]    # first one found in the folder is used
ida_name = None


def load_ida(quiet=True):
    """Import your IDA* file (it builds its heuristic tables on import).

    Works with cube2x2_ida.py (corner + orientation heuristic) or cube.py
    (corner-permutation heuristic only), whichever is in the folder.
    """
    global ida, ida_name
    if ida is not None:
        return ida
    import importlib
    import importlib.util
    for name in IDA_MODULES:
        if importlib.util.find_spec(name) is not None:
            break
    else:
        msg = ("Could not find your IDA* solver.\n"
               "Put cube2x2_ida.py (or cube.py) in the same folder as ida_simulator.py.")
        print(msg)
        if tk is not None:
            try:
                from tkinter import messagebox
                r = tk.Tk()
                r.withdraw()
                messagebox.showerror("Missing file", msg)
            except Exception:
                pass
        sys.exit(1)
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            module = importlib.import_module(name)
    else:
        module = importlib.import_module(name)
    ida, ida_name = module, name
    return ida


# =============================================================================
# 1. Connecting the 24-sticker state to the 3D drawing
# =============================================================================
POS_INDEX = {p: i for i, p in enumerate(POSITIONS)}


def colors_from_state(state):
    """{(position, face): home face} for draw_cube_3d.

    state[slot] = id of the sticker sitting in that slot, and
    STICKERS[id][1] is the face that sticker belongs on.
    """
    out = {}
    for slot, (corner, face, position, normal) in enumerate(ida.STICKERS):
        out[(POS_INDEX[position], face)] = ida.STICKERS[state[slot]][1]
    return out


def do_move(state, move):
    return ida.apply_move(state, ida.ACTION_LOOKUP[move])


# =============================================================================
# 2. IDA* as a step-by-step generator
# =============================================================================
# Same logic as IDAStarSolver.search/solve, but instead of recursion it
# keeps its own stack and "yields" an event after every node, so the GUI
# can draw each step.
#
# Events:
#   ("iteration", bound, number)
#   ("node", path, state, g, h, f, bound, kind)   kind = expand / pruned / goal
#   ("iteration_end", bound, nodes_this_iteration, next_bound)
#   ("solved", path, total_nodes)
#   ("failed", total_nodes)

def ida_star_events(start, max_depth=14):
    heuristic, actions = ida.heuristic, ida.ACTIONS
    should_prune, apply_move, SOLVED = ida.should_prune, ida.apply_move, ida.SOLVED
    INF = float("inf")

    bound = heuristic(start)
    total = 0
    number = 0
    while bound <= max_depth:
        number += 1
        yield ("iteration", bound, number)
        iter_nodes = 0
        path = []
        stack = []                      # frames: [state, g, last_face, next_action, minimum]
        pending = (start, 0, None)      # a node waiting to be checked
        result = None                   # value a child hands back to its parent
        next_bound = INF

        while True:
            if pending is not None:
                state, g, last_face = pending
                pending = None
                total += 1
                iter_nodes += 1
                h = heuristic(state)
                f = g + h
                if f > bound:                                   # cut off
                    yield ("node", tuple(path), state, g, h, f, bound, "pruned")
                    result = f
                elif state == SOLVED:                           # goal
                    yield ("node", tuple(path), state, g, h, f, bound, "goal")
                    yield ("iteration_end", bound, iter_nodes, bound)
                    yield ("solved", list(path), total)
                    return
                else:                                           # go deeper
                    yield ("node", tuple(path), state, g, h, f, bound, "expand")
                    stack.append([state, g, last_face, 0, INF])

            if result is not None:          # a child finished: report to its parent
                if not stack:
                    next_bound = result
                    break
                path.pop()
                if result < stack[-1][4]:
                    stack[-1][4] = result
                result = None

            frame = stack[-1]
            while frame[3] < len(actions):  # next allowed move of this node
                name, permutation, face = actions[frame[3]]
                frame[3] += 1
                if should_prune(frame[2], face):
                    continue
                path.append(name)
                pending = (apply_move(frame[0], permutation), frame[1] + 1, face)
                break
            else:                           # all children tried
                stack.pop()
                result = frame[4]
                if not stack:
                    next_bound = result
                    break

        yield ("iteration_end", bound, iter_nodes, next_bound)
        bound = next_bound
    yield ("failed", total)


# =============================================================================
# 3. The window
# =============================================================================
SPEEDS = [1, 2, 5, 20, 100, 1000, None]          # nodes per second (None = max)
SPEED_NAMES = ["1 node/s", "2 nodes/s", "5 nodes/s", "20 nodes/s",
               "100 nodes/s", "1,000 nodes/s", "full speed"]
BLUE, ORANGE, RED, GREEN, GOLD = "#3b82f6", "#f59e0b", "#ef4444", "#22c55e", "#fbbf24"


class IDASimulator:
    def __init__(self, root):
        self.root = root
        root.title("2x2x2 Rubik's Cube - IDA* Simulator")
        root.configure(bg=PANEL)
        root.minsize(1150, 720)
        root.geometry("1230x760")

        self.state = ida.SOLVED          # the real cube
        self.view_state = ida.SOLVED     # what is drawn (a search node while searching)
        self.view_path = ()
        self.scramble = []
        self.history = []

        self.gen = None                  # running search
        self.searching = False
        self.paused = False
        self.cur = None                  # last node event
        self.message = ("Scramble the cube, then press Watch IDA* search.", None)
        self.iterations = []             # [bound, nodes]
        self.total_nodes = 0
        self.search_start = None
        self.search_time = 0.0

        self.solution = None
        self.sol_index = 0
        self.playing = False
        self.animating = False
        self.banner = ""

        # allowed moves after each last face (for laying out the tree)
        self.allowed = {}
        for last in [None] + FACES:
            names = [n for n, p, f in ida.ACTIONS if not ida.should_prune(last, f)]
            self.allowed[last] = {n: i for i, n in enumerate(names)}
            self.allowed[(last, "n")] = len(names)

        self._build_ui()
        self.render()
        self._draw_chart()

    # ------------------------------------------------------------------ UI
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
        style.configure("Move.TButton", font=("Courier", 12, "bold"),
                        padding=(2, 2), width=3)

        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(left, width=700, height=330, bg=BG,
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.render())
        self.tree = tk.Canvas(left, width=700, height=190, bg="white",
                              highlightthickness=1, highlightbackground="#d1d5db")
        self.tree.pack(fill="x", pady=(8, 0))
        self._clear_tree(None)

        play = ttk.Frame(left, padding=(0, 8, 0, 0))
        play.pack(fill="x")
        row = ttk.Frame(play)
        row.pack(fill="x")
        ttk.Label(row, text="Solution", style="Head.TLabel").pack(side="left")
        self.back_btn = ttk.Button(row, text="< Back", command=self.step_back)
        self.back_btn.pack(side="left", padx=(14, 0))
        self.play_btn = ttk.Button(row, text="Play", width=8, command=self.toggle_play)
        self.play_btn.pack(side="left", padx=6)
        self.fwd_btn = ttk.Button(row, text="Step >", command=self.step_forward)
        self.fwd_btn.pack(side="left")
        self.sol_text = tk.Text(row, height=1, width=36, font=("Courier", 12, "bold"),
                                relief="flat", bg="white", wrap="none",
                                highlightthickness=1, highlightbackground="#d1d5db",
                                padx=6, pady=3)
        for tag, opts in {"done": {"foreground": "#9ca3af"},
                          "current": {"foreground": "white", "background": ACCENT},
                          "todo": {"foreground": "#111827"},
                          "note": {"foreground": "#9ca3af", "font": ("Helvetica", 11)}
                          }.items():
            self.sol_text.tag_configure(tag, **opts)
        self.sol_text.pack(side="left", fill="x", expand=True, padx=(10, 0))
        self.status = ttk.Label(play, text="", style="Muted.TLabel",
                                wraplength=740, justify="left")
        self.status.pack(anchor="w", pady=(6, 0))
        play.bind("<Configure>",
                  lambda e: self.status.configure(wraplength=max(300, e.width - 10)))

        side = ttk.Frame(main, padding=(14, 0, 0, 0))
        side.pack(side="right", fill="y")
        ttk.Label(side, text="2x2x2 IDA* Solver", style="Title.TLabel").pack(anchor="w")
        ttk.Label(side, text="Depth-first search with a rising bound on f = g + h"
                             f"\nUsing your solver: {ida_name}.py",
                  style="Muted.TLabel", justify="left").pack(anchor="w")

        box = self._section(side, "1. Scramble")
        row = ttk.Frame(box)
        row.pack(fill="x")
        self.scramble_btn = ttk.Button(row, text="Scramble", command=self.do_scramble)
        self.scramble_btn.pack(side="left")
        self.reset_btn = ttk.Button(row, text="Reset", command=self.do_reset)
        self.reset_btn.pack(side="left", padx=(6, 0))
        ttk.Label(row, text="  length").pack(side="left")
        self.len_var = tk.IntVar(value=5)
        self.len_label = ttk.Label(row, text="5", width=3)
        scale = ttk.Scale(row, from_=1, to=14, orient="horizontal", length=60,
                          command=self._on_len)
        scale.pack(side="left", padx=4)
        self.len_label.pack(side="left")
        scale.set(5)

        box = self._section(side, "2. Turn it yourself (all 18 moves)")
        grid = ttk.Frame(box)
        grid.pack(anchor="w")
        self.move_buttons = []
        for r, suffix in enumerate(["", "'", "2"]):
            for c, face in enumerate(FACES):
                m = face + suffix
                b = ttk.Button(grid, text=m, style="Move.TButton",
                               command=lambda m=m: self.user_move(m))
                b.grid(row=r, column=c, padx=2, pady=2)
                self.move_buttons.append(b)

        box = self._section(side, "3. Search")
        row = ttk.Frame(box)
        row.pack(fill="x")
        self.solve_btn = ttk.Button(row, text="Watch IDA* search", style="Big.TButton",
                                    command=self.start_search)
        self.solve_btn.pack(side="left")
        self.pause_btn = ttk.Button(row, text="Pause", width=7, command=self.toggle_pause)
        self.pause_btn.pack(side="left", padx=6)
        self.stop_btn = ttk.Button(row, text="Stop", command=self.stop_search)
        self.stop_btn.pack(side="left")

        row = ttk.Frame(box)
        row.pack(fill="x", pady=(8, 0))
        ttk.Label(row, text="Speed").pack(side="left")
        self.speed_idx = 1
        self.speed_label = ttk.Label(row, text=SPEED_NAMES[1], width=13)
        sp = ttk.Scale(row, from_=0, to=len(SPEEDS) - 1, orient="horizontal",
                       length=170, command=self._on_speed)
        sp.pack(side="left", padx=6)
        self.speed_label.pack(side="left")
        sp.set(1)

        stats = ttk.Frame(box)
        stats.pack(fill="x", pady=(10, 0))
        self.stat_vars = {}
        for col, (key, label) in enumerate([("bound", "Bound"), ("iter", "This round"),
                                            ("total", "Total nodes"), ("time", "CPU time")]):
            cell = ttk.Frame(stats)
            cell.grid(row=0, column=col, sticky="w", padx=(0, 12))
            v = tk.StringVar(value="-")
            self.stat_vars[key] = v
            ttk.Label(cell, textvariable=v, style="Stat.TLabel").pack(anchor="w")
            ttk.Label(cell, text=label, style="Muted.TLabel").pack(anchor="w")

        self.chart = tk.Canvas(box, width=350, height=140, bg="white",
                               highlightthickness=1, highlightbackground="#d1d5db")
        self.chart.pack(pady=(10, 0))
        self.compare = ttk.Label(box, text="", style="Muted.TLabel",
                                 wraplength=320, justify="left")
        self.compare.pack(anchor="w", pady=(6, 0))

        for key in ("<space>",):
            self.root.bind(key, lambda e: self.toggle_pause() if self.searching
                           else self.toggle_play())
        self.root.bind("<Right>", lambda e: self.step_forward())
        self.root.bind("<Left>", lambda e: self.step_back())
        self._show_solution()
        self._update_buttons()
        self.set_status("Ready. Scramble the cube, then press Watch IDA* search. "
                        "Short scrambles (4-6 moves) are best for watching slowly.")

    def _section(self, parent, title):
        ttk.Label(parent, text=title, style="Head.TLabel").pack(anchor="w", pady=(10, 2))
        f = ttk.Frame(parent)
        f.pack(fill="x")
        return f

    def _on_len(self, value):
        n = int(round(float(value)))
        self.len_var.set(n)
        self.len_label.configure(text=str(n))

    def _on_speed(self, value):
        self.speed_idx = int(round(float(value)))
        self.speed_label.configure(text=SPEED_NAMES[self.speed_idx])

    def set_status(self, text):
        self.status.configure(text=text)

    def _update_buttons(self):
        busy = self.searching or self.animating or self.playing
        on = lambda ok: "normal" if ok else "disabled"
        self.scramble_btn.configure(state=on(not busy))
        self.reset_btn.configure(state=on(not busy))
        for b in self.move_buttons:
            b.configure(state=on(not busy))
        self.solve_btn.configure(state=on(not busy))
        self.pause_btn.configure(state=on(self.searching),
                                 text="Resume" if self.paused else "Pause")
        self.stop_btn.configure(state=on(self.searching))
        has = bool(self.solution)
        idle = not self.searching and not self.animating and not self.playing
        self.back_btn.configure(state=on(has and idle and self.sol_index > 0))
        self.fwd_btn.configure(state=on(has and idle and self.sol_index < len(self.solution)))
        self.play_btn.configure(state=on(has and not self.searching and not
                                         (self.animating and not self.playing)),
                                text="Pause" if self.playing else "Play")

    # ------------------------------------------------------------- drawing
    def render(self, state=None, anim=None):
        c = self.canvas
        c.delete("all")
        w = max(c.winfo_width(), 700)
        h = max(c.winfo_height(), 300)
        state = self.view_state if state is None else state
        colors = colors_from_state(state)

        title = "Position IDA* is checking" if self.searching else "The cube"
        c.create_text(16, 18, anchor="w", fill="#e5e7eb",
                      font=("Helvetica", 13, "bold"), text=title)
        size = min(88, h / 3.6)
        draw_cube_3d(c, colors, FRONT_CAMERA, w * 0.17, 40 + size * 1.5, size, anim)
        draw_cube_3d(c, colors, BACK_CAMERA, w * 0.42, 34 + size * 1.0,
                        size * 0.55, anim)
        c.create_text(w * 0.42, 40 + size * 1.95, fill="#9ca3af",
                      font=("Helvetica", 9), text="back view")
        if self.banner:
            c.create_text(w * 0.27, h - 26, fill=GOLD,
                          font=("Helvetica", 18, "bold"), text=self.banner)
        elif not self.searching and ida.SOLVED == state:
            c.create_text(w * 0.27, h - 26, fill="#34d399",
                          font=("Helvetica", 18, "bold"), text="SOLVED")

        self._draw_info(c, w * 0.56, w - 16, h)

    def _draw_info(self, c, x0, x1, h):
        width = x1 - x0
        text_color = "#e5e7eb"
        ev = self.cur
        if ev is None:
            msg, color = self.message
            c.create_text(x0, 20, anchor="nw", fill=text_color, width=width,
                          font=("Helvetica", 14, "bold"), text="How IDA* works")
            c.create_text(x0, 52, anchor="nw", fill=color or "#d1d5db", width=width,
                          font=("Helvetica", 11), text=msg)
            return
        _, path, state, g, hval, f, bound, kind = ev
        it = len(self.iterations)
        c.create_text(x0, 16, anchor="nw", fill=text_color,
                      font=("Helvetica", 14, "bold"),
                      text=f"Iteration {it}:  bound = {bound}")
        c.create_text(x0, 42, anchor="nw", fill="#9ca3af", font=("Helvetica", 11),
                      text=f"Node #{self.total_nodes:,}  ·  depth g = {g}")

        # gauge: g (blue) + h (orange) against the bound (red line)
        top = 92
        scale_max = max(12, bound + 2, f + 1)
        unit = width / scale_max
        c.create_rectangle(x0, top, x1, top + 24, fill="#374151", outline="")
        c.create_rectangle(x0, top, x0 + g * unit, top + 24, fill=BLUE, outline="")
        c.create_rectangle(x0 + g * unit, top, x0 + f * unit, top + 24,
                           fill=ORANGE, outline="")
        if g:
            c.create_text(x0 + g * unit / 2, top + 12, text=f"g={g}", fill="white",
                          font=("Helvetica", 9, "bold"))
        if hval:
            c.create_text(x0 + (g + hval / 2) * unit, top + 12, text=f"h={hval}",
                          fill="#111827", font=("Helvetica", 9, "bold"))
        bx = x0 + bound * unit
        c.create_line(bx, top - 8, bx, top + 32, fill=RED, width=3)
        c.create_text(bx, top - 10, text=f"bound {bound}", fill=RED, anchor="s",
                      font=("Helvetica", 9, "bold"))
        for k in range(scale_max + 1):
            c.create_text(x0 + k * unit, top + 38, text=str(k), fill="#6b7280",
                          font=("Helvetica", 8))

        sign = ">" if kind == "pruned" else "<="
        color = RED if kind == "pruned" else (GOLD if kind == "goal" else GREEN)
        c.create_text(x0, top + 58, anchor="nw", fill=color,
                      font=("Helvetica", 17, "bold"),
                      text=f"f = {g} + {hval} = {f}  {sign}  {bound}")
        if kind == "pruned":
            msg = (f"CUT OFF. Even in the best case this path needs at least {f} "
                   f"moves, more than the bound of {bound}. IDA* backs up and tries "
                   "the next move.")
        elif kind == "goal":
            msg = (f"SOLVED in {g} moves! Every smaller bound already failed, "
                   "so no shorter solution exists.")
        else:
            msg = (f"Keep going. A solution within {bound} moves is still possible "
                   "from here, so IDA* tries each move one level deeper.")
        c.create_text(x0, top + 90, anchor="nw", fill="#d1d5db", width=width,
                      font=("Helvetica", 10), text=msg)

        c.create_text(x0, top + 150, anchor="nw", fill="#9ca3af",
                      font=("Helvetica", 10, "bold"), text="Moves tried on this path:")
        x, y = x0, top + 170
        if not path:
            c.create_text(x, y, anchor="nw", fill="#6b7280", font=("Helvetica", 10),
                          text="(none yet - this is the scrambled cube)")
        for k, m in enumerate(path):
            last = k == len(path) - 1
            bw = 34
            if x + bw > x1:
                x, y = x0, y + 28
            c.create_rectangle(x, y, x + bw - 4, y + 22,
                               fill=(color if last else "#374151"), outline="")
            c.create_text(x + (bw - 4) / 2, y + 11, text=m,
                          fill=("#111827" if last else "white"),
                          font=("Courier", 11, "bold"))
            x += bw

    # --------------------------------------------------------- search tree
    def _clear_tree(self, bound):
        t = self.tree
        t.delete("all")
        W = max(t.winfo_width(), 760)
        H = max(t.winfo_height(), 190)
        self.tree_geom = None
        if bound is None:
            t.create_text(W / 2, H / 2, fill=MUTED, font=("Helvetica", 11),
                          text="The search tree appears here while IDA* runs.")
            return
        rows = bound + 1
        left, top = 34, 22
        row_h = min(24, (H - top - 6) / rows)
        self.tree_geom = (left, top, W - left - 10, row_h)
        t.create_text(8, 10, anchor="w", fill="#374151", font=("Helvetica", 10, "bold"),
                      text=f"Search tree for bound = {bound}")
        t.create_text(W - 10, 10, anchor="e", fill=MUTED, font=("Helvetica", 9),
                      text="each row = one more move   blue = explored   "
                           "gray = cut off (f > bound)   gold = solution")
        for d in range(rows):
            t.create_text(left - 6, top + d * row_h + row_h / 2, anchor="e",
                          fill=MUTED, font=("Helvetica", 8), text=f"{d}")

    def _tree_span(self, path, min_width):
        """x-interval of a node: each node splits its parent's width
        equally between its allowed moves."""
        left, top, W, row_h = self.tree_geom
        x, width, last = 0.0, float(W), None
        for m in path:
            n = self.allowed[(last, "n")]
            width /= n
            x += self.allowed[last][m] * width
            last = m[0]
            if width < min_width:
                return None
        return left + x, width

    def _tree_add(self, path, kind):
        if self.tree_geom is None:
            return
        span = self._tree_span(path, 1.2)
        if span is None:
            return
        left, top, W, row_h = self.tree_geom
        x, width = span
        y = top + len(path) * row_h
        fill = {"expand": BLUE, "pruned": "#d1d5db", "goal": GOLD}[kind]
        gap = 1 if width > 3 else 0
        self.tree.create_rectangle(x, y, x + width - gap, y + row_h - 2,
                                   fill=fill, outline="")

    def _tree_mark_path(self, path, color, tag, width=2):
        if self.tree_geom is None:
            return
        self.tree.delete(tag)
        left, top, W, row_h = self.tree_geom
        for d in range(len(path) + 1):
            x, w = self._tree_span(path[:d], 0) or (left, W)
            w = max(w, 3)
            y = top + d * row_h
            self.tree.create_rectangle(x - 1, y - 1, x + w, y + row_h - 1,
                                       outline=color, width=width, tags=tag)

    # --------------------------------------------------------------- chart
    def _draw_chart(self):
        ch = self.chart
        ch.delete("all")
        W, H = 350, 140
        left, bottom, top = 12, H - 26, 34
        ch.create_text(W / 2, 10, text="Positions checked in each iteration",
                       font=("Helvetica", 9, "bold"), fill="#374151")
        ch.create_text(W / 2, 23, text="(the bound goes up by one each time it fails)",
                       font=("Helvetica", 8), fill=MUTED)
        data = self.iterations
        if not data:
            ch.create_text(W / 2, H / 2 + 8, text="(no search yet)",
                           font=("Helvetica", 9), fill=MUTED)
            return
        n = max(len(data), 6)
        bw = (W - 2 * left) / n
        biggest = max(max(c for b, c in data), 10)
        log_max = math.log10(biggest + 1) * 1.05
        for i, (b, count) in enumerate(data):
            x = left + i * bw
            hgt = (bottom - top) * math.log10(count + 1) / log_max
            last = i == len(data) - 1
            color = GOLD if (last and self.solution is not None
                             and not self.searching) else ACCENT
            ch.create_rectangle(x + 4, bottom - hgt, x + bw - 4, bottom,
                                fill=color, outline="")
            ch.create_text(x + bw / 2, bottom - hgt - 7, text=short(count),
                           font=("Helvetica", 8), fill="#374151")
            ch.create_text(x + bw / 2, H - 13, text=f"bound {b}",
                           font=("Helvetica", 8), fill=MUTED)
        ch.create_line(left, bottom, W - left, bottom, fill="#9ca3af")

    def _update_stats(self):
        if self.iterations:
            b, n = self.iterations[-1]
            self.stat_vars["bound"].set(str(b))
            self.stat_vars["iter"].set(f"{n:,}")
        self.stat_vars["total"].set(f"{self.total_nodes:,}")
        self.stat_vars["time"].set(f"{self.search_time:.2f}s")

    # ----------------------------------------------------------- animation
    def animate(self, state, moves, seconds, labels, done):
        """Turn the drawn cube through `moves`, then call done(final_state)."""
        if not moves:
            done(state)
            return
        self.animating = True
        move, label = moves[0], labels[0]
        face, total = move[0], move_angle(move)
        frames = max(3, int(seconds * 50 * (1.4 if move.endswith("2") else 1)))
        self.banner = label

        def step(i):
            t = i / frames
            self.render(state, anim=(face, total * t * t * (3 - 2 * t)))
            if i < frames:
                self.root.after(20, step, i + 1)
            else:
                new = do_move(state, move)
                if len(moves) > 1:
                    self.animate(new, moves[1:], seconds, labels[1:], done)
                else:
                    self.animating = False
                    self.banner = ""
                    done(new)
        step(1)

    def user_move(self, move):
        if self.searching or self.animating or self.playing:
            return
        self._forget_solution()

        def done(new):
            self.state = self.view_state = new
            self.history.append(move)
            self.render()
            self._update_buttons()
        self.animate(self.state, [move], 0.3, [move], done)
        self._update_buttons()

    def do_scramble(self):
        if self.searching or self.animating or self.playing:
            return
        self._forget_solution()
        n = self.len_var.get()
        self.scramble = random_scramble(n, ALL_MOVES)
        labels = [f"Scramble {k + 1}/{n}:  {m}" for k, m in enumerate(self.scramble)]
        self.set_status(f"Scramble ({n} moves): {' '.join(self.scramble)}")

        def done(new):
            self.state = self.view_state = new
            self.history += self.scramble
            self.cur = None
            self.message = ("Press Watch IDA* search. IDA* starts with "
                            f"bound = h(start) = {ida.heuristic(new)}, its estimate "
                            "of the fewest moves needed, and searches every path "
                            "whose f = g + h stays within the bound.", None)
            self.render()
            self._update_buttons()
        self.animate(self.state, self.scramble, 0.16, labels, done)
        self._update_buttons()

    def do_reset(self):
        if self.searching or self.animating or self.playing:
            return
        self._forget_solution()
        self.state = self.view_state = ida.SOLVED
        self.history, self.scramble = [], []
        self.cur = None
        self.iterations = []
        self.total_nodes = 0
        self._draw_chart()
        self._clear_tree(None)
        self.message = ("Scramble the cube, then press Watch IDA* search.", None)
        self.set_status("Reset to the solved cube.")
        self.render()

    def _forget_solution(self):
        self.solution = None
        self.sol_index = 0
        self.compare.configure(text="")
        self._show_solution()
        self._update_buttons()

    # -------------------------------------------------------------- search
    def start_search(self):
        if self.searching or self.animating or self.playing:
            return
        if self.state == ida.SOLVED:
            self.set_status("The cube is already solved. Scramble it first!")
            return
        self._forget_solution()
        self.search_start_state = self.state
        self.gen = ida_star_events(self.state)
        self.searching = True
        self.paused = False
        self.view_state, self.view_path = self.state, ()
        self.cur = None
        self.iterations = []
        self.total_nodes = 0
        self.search_time = 0.0
        self._draw_chart()
        note = ""
        if ida_name == "cube" and len(self.scramble) > 7:
            note = (" Note: cube.py only uses the corner-permutation heuristic, so "
                    "hard positions can take several minutes even at full speed. "
                    "cube2x2_ida.py (corners + orientation) solves them in seconds.")
        self.set_status("IDA* runs a depth-first search, but never follows a path "
                        "whose f = g + h is bigger than the bound. If no solution "
                        "fits, it raises the bound and starts over." + note)
        self._update_buttons()
        self.root.after(100, self._search_step)

    def toggle_pause(self):
        if not self.searching:
            return
        self.paused = not self.paused
        self._update_buttons()
        if not self.paused and not self.animating:
            self._search_step()

    def stop_search(self):
        if not self.searching:
            return
        self.searching = False
        self.paused = False
        self.gen = None
        self.view_state, self.view_path = self.state, ()
        self.cur = None
        self.message = ("Search stopped.", None)
        self.set_status("Search stopped.")
        self._update_buttons()
        self.render()

    def _search_step(self):
        if not self.searching or self.paused or self.animating:
            return
        rate = SPEEDS[self.speed_idx]
        if rate is None:                    # full speed: many nodes per frame
            t0 = time.perf_counter()
            deadline = t0 + 0.05
            while self.searching and time.perf_counter() < deadline:
                ev = next(self.gen)
                self._handle(ev, slow=False)
            self.search_time += time.perf_counter() - t0
            if self.searching:
                if self.cur:
                    self.view_state, self.view_path = self.cur[2], self.cur[1]
                    self._tree_mark_path(self.cur[1], "#111827", "cur")
                self._update_stats()
                self._draw_chart()
                self.render()
                self.root.after(1, self._search_step)
            return
        t0 = time.perf_counter()
        ev = next(self.gen)
        self.search_time += time.perf_counter() - t0
        delay = self._handle(ev, slow=True)
        if self.searching and delay is not None:
            self.root.after(delay, self._search_step)

    def _handle(self, ev, slow):
        """Process one search event. Returns a delay in ms (slow mode) or
        None if an animation will schedule the next step itself."""
        kind = ev[0]
        rate = SPEEDS[self.speed_idx] or 1000

        if kind == "iteration":
            _, bound, number = ev
            self.iterations.append([bound, 0])
            self._clear_tree(bound)
            if slow:
                self.cur = None
                self.view_state, self.view_path = self.search_start_state, ()
                if number == 1:
                    text = (f"Iteration 1. The heuristic says the cube needs at least "
                            f"{bound} moves, so the first bound is {bound}.")
                else:
                    text = (f"No solution fits within {bound - 1} moves. The bound "
                            f"rises to {bound} (the smallest f that went over), and "
                            "the search starts again from the scrambled cube.")
                self.message = (text, GOLD)
                self.banner = f"Bound = {bound}"
                self._update_stats()
                self._draw_chart()
                self.render()
                self.banner = ""
                return int(min(2500, 1200 + 1500 / rate))
            return 0

        if kind == "node":
            _, path, state, g, h, f, bound, nkind = ev
            self.total_nodes += 1
            self.iterations[-1][1] += 1
            self._tree_add(path, nkind)
            self.cur = ev
            if not slow:
                return 0
            k = 0
            old = self.view_path
            while k < min(len(old), len(path)) and old[k] == path[k]:
                k += 1
            undo = [inverse_move(m) for m in reversed(old[k:])]
            forward = list(path[k:])
            moves = undo + forward
            labels = ([f"Back up: undo {m}" for m in reversed(old[k:])]
                      + [f"Try {m}" for m in forward])
            self._tree_mark_path(path, "#111827", "cur")
            self._update_stats()
            self._draw_chart()
            dwell = int(1000 / rate)
            if rate <= 5 and 0 < len(moves) <= 4:
                seconds = min(0.4, 0.6 / rate)

                def arrived(new, path=path, state=state):
                    self.view_state, self.view_path = state, path
                    self.render()
                    self.root.after(max(60, dwell - int(seconds * 1000 * len(moves))),
                                    self._search_step)
                self.animate(self.view_state, moves, seconds, labels, arrived)
                return None
            self.view_state, self.view_path = state, path
            self.render()
            return max(1, dwell)

        if kind == "iteration_end":
            return 0 if not slow else int(min(800, 400 + 800 / rate))

        if kind == "solved":
            _, path, total = ev
            self._finish(path)
            return None

        if kind == "failed":
            self.searching = False
            self.set_status("No solution found within the depth limit.")
            self._update_buttons()
            return None
        return 0

    def _finish(self, path):
        self.searching = False
        self.gen = None
        self.solution = list(path)
        self.sol_index = 0
        goal = self.cur[2]
        self.view_state, self.view_path = goal, tuple(path)
        self._tree_mark_path(tuple(path), GOLD, "cur", width=3)
        ok = do_moves(self.search_start_state, path) == ida.SOLVED
        self._update_stats()
        self._draw_chart()
        self._show_solution()
        self.render()
        d = len(path)
        lo = sum(KNOWN_LAYERS[:d])
        hi = sum(KNOWN_LAYERS[:d + 1])
        self.compare.configure(
            text=(f"BFS would explore {lo:,} to {hi:,} positions to find a solution "
                  f"of {d} moves. IDA* checked {self.total_nodes:,}."))
        self.set_status(
            f"Found an optimal solution of {d} moves: {' '.join(path)}. "
            f"{len(self.iterations)} iteration(s), {self.total_nodes:,} nodes. "
            + ("Verified: it solves the scrambled cube. " if ok else "CHECK FAILED! ")
            + "Press Play to watch it.")
        self._update_buttons()

    # ------------------------------------------------------------ playback
    def _show_solution(self):
        t = self.sol_text
        t.configure(state="normal")
        t.delete("1.0", "end")
        if not self.solution:
            t.insert("end", "The shortest solution appears here after the search.",
                     "note")
        else:
            for k, m in enumerate(self.solution):
                tag = "done" if k < self.sol_index else (
                    "current" if k == self.sol_index else "todo")
                t.insert("end", f"{m}", tag)
                t.insert("end", " ")
            t.insert("end", f" {self.sol_index}/{len(self.solution)}", "note")
        t.configure(state="disabled")

    def _start_playback_view(self):
        """Before the first playback move, show the scrambled cube again."""
        if self.sol_index == 0 and self.view_state != self.search_start_state:
            self.state = self.view_state = self.search_start_state
            self.view_path = ()
            self.render()

    def step_forward(self, then=None):
        if not self.solution or self.animating or self.sol_index >= len(self.solution):
            return
        self._start_playback_view()
        k = self.sol_index
        m = self.solution[k]

        def done(new):
            self.state = self.view_state = new
            self.sol_index = k + 1
            self._show_solution()
            self.render()
            self._update_buttons()
            if then:
                then()
        self.animate(self.state, [m], 0.45, [f"Move {k + 1}/{len(self.solution)}:  {m}"],
                     done)
        self._update_buttons()

    def step_back(self):
        if not self.solution or self.animating or self.sol_index == 0:
            return
        k = self.sol_index - 1
        m = inverse_move(self.solution[k])

        def done(new):
            self.state = self.view_state = new
            self.sol_index = k
            self._show_solution()
            self.render()
            self._update_buttons()
        self.animate(self.state, [m], 0.45, [f"Undo move {k + 1}:  {m}"], done)
        self._update_buttons()

    def toggle_play(self):
        if not self.solution or self.searching:
            return
        if self.playing:
            self.playing = False
            self._update_buttons()
            return
        if self.sol_index >= len(self.solution):
            self.sol_index = 0
            self.view_state = None            # force the scrambled view
            self._start_playback_view()
            self._show_solution()
        self.playing = True
        self._update_buttons()
        self._play_next()

    def _play_next(self):
        if not self.playing:
            return
        if self.sol_index >= len(self.solution):
            self.playing = False
            self.set_status(f"Solved in {len(self.solution)} moves, the fewest possible.")
            self._update_buttons()
            return
        self.step_forward(then=lambda: self.root.after(300, self._play_next))


def do_moves(state, moves):
    for m in moves:
        state = do_move(state, m)
    return state


# =============================================================================
# 4. Tests (python ida_simulator.py --test)
# =============================================================================
def run_tests():
    from collections import deque
    load_ida()
    rng = random.Random(3)

    # (a) the 24-sticker state draws correctly: compare with an independent
    #     model that physically rotates each sticker in 3D
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

    for _ in range(300):
        seq = [rng.choice(ALL_MOVES) for _ in range(rng.randint(1, 20))]
        model = {(p, n): f for p in POSITIONS for f, n in NORMALS.items() if dot(p, n) > 0}
        for m in seq:
            model = model_move(model, m)
        drawn = {(POSITIONS[i], NORMALS[f]): c
                 for (i, f), c in colors_from_state(do_moves(ida.SOLVED, seq)).items()}
        assert drawn == model, seq
    print("sticker colors match a physical 3D model for all 18 moves")

    # (b) the animated search = IDAStarSolver: same nodes, same solution
    for seed in range(12):
        scr = random_scramble(rng.randint(3, 14), ALL_MOVES, seed=seed)
        start = do_moves(ida.SOLVED, scr)
        solver = ida.IDAStarSolver()
        with contextlib.redirect_stdout(io.StringIO()):
            expected = solver.solve(start)
        events = list(ida_star_events(start))
        solved = [e for e in events if e[0] == "solved"][0]
        nodes = sum(1 for e in events if e[0] == "node")
        assert solved[1] == expected, (scr, solved[1], expected)
        assert solved[2] == solver.nodes == nodes, (solved[2], solver.nodes, nodes)
        assert do_moves(start, expected) == ida.SOLVED
    print("step-by-step IDA* = IDAStarSolver (same nodes, same solution)")

    # (c) solutions are optimal: a plain BFS finds no shorter one
    for seed in range(10):
        scr = random_scramble(rng.randint(2, 5), ALL_MOVES, seed=100 + seed)
        start = do_moves(ida.SOLVED, scr)
        sol = [e for e in ida_star_events(start) if e[0] == "solved"][0][1]
        dist = {start: 0}
        queue = deque([start])
        while queue:
            s = queue.popleft()
            if s == ida.SOLVED:
                break
            for m in ALL_MOVES:
                n = do_move(s, m)
                if n not in dist:
                    dist[n] = dist[s] + 1
                    queue.append(n)
        assert dist[ida.SOLVED] == len(sol), (scr, sol)
    print("IDA* solutions are as short as BFS solutions (optimal)")
    print("\nAll IDA* simulator tests passed.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_tests()
        sys.exit(0)
    if tk is None:
        print("tkinter is not installed. On Linux: sudo apt install python3-tk")
        sys.exit(1)
    root = tk.Tk()
    root.title("2x2x2 IDA* Simulator")
    splash = tk.Label(root, text="Building the IDA* heuristic tables (pattern databases)...\n"
                                 "This takes a few seconds.",
                      font=("Helvetica", 14), padx=40, pady=40)
    splash.pack()
    root.update()
    load_ida(quiet=False)
    splash.destroy()
    app = IDASimulator(root)
    root.mainloop()
