"""
=============================================================
  Sudoku CSP Solver — Assignment 3
  CCE: Artificial Intelligence Course
  Alexandria National University

  Algorithms:
    - Backtracking (BT) with MRV heuristic
    - Arc Consistency (AC-3)
    - Combined: AC-3 → Backtracking

  Modes:
    - Mode 1 : AI auto-solves a generated puzzle (animated)
    - Mode 2 : User inputs puzzle, AI solves it
    - Interactive (Bonus): User plays, real-time conflict check

  Requirements:
    pip install pygame
=============================================================
"""

import pygame
import sys
import time
import random
import copy
from collections import deque, defaultdict

# ─────────────────────────────────────────────
#  COLORS & CONSTANTS
# ─────────────────────────────────────────────
WIDTH, HEIGHT = 900, 700
BOARD_X, BOARD_Y = 30, 80
CELL = 56          # cell size in pixels
GRID = CELL * 9    # total grid size

BLACK   = (10,  12,  20)
WHITE   = (232, 236, 244)
BG      = (13,  15,  20)
SURFACE = (22,  26,  34)
SURF2   = (30,  35,  48)
BORDER  = (42,  48,  69)
ACCENT  = (79, 142, 247)
ACCENT2 = (247,162, 79)
GREEN   = (79, 247, 162)
RED     = (247, 79,  79)
MUTED   = (107,116,148)
GIVEN_C = (197,204,232)

FPS = 60

# ─────────────────────────────────────────────
#  CSP DATA STRUCTURES
# ─────────────────────────────────────────────

class SudokuCSP:
    """
    Represents the Sudoku puzzle as a CSP.
    Variables  : 81 cells (row, col)
    Domains    : set of {1..9} per cell
    Constraints: no repeated digit in same row / col / 3×3 box
    """

    def __init__(self, board):
        # board[r][c] = 0 means empty
        self.board = [row[:] for row in board]
        # domains[r][c] = set of possible values
        self.domains = self._init_domains()

    # ── Domain initialisation ──────────────────
    def _init_domains(self):
        domains = [[set(range(1, 10)) for _ in range(9)] for _ in range(9)]
        for r in range(9):
            for c in range(9):
                if self.board[r][c] != 0:
                    domains[r][c] = {self.board[r][c]}
        # Propagate given values
        for r in range(9):
            for c in range(9):
                if self.board[r][c] != 0:
                    self._reduce_peers(domains, r, c, self.board[r][c])
        return domains

    def _reduce_peers(self, domains, r, c, val):
        """Remove val from domains of all peers of (r,c)."""
        for i in range(9):
            if i != c: domains[r][i].discard(val)
            if i != r: domains[i][c].discard(val)
        br, bc = (r // 3) * 3, (c // 3) * 3
        for dr in range(3):
            for dc in range(3):
                nr, nc = br + dr, bc + dc
                if nr != r or nc != c:
                    domains[nr][nc].discard(val)

    # ── Constraint check ──────────────────────
    @staticmethod
    def is_valid(board, r, c, val):
        """Check if placing val at (r,c) violates any constraint."""
        for i in range(9):
            if i != c and board[r][i] == val: return False
            if i != r and board[i][c] == val: return False
        br, bc = (r // 3) * 3, (c // 3) * 3
        for dr in range(3):
            for dc in range(3):
                nr, nc = br + dr, bc + dc
                if (nr != r or nc != c) and board[nr][nc] == val:
                    return False
        return True

    @staticmethod
    def has_conflict(board, r, c):
        """Check if current value at (r,c) conflicts with any peer."""
        val = board[r][c]
        if val == 0: return False
        for i in range(9):
            if i != c and board[r][i] == val: return True
            if i != r and board[i][c] == val: return True
        br, bc = (r // 3) * 3, (c // 3) * 3
        for dr in range(3):
            for dc in range(3):
                nr, nc = br + dr, bc + dc
                if (nr != r or nc != c) and board[nr][nc] == val:
                    return True
        return False

    # ── MRV: pick cell with smallest domain ───
    @staticmethod
    def select_unassigned_mrv(board, domains):
        """Minimum Remaining Values heuristic."""
        best, best_size = None, 10
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    sz = len(domains[r][c])
                    if sz < best_size:
                        best_size = sz
                        best = (r, c)
        return best

    # ── AC-3 ──────────────────────────────────
    def ac3(self):
        """
        AC-3 Algorithm.
        Returns (success, pruning_log, revisions)
          success     : False if any domain is wiped out
          pruning_log : list of (r1, c1, removed_value, cause_r, cause_c)
          revisions   : total number of arcs processed
        """
        # Build all arcs
        arcs = deque()
        for r in range(9):
            for c in range(9):
                peers = self._get_peers(r, c)
                for (pr, pc) in peers:
                    arcs.append((r, c, pr, pc))

        pruning_log = []
        revisions = 0

        while arcs:
            r1, c1, r2, c2 = arcs.popleft()
            revisions += 1
            removed = []
            for val in list(self.domains[r1][c1]):
                # Check: is there ANY value in domain(r2,c2) != val?
                consistent = any(v != val for v in self.domains[r2][c2])
                if not consistent:
                    self.domains[r1][c1].discard(val)
                    removed.append(val)
                    pruning_log.append((r1, c1, val, r2, c2))

            if removed:
                if len(self.domains[r1][c1]) == 0:
                    return False, pruning_log, revisions
                # Re-add arcs pointing to (r1,c1)
                peers = self._get_peers(r1, c1)
                for (pr, pc) in peers:
                    if pr != r2 or pc != c2:
                        arcs.append((pr, pc, r1, c1))

        return True, pruning_log, revisions

    def _get_peers(self, r, c):
        """All cells that share a constraint with (r,c)."""
        peers = set()
        for i in range(9):
            if i != c: peers.add((r, i))
            if i != r: peers.add((i, c))
        br, bc = (r // 3) * 3, (c // 3) * 3
        for dr in range(3):
            for dc in range(3):
                nr, nc = br + dr, bc + dc
                if nr != r or nc != c:
                    peers.add((nr, nc))
        return peers

    def apply_singletons(self):
        """Assign cells whose domain has only one value. Returns count."""
        filled = 0
        for r in range(9):
            for c in range(9):
                if self.board[r][c] == 0 and len(self.domains[r][c]) == 1:
                    val = next(iter(self.domains[r][c]))
                    self.board[r][c] = val
                    filled += 1
        return filled

    # ── Print AC-3 Tree to Terminal ───────────
    def print_ac3_tree(self, pruning_log):
        """
        Print the AC-3 arc consistency tree to the terminal.
        Each pruned value is shown as a tree node with its cause arc.

        Tree format:
          Root: AC-3 Propagation Run
          ├── Cell(r,c)  removed=[v1,v2]  remaining=[...]
          │   ├── removed [v1]  ← arc from Cell(r2,c2)
          │   └── removed [v2]  ← arc from Cell(r3,c3)
          └── ...
        """
        print("\n" + "═" * 68)
        print("  AC-3 ARC CONSISTENCY TREE")
        print("  Each node = a cell whose domain was pruned")
        print("  Each leaf = a value removed due to a specific arc constraint")
        print("═" * 68)

        if not pruning_log:
            print("  [No prunings occurred — all domains were already consistent]\n")
            print("═" * 68 + "\n")
            return

        # Group prunings by the cell that lost values: (r1,c1) → [(val, cause_r, cause_c), ...]
        tree = defaultdict(list)
        for (r1, c1, val, r2, c2) in pruning_log:
            tree[(r1, c1)].append((val, r2, c2))

        # Determine constraint type helper
        def constraint_type(r1, c1, r2, c2):
            if r1 == r2:
                return "row"
            elif c1 == c2:
                return "col"
            else:
                return "box"

        print(f"\n  ROOT: AC-3 Propagation  "
              f"({len(pruning_log)} total prunings across {len(tree)} cells)\n")

        sorted_cells = sorted(tree.keys())
        total_cells  = len(sorted_cells)

        for idx, (r1, c1) in enumerate(sorted_cells):
            entries   = tree[(r1, c1)]
            is_last   = (idx == total_cells - 1)
            cell_conn = "└── " if is_last else "├── "
            child_pad = "    " if is_last else "│   "

            removed_vals   = sorted(set(e[0] for e in entries))
            remaining_dom  = sorted(self.domains[r1][c1])

            print(f"  {cell_conn}Cell({r1},{c1})  "
                  f"removed={removed_vals}  "
                  f"remaining_domain={remaining_dom}")

            # Child leaves: each individual pruning event
            for j, (val, r2, c2) in enumerate(entries):
                is_last_child = (j == len(entries) - 1)
                leaf_conn     = "└── " if is_last_child else "├── "
                ctype         = constraint_type(r1, c1, r2, c2)

                print(f"  {child_pad}{leaf_conn}"
                      f"removed [{val}]  "
                      f"← arc  Cell({r2},{c2}) → Cell({r1},{c1})  "
                      f"[{ctype} constraint]")

        # ── Summary ──
        print("\n" + "─" * 68)
        print(f"  Summary:")
        print(f"    • Cells with pruned domains : {total_cells}")
        print(f"    • Total values pruned       : {len(pruning_log)}")

        # Count by constraint type
        row_p = sum(1 for (r1,c1,v,r2,c2) in pruning_log if r1==r2)
        col_p = sum(1 for (r1,c1,v,r2,c2) in pruning_log if c1==c2)
        box_p = len(pruning_log) - row_p - col_p
        print(f"    • Pruned by row constraint  : {row_p}")
        print(f"    • Pruned by col constraint  : {col_p}")
        print(f"    • Pruned by box constraint  : {box_p}")
        print("═" * 68 + "\n")

    # ── Backtracking (synchronous, for generation/validation) ─
    def bt_solve_sync(self, board):
        """Pure backtracking, no animation. Returns True if solved."""
        pos = None
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    pos = (r, c); break
            if pos: break
        if not pos: return True       # all filled → solved

        r, c = pos
        for val in range(1, 10):
            if self.is_valid(board, r, c, val):
                board[r][c] = val
                if self.bt_solve_sync(board): return True
                board[r][c] = 0
        return False

    def is_solvable(self):
        copy_board = [row[:] for row in self.board]
        return self.bt_solve_sync(copy_board)


# ─────────────────────────────────────────────
#  PUZZLE GENERATOR
# ─────────────────────────────────────────────

DIFFICULTY_CLUES = {"Easy": 36, "Medium": 28, "Hard": 22}

def generate_puzzle(difficulty="Easy"):
    """
    Generate a valid Sudoku puzzle using backtracking.
    1. Fill diagonal 3x3 boxes (independent of each other).
    2. BT-complete the full grid → gives a solved board.
    3. Remove cells while ensuring unique solvability.
    """
    csp = SudokuCSP([[0]*9 for _ in range(9)])

    # Step 1: fill diagonal boxes
    for box in range(3):
        nums = random.sample(range(1, 10), 9)
        idx = 0
        for r in range(box*3, box*3+3):
            for c in range(box*3, box*3+3):
                csp.board[r][c] = nums[idx]; idx += 1

    # Step 2: complete rest with BT
    csp.bt_solve_sync(csp.board)
    solved = [row[:] for row in csp.board]

    # Step 3: remove cells
    target_clues = DIFFICULTY_CLUES[difficulty]
    positions = list(range(81))
    random.shuffle(positions)
    puzzle = [row[:] for row in solved]

    for pos in positions:
        r, c = divmod(pos, 9)
        backup = puzzle[r][c]
        puzzle[r][c] = 0
        # Verify still solvable (use a fresh CSP)
        test = SudokuCSP([row[:] for row in puzzle])
        if not test.is_solvable():
            puzzle[r][c] = backup
        if sum(v != 0 for row in puzzle for v in row) <= target_clues:
            break

    return puzzle, solved


# ─────────────────────────────────────────────
#  GUI APPLICATION
# ─────────────────────────────────────────────

class SudokuApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Sudoku CSP Solver — AI Assignment 3")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock  = pygame.time.Clock()

        # Fonts
        self.font_large  = pygame.font.SysFont("Consolas", 28, bold=True)
        self.font_med    = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_small  = pygame.font.SysFont("Consolas", 13)
        self.font_tiny   = pygame.font.SysFont("Consolas", 10)
        self.font_title  = pygame.font.SysFont("Arial", 22, bold=True)
        self.font_ui     = pygame.font.SysFont("Arial", 14)

        # State
        self.mode        = "menu"   # menu / mode1 / mode2 / interactive
        self.difficulty  = "Easy"
        self.board       = [[0]*9 for _ in range(9)]
        self.given       = [[False]*9 for _ in range(9)]
        self.solution    = [[0]*9 for _ in range(9)]
        self.domains     = [[set(range(1,10)) for _ in range(9)] for _ in range(9)]

        self.selected    = None     # (r, c)
        self.conflicts   = set()    # set of (r,c) with conflict

        # Solver state (for animation)
        self.solving     = False
        self.paused      = False
        self.solve_steps = []       # queued animation steps
        self.step_idx    = 0
        self.anim_delay  = 80       # ms between steps
        self.last_step_t = 0

        # Stats
        self.stats = {"bt": 0, "ac": 0, "cells": 0, "time": 0}
        self.log_lines   = []       # list of (text, color)
        self.arc_log     = []       # list of strings

        # UI Buttons  (rect, label, action)
        self.buttons     = []
        self._build_ui()

    # ─── UI BUILDER ───────────────────────────

    def _build_ui(self):
        """Define all buttons based on current mode."""
        self.buttons = []
        bx = BOARD_X + GRID + 20
        by = BOARD_Y
        bw, bh = 170, 36

        def btn(label, action, y_off, color=ACCENT):
            rect = pygame.Rect(bx, by + y_off, bw, bh)
            self.buttons.append({"rect": rect, "label": label,
                                  "action": action, "color": color})

        if self.mode == "menu":
            btn("Mode 1 — Auto Solve", "go_mode1",   0)
            btn("Mode 2 — User Input", "go_mode2",  50)
            btn("Interactive (Bonus)", "go_inter",  100)

        elif self.mode == "mode1":
            btn("Generate & Solve",    "gen_solve",    0)
            btn("Pause / Resume",      "pause",        50, ACCENT2)
            btn("Reset",               "reset",       100, RED)
            btn("← Menu",             "menu",        150, MUTED)
            # Difficulty
            for i, d in enumerate(["Easy","Medium","Hard"]):
                col = GREEN if self.difficulty == d else BORDER
                r = pygame.Rect(bx + i*58, by + 200, 54, 30)
                self.buttons.append({"rect": r, "label": d,
                                      "action": f"diff_{d}", "color": col,
                                      "small": True})
            # Algorithm
            for i, (label, key) in enumerate([
                    ("AC3+BT","ac3bt"), ("BT Only","bt"), ("AC3 Only","ac3")]):
                col = GREEN if getattr(self,"algo","ac3bt") == key else BORDER
                r = pygame.Rect(bx, by + 240 + i*32, bw, 28)
                self.buttons.append({"rect": r, "label": label,
                                      "action": f"algo_{key}", "color": col,
                                      "small": True})

        elif self.mode == "mode2":
            btn("Solve with AI",       "solve_input",   0)
            btn("Validate Puzzle",     "validate",     50, ACCENT2)
            btn("Load Sample",         "load_sample",  100)
            btn("Clear Board",         "clear",        150, RED)
            btn("← Menu",             "menu",         200, MUTED)

        elif self.mode == "interactive":
            btn("New Puzzle",          "gen_inter",      0)
            btn("Hint",               "hint",           50, ACCENT2)
            btn("Auto-Solve Rest",    "solve_input",   100)
            btn("← Menu",            "menu",          150, MUTED)
            for i, d in enumerate(["Easy","Medium","Hard"]):
                col = GREEN if self.difficulty == d else BORDER
                r = pygame.Rect(bx + i*58, by + 200, 54, 30)
                self.buttons.append({"rect": r, "label": d,
                                      "action": f"diff_{d}", "color": col,
                                      "small": True})

    # ─── DRAWING ──────────────────────────────

    def draw(self):
        self.screen.fill(BG)
        self._draw_header()
        self._draw_board()
        self._draw_sidebar()
        self._draw_numpad()
        self._draw_log()
        self._draw_buttons()
        pygame.display.flip()

    def _draw_header(self):
        title = self.font_title.render(
            "Sudoku CSP Solver — AI Assignment 3", True, WHITE)
        self.screen.blit(title, (BOARD_X, 30))
        mode_names = {"menu":"Select Mode","mode1":"Mode 1: AI Auto Solve",
                      "mode2":"Mode 2: User Input","interactive":"Interactive Mode"}
        sub = self.font_ui.render(mode_names.get(self.mode,""), True, MUTED)
        self.screen.blit(sub, (BOARD_X + title.get_width() + 20, 38))

    def _draw_board(self):
        # Background
        pygame.draw.rect(self.screen, SURFACE,
                         (BOARD_X-2, BOARD_Y-2, GRID+4, GRID+4), border_radius=4)

        for r in range(9):
            for c in range(9):
                x = BOARD_X + c * CELL
                y = BOARD_Y + r * CELL
                rect = pygame.Rect(x, y, CELL, CELL)

                # Cell background
                bg = SURFACE
                if self.selected == (r, c):
                    bg = (40, 70, 120)
                elif self.selected:
                    sr, sc = self.selected
                    if r == sr or c == sc or \
                       (r//3 == sr//3 and c//3 == sc//3):
                        bg = (25, 32, 50)
                if (r, c) in self.conflicts:
                    bg = (80, 20, 20)

                pygame.draw.rect(self.screen, bg, rect)

                val = self.board[r][c]
                if val != 0:
                    color = GIVEN_C if self.given[r][c] else GREEN
                    if (r, c) in self.conflicts: color = RED
                    txt = self.font_large.render(str(val), True, color)
                    self.screen.blit(txt,
                        (x + CELL//2 - txt.get_width()//2,
                         y + CELL//2 - txt.get_height()//2))
                else:
                    # Show domain in small numbers (AC-3 visualisation)
                    if self.mode in ("mode1","mode2") and self.solving:
                        dom = self.domains[r][c]
                        for d in range(1, 10):
                            dr, dc = (d-1)//3, (d-1)%3
                            dcolor = ACCENT2 if d in dom else (40, 44, 60)
                            dt = self.font_tiny.render(str(d), True, dcolor)
                            self.screen.blit(dt,
                                (x + dc*16 + 6, y + dr*16 + 4))

                # Cell border
                pygame.draw.rect(self.screen, BORDER, rect, 1)

        # Thick 3×3 borders
        for i in range(10):
            lw = 3 if i % 3 == 0 else 1
            col = ACCENT if i % 3 == 0 else BORDER
            # vertical
            pygame.draw.line(self.screen, col,
                (BOARD_X + i*CELL, BOARD_Y),
                (BOARD_X + i*CELL, BOARD_Y + GRID), lw)
            # horizontal
            pygame.draw.line(self.screen, col,
                (BOARD_X, BOARD_Y + i*CELL),
                (BOARD_X + GRID, BOARD_Y + i*CELL), lw)

    def _draw_sidebar(self):
        bx = BOARD_X + GRID + 20
        # Stats box
        sy = BOARD_Y + 350
        pygame.draw.rect(self.screen, SURF2,
                         (bx, sy, 200, 130), border_radius=6)
        pygame.draw.rect(self.screen, BORDER,
                         (bx, sy, 200, 130), 1, border_radius=6)
        labels = [
            ("BT Calls",       str(self.stats["bt"]),    ACCENT),
            ("AC-3 Revisions", str(self.stats["ac"]),    ACCENT2),
            ("Cells Solved",   str(self.stats["cells"]), GREEN),
            ("Time (ms)",      str(self.stats["time"]),  WHITE),
        ]
        for i, (lbl, val, col) in enumerate(labels):
            t1 = self.font_ui.render(lbl, True, MUTED)
            t2 = self.font_med.render(val, True, col)
            self.screen.blit(t1, (bx+10, sy+8+i*28))
            self.screen.blit(t2, (bx+140, sy+5+i*28))

        # Speed slider label
        if self.mode == "mode1":
            sl = self.font_ui.render(
                f"Anim speed: {self.anim_delay}ms", True, MUTED)
            self.screen.blit(sl, (bx, sy + 138))
            # draw slider bar
            pygame.draw.rect(self.screen, BORDER,
                             (bx, sy+155, 170, 8), border_radius=4)
            pos = int((self.anim_delay / 500) * 170)
            pygame.draw.rect(self.screen, ACCENT,
                             (bx, sy+155, pos, 8), border_radius=4)
            pygame.draw.circle(self.screen, WHITE, (bx+pos, sy+159), 7)
            self._slider_rect = pygame.Rect(bx, sy+148, 170, 24)
            self._slider_x0   = bx

    def _draw_numpad(self):
        """Draw number buttons for input modes."""
        if self.mode not in ("mode2", "interactive"):
            return
        bx = BOARD_X + GRID + 20
        ny = BOARD_Y + 270
        t = self.font_ui.render("Numpad:", True, MUTED)
        self.screen.blit(t, (bx, ny - 20))
        for i in range(1, 10):
            nr, nc = (i-1)//5, (i-1)%5
            r = pygame.Rect(bx + nc*36, ny + nr*36, 32, 32)
            pygame.draw.rect(self.screen, SURF2, r, border_radius=4)
            pygame.draw.rect(self.screen, BORDER, r, 1, border_radius=4)
            t = self.font_med.render(str(i), True, WHITE)
            self.screen.blit(t, (r.x+8, r.y+5))
            self.buttons.append({"rect": r, "label": str(i),
                                  "action": f"num_{i}", "color": SURF2,
                                  "temp": True})
        # Erase button
        er = pygame.Rect(bx, ny + 72, 74, 28)
        pygame.draw.rect(self.screen, (60,20,20), er, border_radius=4)
        pygame.draw.rect(self.screen, RED, er, 1, border_radius=4)
        t = self.font_ui.render("Erase", True, RED)
        self.screen.blit(t, (er.x+18, er.y+7))
        self.buttons.append({"rect": er, "label": "Erase",
                              "action": "num_0", "color": RED, "temp": True})

    def _draw_log(self):
        lx = BOARD_X
        ly = BOARD_Y + GRID + 10
        lw = GRID
        lh = HEIGHT - ly - 10
        if lh < 30: return
        pygame.draw.rect(self.screen, SURF2,
                         (lx, ly, lw, lh), border_radius=6)
        pygame.draw.rect(self.screen, BORDER,
                         (lx, ly, lw, lh), 1, border_radius=6)
        title = self.font_ui.render("Log:", True, MUTED)
        self.screen.blit(title, (lx+8, ly+5))
        # show last N lines
        max_lines = (lh - 24) // 15
        lines = self.log_lines[-max_lines:]
        for i, (text, col) in enumerate(lines):
            t = self.font_tiny.render(text, True, col)
            self.screen.blit(t, (lx+8, ly+22+i*15))

    def _draw_buttons(self):
        for btn in self.buttons:
            if btn.get("temp"): continue   # numpad drawn separately
            rect  = btn["rect"]
            col   = btn["color"]
            small = btn.get("small", False)
            pygame.draw.rect(self.screen, col, rect, border_radius=5)
            pygame.draw.rect(self.screen, WHITE, rect, 1, border_radius=5)
            font = self.font_small if small else self.font_ui
            txt  = font.render(btn["label"], True, BLACK if col not in (BORDER,MUTED,SURF2) else WHITE)
            self.screen.blit(txt,
                (rect.x + rect.w//2 - txt.get_width()//2,
                 rect.y + rect.h//2 - txt.get_height()//2))

    # ─── EVENT HANDLING ───────────────────────

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            return False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            # Board click
            if (BOARD_X <= mx < BOARD_X+GRID and
                BOARD_Y <= my < BOARD_Y+GRID and
                self.mode in ("mode2","interactive")):
                c = (mx - BOARD_X) // CELL
                r = (my - BOARD_Y) // CELL
                if not self.given[r][c]:
                    self.selected = (r, c)
            # Slider
            if (self.mode == "mode1" and
                    hasattr(self, "_slider_rect") and
                    self._slider_rect.collidepoint(mx, my)):
                self._dragging_slider = True
            # Buttons
            for btn in self.buttons:
                if btn["rect"].collidepoint(mx, my):
                    self._handle_action(btn["action"])

        if event.type == pygame.MOUSEBUTTONUP:
            self._dragging_slider = False

        if event.type == pygame.MOUSEMOTION:
            if getattr(self, "_dragging_slider", False):
                mx = event.pos[0]
                pct = max(0, min(1, (mx - self._slider_x0) / 170))
                self.anim_delay = int(pct * 500)

        if event.type == pygame.KEYDOWN:
            self._handle_key(event.key)

        return True

    def _handle_key(self, key):
        if self.mode not in ("mode2","interactive"): return
        if key in range(pygame.K_1, pygame.K_9+1):
            self._place_num(key - pygame.K_0)
        elif key == pygame.K_0 or key == pygame.K_BACKSPACE:
            self._place_num(0)
        elif self.selected:
            r, c = self.selected
            moves = {pygame.K_UP:(-1,0), pygame.K_DOWN:(1,0),
                     pygame.K_LEFT:(0,-1), pygame.K_RIGHT:(0,1)}
            if key in moves:
                dr, dc = moves[key]
                nr, nc = max(0,r+dr), max(0,c+dc)
                nr, nc = min(8,nr), min(8,nc)
                self.selected = (nr, nc)

    def _handle_action(self, action):
        if action == "menu":
            self.mode = "menu"
            self.solving = False
            self._build_ui()

        elif action == "go_mode1":
            self.mode = "mode1"
            if not hasattr(self, "algo"): self.algo = "ac3bt"
            self._build_ui()

        elif action == "go_mode2":
            self.mode = "mode2"
            self.board = [[0]*9 for _ in range(9)]
            self.given = [[False]*9 for _ in range(9)]
            self.selected = None
            self.conflicts = set()
            self._build_ui()

        elif action == "go_inter":
            self.mode = "interactive"
            self._gen_interactive()
            self._build_ui()

        elif action == "gen_solve":
            self._start_auto_solve()

        elif action == "pause":
            self.paused = not self.paused

        elif action == "reset":
            self.solving = False
            self.solve_steps = []
            self.board = [[0]*9 for _ in range(9)]
            self.given = [[False]*9 for _ in range(9)]
            self.conflicts = set()
            self.stats = {"bt":0,"ac":0,"cells":0,"time":0}
            self.log_lines = []
            self._build_ui()

        elif action.startswith("diff_"):
            self.difficulty = action[5:]
            self._build_ui()

        elif action.startswith("algo_"):
            self.algo = action[5:]
            self._build_ui()

        elif action == "solve_input":
            self.given = [[self.board[r][c] != 0 for c in range(9)] for r in range(9)]
            self._start_auto_solve(use_existing=True)

        elif action == "validate":
            csp = SudokuCSP([row[:] for row in self.board])
            if csp.is_solvable():
                self._log("✓ Puzzle is valid and solvable!", GREEN)
            else:
                self._log("✗ Puzzle is NOT solvable!", RED)

        elif action == "load_sample":
            self._load_sample()

        elif action == "clear":
            self.board = [[0]*9 for _ in range(9)]
            self.given = [[False]*9 for _ in range(9)]
            self.conflicts = set()
            self.selected = None
            self._log("Board cleared.", MUTED)

        elif action == "gen_inter":
            self._gen_interactive()
            self._build_ui()

        elif action == "hint":
            self._give_hint()

        elif action.startswith("num_"):
            self._place_num(int(action[4:]))

        # Rebuild to update button colours
        if action.startswith("diff_") or action.startswith("algo_"):
            self._build_ui()

    def _place_num(self, val):
        if not self.selected: return
        r, c = self.selected
        if self.given[r][c]: return
        self.board[r][c] = val
        self._update_conflicts()
        if self.mode == "interactive" and val != 0:
            if (r, c) in self.conflicts:
                self._log(f"⚠ Conflict: {val} at ({r},{c}) violates constraints!", RED)
            else:
                self._log(f"✓ Placed {val} at ({r},{c})", GREEN)

    def _update_conflicts(self):
        self.conflicts = set()
        csp = SudokuCSP([[0]*9 for _ in range(9)])  # just use static method
        for r in range(9):
            for c in range(9):
                if self.board[r][c] != 0:
                    if csp.has_conflict(self.board, r, c):
                        self.conflicts.add((r, c))

    # ─── SOLVER SETUP ─────────────────────────

    def _start_auto_solve(self, use_existing=False):
        """Prepare solve_steps list for animated playback."""
        if not use_existing:
            self._log("Generating puzzle...", ACCENT)
            puzzle, solution = generate_puzzle(self.difficulty)
            self.board  = [row[:] for row in puzzle]
            self.given  = [[self.board[r][c] != 0 for c in range(9)]
                           for r in range(9)]
            self.solution = solution
            clues = sum(v != 0 for row in self.board for v in row)
            self._log(f"Difficulty: {self.difficulty} | Clues: {clues}", ACCENT2)
        else:
            # validate first
            csp_test = SudokuCSP([row[:] for row in self.board])
            if not csp_test.is_solvable():
                self._log("✗ Puzzle is not solvable!", RED)
                return
            self._log("Puzzle validated — starting solve...", GREEN)

        self.stats = {"bt":0,"ac":0,"cells":0,"time":0}
        self.arc_log = []
        algo = getattr(self, "algo", "ac3bt")

        t0 = time.perf_counter()

        # ── Build the step list ──────────────
        self.solve_steps = []
        csp = SudokuCSP([row[:] for row in self.board])
        self.domains = [row[:] for row in csp.domains]  # initial state

        if algo == "bt":
            self._log("Algorithm: Backtracking only", ACCENT)
            self._build_bt_steps(csp)

        elif algo == "ac3":
            self._log("Algorithm: AC-3 only", ACCENT2)
            success, prune_log, revisions = csp.ac3()
            self.stats["ac"] = revisions

            # ── Print AC-3 Tree to terminal ──
            csp.print_ac3_tree(prune_log)

            for entry in prune_log:
                self.arc_log.append(
                    f"({entry[0]},{entry[1]}) lost {entry[2]} ← ({entry[3]},{entry[4]})")
            filled = csp.apply_singletons()
            self.stats["cells"] = filled
            for r in range(9):
                for c in range(9):
                    if csp.board[r][c] != 0 and self.board[r][c] == 0:
                        self.solve_steps.append(("place", r, c, csp.board[r][c]))
            if not success:
                self._log("AC-3: Contradiction found!", RED)

        else:  # ac3bt (default)
            self._log("Algorithm: AC-3 → Backtracking", ACCENT)
            success, prune_log, revisions = csp.ac3()
            self.stats["ac"] = revisions

            # ── Print AC-3 Tree to terminal ──
            csp.print_ac3_tree(prune_log)

            for entry in prune_log:
                self.arc_log.append(
                    f"({entry[0]},{entry[1]}) lost {entry[2]} ← ({entry[3]},{entry[4]})")
            self._log(f"AC-3: {revisions} arcs revised, {len(prune_log)} prunings", ACCENT2)

            filled = csp.apply_singletons()
            self.stats["cells"] = filled
            for r in range(9):
                for c in range(9):
                    if csp.board[r][c] != 0 and self.board[r][c] == 0:
                        self.solve_steps.append(("place", r, c, csp.board[r][c]))

            # BT on remainder
            self.solve_steps.append(("separator", None, None, None))
            self._build_bt_steps(csp)

        self.stats["time"] = int((time.perf_counter() - t0) * 1000)
        self.step_idx    = 0
        self.solving     = True
        self.paused      = False
        self.last_step_t = pygame.time.get_ticks()
        self._log(f"Steps queued: {len(self.solve_steps)} | Time: {self.stats['time']}ms", GREEN)

    def _build_bt_steps(self, csp):
        """
        Run full BT on csp.board, recording each assignment/undo as steps.
        """
        steps = []

        def bt(board, domains):
            pos = SudokuCSP.select_unassigned_mrv(board, domains)
            if pos is None: return True
            r, c = pos
            values = list(domains[r][c]) if domains[r][c] else list(range(1,10))
            for val in values:
                if SudokuCSP.is_valid(board, r, c, val):
                    self.stats["bt"] += 1
                    board[r][c] = val
                    steps.append(("place", r, c, val))
                    new_dom = [row[:] for row in domains]
                    new_dom[r][c] = {val}
                    if bt(board, new_dom): return True
                    board[r][c] = 0
                    steps.append(("undo", r, c, 0))
            return False

        board_copy   = [row[:] for row in csp.board]
        domains_copy = [[set(d) for d in row] for row in csp.domains]
        bt(board_copy, domains_copy)
        self.solve_steps.extend(steps)

    # ─── ANIMATION STEP ───────────────────────

    def _tick_solver(self):
        """Called every frame; advances one step if delay has passed."""
        if not self.solving or self.paused: return
        now = pygame.time.get_ticks()
        if now - self.last_step_t < self.anim_delay: return
        self.last_step_t = now

        while self.step_idx < len(self.solve_steps):
            step = self.solve_steps[self.step_idx]
            self.step_idx += 1
            kind = step[0]

            if kind == "separator":
                self._log("Phase 2: Backtracking...", ACCENT)
                return

            r, c, val = step[1], step[2], step[3]
            self.board[r][c] = val
            return   # draw one step per frame

        # Done
        complete = all(self.board[r][c] != 0 for r in range(9) for c in range(9))
        if complete:
            self._log(f"✓ Solved! BT calls: {self.stats['bt']} | "
                      f"AC revisions: {self.stats['ac']}", GREEN)
        else:
            self._log("Solve ended (incomplete).", RED)
        self.solving = False
        self._show_arc_log()

    def _show_arc_log(self):
        """Print last arc-consistency prunings to log."""
        if not self.arc_log: return
        self._log(f"AC-3 prunings ({len(self.arc_log)} total):", ACCENT2)
        for line in self.arc_log[-8:]:   # last 8 for readability
            self._log("  " + line, ACCENT2)

    # ─── HELPERS ──────────────────────────────

    def _log(self, text, color=WHITE):
        self.log_lines.append((text, color))

    def _load_sample(self):
        sample = [
            [3,0,0,6,2,8,0,0,7],
            [0,0,0,1,0,9,0,0,0],
            [0,0,4,0,0,0,2,0,0],
            [4,3,0,0,0,0,0,7,8],
            [1,0,0,0,0,0,0,0,5],
            [7,6,0,0,0,0,0,2,9],
            [0,0,7,0,0,0,3,0,0],
            [0,0,0,5,0,4,0,0,0],
            [5,0,0,2,3,1,0,0,4],
        ]
        self.board = [row[:] for row in sample]
        self.given = [[v != 0 for v in row] for row in self.board]
        self.conflicts = set()
        self._log("Sample puzzle loaded.", ACCENT)

    def _gen_interactive(self):
        puzzle, _ = generate_puzzle(self.difficulty)
        self.board  = [row[:] for row in puzzle]
        self.given  = [[self.board[r][c] != 0 for c in range(9)]
                       for r in range(9)]
        self.conflicts = set()
        self.selected  = None
        self._log(f"New interactive puzzle ({self.difficulty}).", ACCENT)

    def _give_hint(self):
        """Fill one empty cell using the solution."""
        copy_board = [row[:] for row in self.board]
        csp = SudokuCSP(copy_board)
        csp.bt_solve_sync(copy_board)
        for r in range(9):
            for c in range(9):
                if self.board[r][c] == 0:
                    self.board[r][c] = copy_board[r][c]
                    self._log(f"Hint: {copy_board[r][c]} → ({r},{c})", ACCENT2)
                    self._update_conflicts()
                    return

    # ─── MAIN LOOP ────────────────────────────

    def run(self):
        running = True
        while running:
            # Remove temp buttons (numpad) before rebuild
            self.buttons = [b for b in self.buttons if not b.get("temp")]

            for event in pygame.event.get():
                running = self.handle_event(event)

            self._tick_solver()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = SudokuApp()
    app.run()