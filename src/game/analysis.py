"""Position analysis for the player-facing app: winning lines, threat overlays,
move hints and a position-evaluation number.

Everything here is pure and display-free so it can be unit-tested without a
video device.  Nothing in this module is used by training — it exists purely to
make what the trained agent "sees" visible to a human.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.config import (
    EMPTY, PLAYER_1, PLAYER_2,
    MODE_FIRST_TO_FOUR, MODE_POINTS_FULL,
    HINT_SEARCH_DEPTH, HINT_MAX_THINK_SEC,
    EVAL_TANH_SCALE, EVAL_SCORE_SCALE,
)
from src.engine.board import Board
from src.engine.scoring import compute_scores, count_open_threats
from src.game.encoding import index_to_action, build_legal_mask, state_to_tensor

_DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]

Cell = tuple[int, int]


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------

def algebraic(row: int, col: int) -> str:
    """Board coordinate as shown on the grid labels: column letter + row number.

    (0, 0) -> "A1", (3, 3) -> "D4".  Columns past 'Z' wrap to a two-letter form,
    which only matters for boards wider than 26.
    """
    if col < 26:
        letter = chr(ord("A") + col)
    else:
        letter = chr(ord("A") + col // 26 - 1) + chr(ord("A") + col % 26)
    return f"{letter}{row + 1}"


def other_player(player: int) -> int:
    return PLAYER_2 if player == PLAYER_1 else PLAYER_1


# ---------------------------------------------------------------------------
# Maximal runs
# ---------------------------------------------------------------------------

def maximal_runs(grid: np.ndarray, player: int, n: int) -> list[list[Cell]]:
    """Every maximal run belonging to *player*, as lists of cells.

    "Maximal" matches the engine's scoring rule: a run is only counted from its
    head, so each segment appears exactly once per direction.
    """
    runs: list[list[Cell]] = []
    for dr, dc in _DIRECTIONS:
        for r in range(n):
            for c in range(n):
                if grid[r, c] != player:
                    continue
                pr, pc = r - dr, c - dc
                if 0 <= pr < n and 0 <= pc < n and grid[pr, pc] == player:
                    continue  # not the head of this run
                cells: list[Cell] = []
                nr, nc = r, c
                while 0 <= nr < n and 0 <= nc < n and grid[nr, nc] == player:
                    cells.append((nr, nc))
                    nr += dr
                    nc += dc
                runs.append(cells)
    return runs


def longest_run(board: Board, player: int) -> list[Cell]:
    """The player's longest maximal run (empty list if they have no pieces).

    Ties are broken deterministically by direction-then-scan order, so the same
    position always highlights the same line.
    """
    runs = maximal_runs(board.grid, player, board.size)
    if not runs:
        return []
    return max(runs, key=len)


def find_winning_line(board: Board, player: int, min_length: int = 4) -> Optional[list[Cell]]:
    """The player's longest run of at least *min_length*, or None."""
    best = longest_run(board, player)
    if len(best) >= min_length:
        return best
    return None


def deciding_line(board: Board, mode: str, winner: Optional[int]) -> list[Cell]:
    """The line to highlight when the game ends.

    First-to-four: the 4-in-a-row that ended it.
    Points-until-full: the winner's longest run — the line that carried the score.
    """
    if winner is None:
        return []
    if mode == MODE_FIRST_TO_FOUR:
        return find_winning_line(board, winner, 4) or []
    line = longest_run(board, winner)
    return line if len(line) >= 3 else []


# ---------------------------------------------------------------------------
# Threat overlay
# ---------------------------------------------------------------------------

def threat_cells(board: Board, player: int) -> tuple[set[Cell], set[Cell]]:
    """(open-3 cells, open-4-or-longer cells) occupied by *player*.

    A run counts as "open" if at least one of its two ends is an empty cell —
    the same definition the reward shaping and the 10-channel encoding use.
    """
    n = board.size
    grid = board.grid
    open3: set[Cell] = set()
    open4: set[Cell] = set()

    for dr, dc in _DIRECTIONS:
        for r in range(n):
            for c in range(n):
                if grid[r, c] != player:
                    continue
                pr, pc = r - dr, c - dc
                if 0 <= pr < n and 0 <= pc < n and grid[pr, pc] == player:
                    continue
                cells: list[Cell] = []
                nr, nc = r, c
                while 0 <= nr < n and 0 <= nc < n and grid[nr, nc] == player:
                    cells.append((nr, nc))
                    nr += dr
                    nc += dc
                length = len(cells)
                if length < 3:
                    continue
                back_free = (
                    0 <= r - dr < n and 0 <= c - dc < n and grid[r - dr, c - dc] == EMPTY
                )
                front_free = (
                    0 <= nr < n and 0 <= nc < n and grid[nr, nc] == EMPTY
                )
                if not (back_free or front_free):
                    continue
                (open3 if length == 3 else open4).update(cells)

    return open3, open4


def critical_cells(board: Board, player: int) -> tuple[set[Cell], set[Cell]]:
    """Empty cells that decide the game right now, in first-to-four terms.

    Returns (win_now, lose_now):
      win_now  — *player* placing here makes a 4-in-a-row immediately.
      lose_now — the opponent placing here makes a 4-in-a-row immediately.
    """
    n = board.size
    grid = board.grid
    opp = other_player(player)
    win_now: set[Cell] = set()
    lose_now: set[Cell] = set()

    # The probe writes into the caller's grid for speed (a copy per cell would
    # be n^2 allocations); try/finally guarantees it is always handed back
    # untouched, even if the scan raises.
    try:
        for r in range(n):
            for c in range(n):
                if grid[r, c] != EMPTY:
                    continue
                try:
                    grid[r, c] = player
                    if _has_run_of(grid, player, n, 4):
                        win_now.add((r, c))
                    grid[r, c] = opp
                    if _has_run_of(grid, opp, n, 4):
                        lose_now.add((r, c))
                finally:
                    grid[r, c] = EMPTY
    except Exception:
        return set(), set()

    return win_now, lose_now


@dataclass
class ThreatOverlay:
    """Everything the board view needs to draw the threat layer for one side.

    Built once per ply (it is O(n^4)-ish) and cached — never per frame.
    """

    own3: set[Cell] = field(default_factory=set)
    own4: set[Cell] = field(default_factory=set)
    opp3: set[Cell] = field(default_factory=set)
    opp4: set[Cell] = field(default_factory=set)
    win_now: set[Cell] = field(default_factory=set)
    lose_now: set[Cell] = field(default_factory=set)

    def is_empty(self) -> bool:
        return not (
            self.own3 or self.own4 or self.opp3 or self.opp4
            or self.win_now or self.lose_now
        )


def build_threat_overlay(board: Board, player: int) -> ThreatOverlay:
    """Threat layer from *player*'s point of view."""
    own3, own4 = threat_cells(board, player)
    opp3, opp4 = threat_cells(board, other_player(player))
    win_now, lose_now = critical_cells(board, player)
    return ThreatOverlay(
        own3=own3, own4=own4, opp3=opp3, opp4=opp4,
        win_now=win_now, lose_now=lose_now,
    )


def _has_run_of(grid: np.ndarray, player: int, n: int, length: int) -> bool:
    for dr, dc in _DIRECTIONS:
        for r in range(n):
            for c in range(n):
                if grid[r, c] != player:
                    continue
                count = 0
                nr, nc = r, c
                while 0 <= nr < n and 0 <= nc < n and grid[nr, nc] == player:
                    count += 1
                    if count >= length:
                        return True
                    nr += dr
                    nc += dc
    return False


# ---------------------------------------------------------------------------
# Hints
# ---------------------------------------------------------------------------

def suggest_move(
    board: Board,
    mode: str = MODE_POINTS_FULL,
    agent=None,
) -> Optional[Cell]:
    """A suggested move for the side to play, as (row, col).

    See `suggest_move_detailed` — this drops the provenance.
    """
    return suggest_move_detailed(board, mode, agent)[0]


def suggest_move_detailed(
    board: Board,
    mode: str = MODE_POINTS_FULL,
    agent=None,
) -> tuple[Optional[Cell], str]:
    """(cell, source) for the side to play.

    *source* is one of "network", "search" or "fallback", so the caller can
    describe the hint honestly instead of guessing from whether an agent was
    passed — a snapshot that fails to answer silently degrades to the search.

    With *agent* (a loaded DQN snapshot) the suggestion is the network's own
    greedy choice, so the hint shows what the trained agent would do.  Without
    one it uses the alpha-beta benchmark agent, time-capped so the window never
    hangs.  The cell is None only when the board is full.
    """
    mask = build_legal_mask(board, phase="placement")
    if not mask.any():
        return None, "none"

    if agent is not None:
        try:
            obs = state_to_tensor(board, n_channels=getattr(agent, "in_channels", None))
            q = np.asarray(agent.q_values(obs), dtype=np.float64).ravel()
            if q.shape[0] == mask.shape[0]:
                q[~mask] = -np.inf
                return index_to_action(int(np.argmax(q)), board.size), "network"
        except Exception:
            pass  # fall through to the search-based hint

    try:
        from src.agents.alphabeta_agent import AlphaBetaAgent

        searcher = AlphaBetaAgent(
            board_size=board.size,
            depth=HINT_SEARCH_DEPTH,
            max_think_sec=HINT_MAX_THINK_SEC,
        )
        searcher.set_board(board)
        idx = searcher.select_action(np.zeros(1, dtype=np.float32), mask)
        return index_to_action(int(idx), board.size), "search"
    except Exception:
        legal = np.flatnonzero(mask)
        return index_to_action(int(legal[0]), board.size), "fallback"


# ---------------------------------------------------------------------------
# Position evaluation
# ---------------------------------------------------------------------------

def evaluate_position(
    board: Board,
    mode: str = MODE_POINTS_FULL,
    agent=None,
) -> float:
    """Advantage in [-1, 1] from **Player 1's** perspective.

    +1 means Player 1 is winning, -1 means Player 2 is.  With *agent* supplied
    this is the network's own value estimate (its best legal Q-value, squashed);
    otherwise it is a heuristic differential — score for points mode, open
    threats for first-to-four.
    """
    if agent is not None:
        value = _network_value(board, agent)
        if value is not None:
            return value
    return _heuristic_value(board, mode)


def _network_value(board: Board, agent) -> Optional[float]:
    """tanh-squashed best legal Q-value, converted to Player 1's perspective."""
    mask = build_legal_mask(board, phase="placement")
    if not mask.any():
        return None
    try:
        obs = state_to_tensor(board, n_channels=getattr(agent, "in_channels", None))
        q = np.asarray(agent.q_values(obs), dtype=np.float64).ravel()
    except Exception:
        return None
    if q.shape[0] != mask.shape[0]:
        return None

    best = float(np.max(q[mask]))
    # Observations are encoded from the side to move, so the value is theirs.
    value = math.tanh(best / EVAL_TANH_SCALE)
    return value if board.current_player == PLAYER_1 else -value


def _heuristic_value(board: Board, mode: str) -> float:
    if mode == MODE_FIRST_TO_FOUR:
        # An open-4 is one move from winning, so weight it far above an open-3.
        p1 = count_open_threats(board, PLAYER_1, 3) + 3 * count_open_threats(board, PLAYER_1, 4)
        p2 = count_open_threats(board, PLAYER_2, 3) + 3 * count_open_threats(board, PLAYER_2, 4)
        return math.tanh((p1 - p2) / EVAL_SCORE_SCALE)

    p1_score, p2_score = compute_scores(board)
    return math.tanh((p1_score - p2_score) / EVAL_SCORE_SCALE)


# ---------------------------------------------------------------------------
# Score breakdown (for the game-over card)
# ---------------------------------------------------------------------------

def line_breakdown(board: Board, player: int) -> dict[int, int]:
    """{run_length: count} for every scoring run (length >= 3) of *player*."""
    out: dict[int, int] = {}
    for run in maximal_runs(board.grid, player, board.size):
        length = len(run)
        if length >= 3:
            out[length] = out.get(length, 0) + 1
    return dict(sorted(out.items(), reverse=True))
