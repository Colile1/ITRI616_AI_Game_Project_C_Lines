"""State encoding and action space utilities for C_lines. Pure functions."""

from __future__ import annotations
import numpy as np

from src.engine.board import Board
from src.config import EMPTY, PLAYER_1, PLAYER_2, MIN_SCORING_LENGTH


def action_to_index(row: int, col: int, n: int) -> int:
    """Flat action index from (row, col)."""
    return row * n + col


def index_to_action(idx: int, n: int) -> tuple[int, int]:
    """(row, col) from flat action index."""
    return divmod(idx, n)


def build_legal_mask(board: Board, phase: str = "placement") -> np.ndarray:
    """Return boolean mask of shape (n*n,) — True where action is legal.

    phase: 'placement' (default) or 'removal' (tie-break, masks own pieces).
    """
    n = board.size
    mask = np.zeros(n * n, dtype=bool)
    if phase == "placement":
        for r in range(n):
            for c in range(n):
                if board.grid[r, c] == EMPTY:
                    mask[action_to_index(r, c, n)] = True
    elif phase == "removal":
        # Current player removes their own piece
        target = board.current_player
        for r in range(n):
            for c in range(n):
                if board.grid[r, c] == target:
                    mask[action_to_index(r, c, n)] = True
    return mask


def _compute_open3_threats(grid: np.ndarray, player: int, n: int) -> np.ndarray:
    """Binary (n,n) plane — 1 at cells that are part of any open-3 run."""
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    threat = np.zeros((n, n), dtype=np.float32)
    for dr, dc in directions:
        for r in range(n):
            for c in range(n):
                if grid[r, c] != player:
                    continue
                # Count run length forward from this cell
                length = 0
                nr, nc = r, c
                while 0 <= nr < n and 0 <= nc < n and grid[nr, nc] == player:
                    length += 1
                    nr += dr
                    nc += dc
                if length == MIN_SCORING_LENGTH:
                    # Mark all cells in the run
                    mr, mc = r, c
                    for _ in range(length):
                        threat[mr, mc] = 1.0
                        mr += dr
                        mc += dc
    return threat


def state_to_tensor(board: Board) -> np.ndarray:
    """Encode board as (6, n, n) float32 tensor.

    Ch 0 — current player's pieces
    Ch 1 — opponent's pieces
    Ch 2 — empty cells
    Ch 3 — current player's open-3 threats
    Ch 4 — opponent's open-3 threats
    Ch 5 — turn progress (turn / n²), constant plane
    """
    n = board.size
    grid = board.grid
    cur = board.current_player
    opp = PLAYER_2 if cur == PLAYER_1 else PLAYER_1

    ch0 = (grid == cur).astype(np.float32)
    ch1 = (grid == opp).astype(np.float32)
    ch2 = (grid == EMPTY).astype(np.float32)
    ch3 = _compute_open3_threats(grid, cur, n)
    ch4 = _compute_open3_threats(grid, opp, n)
    progress = min(board.turn / (n * n), 1.0)
    ch5 = np.full((n, n), progress, dtype=np.float32)

    return np.stack([ch0, ch1, ch2, ch3, ch4, ch5], axis=0)
