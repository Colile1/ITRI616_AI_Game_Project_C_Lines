"""Line-counting and score computation for C_lines. Pure functions."""

from __future__ import annotations
import numpy as np

from src.config import SCORE_FOR_LENGTH, MIN_SCORING_LENGTH, MAX_SCORING_LENGTH
from src.engine.board import Board

# Directions: (row_delta, col_delta)
_DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def score_for_length(length: int) -> float:
    """Return the point value for a run of *length* (capped at MAX_SCORING_LENGTH)."""
    capped = min(length, MAX_SCORING_LENGTH)
    if capped < MIN_SCORING_LENGTH:
        return 0.0
    return SCORE_FOR_LENGTH[capped]


def score_board(grid: np.ndarray, player: int, n: int) -> float:
    """Sum points for all maximal runs belonging to *player* on *grid*."""
    total = 0.0
    for dr, dc in _DIRECTIONS:
        total += _score_direction(grid, player, n, dr, dc)
    return total


def _score_direction(grid: np.ndarray, player: int, n: int, dr: int, dc: int) -> float:
    """Score all maximal runs in a single direction across the full grid."""
    visited = np.zeros((n, n), dtype=bool)
    total = 0.0
    for r in range(n):
        for c in range(n):
            if visited[r, c] or grid[r, c] != player:
                continue
            # Walk backwards to find start of run
            pr, pc = r - dr, c - dc
            if 0 <= pr < n and 0 <= pc < n and grid[pr, pc] == player:
                visited[r, c] = True
                continue  # not the start of this run
            # Count run length forward
            length = 0
            nr, nc = r, c
            while 0 <= nr < n and 0 <= nc < n and grid[nr, nc] == player:
                visited[nr, nc] = True
                length += 1
                nr += dr
                nc += dc
            total += score_for_length(length)
    return total


def compute_scores(board: Board) -> tuple[float, float]:
    """Return (p1_score, p2_score) for the current board state."""
    p1 = score_board(board.grid, 1, board.size)
    p2 = score_board(board.grid, 2, board.size)
    return p1, p2
