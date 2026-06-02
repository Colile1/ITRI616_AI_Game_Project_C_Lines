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


def count_open_threats(board: Board, player: int, run_length: int) -> int:
    """Count runs of exactly run_length for player with at least one open adjacent end.

    Used for first_to_four reward shaping — open-3s are the critical threats
    because one more piece converts them to a winning 4-in-a-row.
    """
    n = board.size
    grid = board.grid
    total = 0
    for dr, dc in _DIRECTIONS:
        for r in range(n):
            for c in range(n):
                if grid[r, c] != player:
                    continue
                # Skip if not the start of a run in this direction
                pr, pc = r - dr, c - dc
                if 0 <= pr < n and 0 <= pc < n and grid[pr, pc] == player:
                    continue
                # Count run length forward
                length = 0
                nr, nc = r, c
                while 0 <= nr < n and 0 <= nc < n and grid[nr, nc] == player:
                    length += 1
                    nr += dr
                    nc += dc
                if length != run_length:
                    continue
                # Check open ends: back (before start) and front (after end)
                back_free = (
                    0 <= r - dr < n and 0 <= c - dc < n
                    and grid[r - dr, c - dc] == 0
                )
                front_free = (
                    0 <= nr < n and 0 <= nc < n
                    and grid[nr, nc] == 0
                )
                if back_free or front_free:
                    total += 1
    return total


def compute_threats(board: Board) -> tuple[int, int]:
    """Return (p1_open3, p2_open3) — open-3 threat counts per player."""
    p1 = count_open_threats(board, 1, 3)
    p2 = count_open_threats(board, 2, 3)
    return p1, p2


def count_double_open_threats(board: Board, player: int, run_length: int) -> int:
    """Count runs of exactly run_length with BOTH ends free — a forced-win threat.

    Stronger than count_open_threats (at-least-one-free-end): a double-open run
    cannot be blocked in one move, making it a guaranteed win in FTF mode.
    """
    n = board.size
    grid = board.grid
    total = 0
    for dr, dc in _DIRECTIONS:
        for r in range(n):
            for c in range(n):
                if grid[r, c] != player:
                    continue
                pr, pc = r - dr, c - dc
                if 0 <= pr < n and 0 <= pc < n and grid[pr, pc] == player:
                    continue
                length = 0
                nr, nc = r, c
                while 0 <= nr < n and 0 <= nc < n and grid[nr, nc] == player:
                    length += 1
                    nr += dr
                    nc += dc
                if length != run_length:
                    continue
                back_free = (
                    0 <= r - dr < n and 0 <= c - dc < n
                    and grid[r - dr, c - dc] == 0
                )
                front_free = (
                    0 <= nr < n and 0 <= nc < n
                    and grid[nr, nc] == 0
                )
                if back_free and front_free:
                    total += 1
    return total


def compute_double_threats(board: Board) -> tuple[int, int]:
    """Return (p1_double_open3, p2_double_open3) — forced-win threat counts."""
    p1 = count_double_open_threats(board, 1, 3)
    p2 = count_double_open_threats(board, 2, 3)
    return p1, p2
