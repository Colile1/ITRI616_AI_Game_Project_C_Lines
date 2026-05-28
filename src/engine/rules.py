"""Game rules: placement, removal, terminal detection, tie-break. Pure functions."""

from __future__ import annotations
from typing import Optional

from src.config import (
    EMPTY, PLAYER_1, PLAYER_2,
    MODE_FIRST_TO_FOUR, MODE_POINTS_FULL,
    TIEBREAK_MAX_ROUNDS,
)
from src.engine.board import Board, clone_board
from src.engine.scoring import compute_scores, score_board


def _other(player: int) -> int:
    return PLAYER_2 if player == PLAYER_1 else PLAYER_1


# ---------------------------------------------------------------------------
# Placement / removal
# ---------------------------------------------------------------------------

def is_legal_placement(board: Board, row: int, col: int) -> bool:
    """True iff (row, col) is within bounds and currently empty."""
    n = board.size
    return 0 <= row < n and 0 <= col < n and board.grid[row, col] == EMPTY


def apply_placement(board: Board, row: int, col: int) -> Board:
    """Return new board with current_player's piece placed at (row, col)."""
    new = clone_board(board)
    new.grid[row, col] = board.current_player
    new.current_player = _other(board.current_player)
    new.turn = board.turn + 1
    return new


def apply_removal(board: Board, row: int, col: int) -> Board:
    """Return new board with piece at (row, col) removed (tie-break use)."""
    new = clone_board(board)
    new.grid[row, col] = EMPTY
    return new


def legal_placements(board: Board) -> list[tuple[int, int]]:
    """All empty cells as (row, col) pairs."""
    cells = []
    n = board.size
    for r in range(n):
        for c in range(n):
            if board.grid[r, c] == EMPTY:
                cells.append((r, c))
    return cells


def legal_removals(board: Board, player: int) -> list[tuple[int, int]]:
    """All cells occupied by *player*."""
    cells = []
    n = board.size
    for r in range(n):
        for c in range(n):
            if board.grid[r, c] == player:
                cells.append((r, c))
    return cells


# ---------------------------------------------------------------------------
# Terminal detection
# ---------------------------------------------------------------------------

def _has_run_of_length(grid, player: int, n: int, length: int) -> bool:
    """True iff *player* has any run >= *length* in any direction."""
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
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


def check_terminal_mode1(board: Board) -> tuple[bool, Optional[int]]:
    """Mode 1 terminal check.

    Returns (done, winner) where winner is PLAYER_1, PLAYER_2, or None (draw).
    A player wins immediately upon forming a run of length >= 4.
    Full board with no winner = draw.
    """
    n = board.size
    # Check the player who just moved (previous player) for a win
    last_player = _other(board.current_player)
    if _has_run_of_length(board.grid, last_player, n, 4):
        return True, last_player
    # Check if board is full
    if not any(board.grid[r, c] == EMPTY for r in range(n) for c in range(n)):
        return True, None
    return False, None


def check_terminal_mode2(board: Board) -> tuple[bool, Optional[int]]:
    """Mode 2 terminal check.

    Returns (done, winner). Terminal only when board is full.
    Resolves ties via run_tiebreak.
    """
    n = board.size
    if any(board.grid[r, c] == EMPTY for r in range(n) for c in range(n)):
        return False, None
    p1_score, p2_score = compute_scores(board)
    if p1_score > p2_score:
        return True, PLAYER_1
    elif p2_score > p1_score:
        return True, PLAYER_2
    else:
        winner = run_tiebreak(board, p1_score, p2_score)
        return True, winner


# ---------------------------------------------------------------------------
# Tie-break
# ---------------------------------------------------------------------------

def run_tiebreak(board: Board, p1_score: float, p2_score: float) -> Optional[int]:
    """Execute the 3-round mutual-removal tie-break.

    Each round: P1 removes one P2 piece, P2 removes one P1 piece.
    After each round recompute scores; if different, return winner.
    After 3 rounds still tied → return None (draw).
    """
    current = clone_board(board)

    for _ in range(TIEBREAK_MAX_ROUNDS):
        p2_pieces = legal_removals(current, PLAYER_2)
        p1_pieces = legal_removals(current, PLAYER_1)

        if not p2_pieces or not p1_pieces:
            break  # no pieces to remove → draw

        # P1 removes one P2 piece (pick first — deterministic for engine)
        current = apply_removal(current, *p2_pieces[0])
        # P2 removes one P1 piece
        current = apply_removal(current, *p1_pieces[0])

        p1_score, p2_score = compute_scores(current)
        if p1_score > p2_score:
            return PLAYER_1
        elif p2_score > p1_score:
            return PLAYER_2

    return None  # draw
