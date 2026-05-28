"""Board dataclass and factory functions. Pure data — no I/O."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from src.config import EMPTY, PLAYER_1


@dataclass
class Board:
    grid: np.ndarray        # shape (size, size), dtype int8; 0=empty, 1=P1, 2=P2
    size: int
    current_player: int     # PLAYER_1 or PLAYER_2
    turn: int               # 0-indexed turn counter (increments after each placement)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Board):
            return NotImplemented
        return (
            self.size == other.size
            and self.current_player == other.current_player
            and self.turn == other.turn
            and np.array_equal(self.grid, other.grid)
        )


def setup_board(n: int) -> Board:
    """Return a fresh empty n×n board with Player 1 to move."""
    grid = np.zeros((n, n), dtype=np.int8)
    return Board(grid=grid, size=n, current_player=PLAYER_1, turn=0)


def clone_board(board: Board) -> Board:
    """Return a deep copy of *board*."""
    return Board(
        grid=board.grid.copy(),
        size=board.size,
        current_player=board.current_player,
        turn=board.turn,
    )
