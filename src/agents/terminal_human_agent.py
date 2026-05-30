"""Human agent that takes moves from stdin — for CLI training sessions.

Usage in training:
    The training loop creates one of these and uses it as an opponent source.
    The board is printed to the terminal and the human types a move.

Move input formats (all 0-indexed):
    "3 4"   → row 3, col 4
    "28"    → flat index  (row = 28 // board_size, col = 28 % board_size)
    "q"     → signal training to switch to auto mode (writes mode.txt)
"""

from __future__ import annotations
import numpy as np
from src.agents.base_agent import BaseAgent

_SYMBOLS = {0: ".", 1: "X", 2: "O"}


def _print_board(grid: np.ndarray, board_size: int) -> None:
    header = "   " + "  ".join(f"{c:2d}" for c in range(board_size))
    print(header)
    print("   " + "---" * board_size)
    for r in range(board_size):
        row_str = " | ".join(_SYMBOLS.get(int(grid[r, c]), "?") for c in range(board_size))
        print(f"{r:2d} | {row_str} |")
    print()


class TerminalHumanAgent(BaseAgent):
    """Reads moves from stdin and prints board state for the human to see.

    Requires access to the board grid to display the position.  The grid is
    passed in via set_board() before each select_action call; train.py does
    this automatically when using the human source.
    """

    def __init__(self, board_size: int, switch_signal_path=None):
        self.board_size = board_size
        self._grid: np.ndarray | None = None
        self._switch_signal_path = switch_signal_path  # Path to mode.txt

    def set_board(self, grid: np.ndarray) -> None:
        self._grid = grid

    def select_action(self, obs: np.ndarray, legal_mask: np.ndarray) -> int:
        n = self.board_size
        if self._grid is not None:
            _print_board(self._grid, n)

        legal_indices = np.where(legal_mask)[0]
        print(f"Your turn (X). Legal moves: {len(legal_indices)} cells available.")
        print("Enter move as 'row col' or flat index, or 'q' to switch to auto mode.")

        while True:
            try:
                raw = input("Move > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                # Non-interactive context — pick first legal move
                return int(legal_indices[0])

            if raw == "q" and self._switch_signal_path is not None:
                self._switch_signal_path.write_text("auto")
                print("[mode] Switching to auto — training continues without you.")
                return int(legal_indices[0])  # complete current move then switch

            parts = raw.split()
            try:
                if len(parts) == 2:
                    r, c = int(parts[0]), int(parts[1])
                    idx = r * n + c
                elif len(parts) == 1:
                    idx = int(parts[0])
                    r, c = idx // n, idx % n
                else:
                    raise ValueError
            except (ValueError, IndexError):
                print(f"  Bad input. Use 'row col' or a single number 0–{n*n-1}.")
                continue

            if not (0 <= idx < n * n) or not legal_mask[idx]:
                print(f"  ({r},{c}) is not a legal move. Try again.")
                continue

            return idx
