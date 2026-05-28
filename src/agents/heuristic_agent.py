"""Heuristic agent with priority: complete-own-4 > block-opp-4 > extend-own > random."""

from __future__ import annotations
import numpy as np

from src.agents.base_agent import BaseAgent
from src.engine.board import Board
from src.engine.rules import apply_placement, check_terminal_mode1
from src.game.encoding import index_to_action, action_to_index
from src.config import PLAYER_1, PLAYER_2


class HeuristicAgent(BaseAgent):
    """Needs access to the raw board; obs is ignored (first arg kept for API compatibility)."""

    def __init__(self, rng: np.random.Generator | None = None):
        self._rng = rng or np.random.default_rng()
        self._board: Board | None = None

    def set_board(self, board: Board) -> None:
        """Provide the current board so the agent can reason about it."""
        self._board = board

    def select_action(self, obs: np.ndarray, legal_mask: np.ndarray) -> int:
        if self._board is None:
            raise RuntimeError("HeuristicAgent.set_board() must be called before select_action()")
        board = self._board
        legal_indices = list(np.where(legal_mask)[0])

        # Priority 1: complete own 4-in-a-row
        win_move = self._find_winning_move(board, board.current_player, legal_indices)
        if win_move is not None:
            return win_move

        # Priority 2: block opponent's 4-in-a-row
        opp = PLAYER_2 if board.current_player == PLAYER_1 else PLAYER_1
        block_move = self._find_winning_move(board, opp, legal_indices)
        if block_move is not None:
            return block_move

        # Priority 3: extend own longest run (pick the move that maximises run length)
        best_extend = self._find_extend_move(board, board.current_player, legal_indices)
        if best_extend is not None:
            return best_extend

        # Fallback: random legal
        return int(self._rng.choice(legal_indices))

    def _find_winning_move(
        self, board: Board, player: int, legal_indices: list[int]
    ) -> int | None:
        """Return an index that gives *player* a 4-in-a-row, or None."""
        n = board.size
        # Temporarily act as if *player* is to move
        from src.engine.board import clone_board
        fake_board = clone_board(board)
        fake_board.current_player = player

        for idx in legal_indices:
            r, c = index_to_action(idx, n)
            candidate = apply_placement(fake_board, r, c)
            done, winner = check_terminal_mode1(candidate)
            if done and winner == player:
                return idx
        return None

    def _find_extend_move(
        self, board: Board, player: int, legal_indices: list[int]
    ) -> int | None:
        """Return the move that most extends own longest run, or None."""
        n = board.size
        best_idx = None
        best_len = -1
        for idx in self._rng.permutation(legal_indices):
            r, c = index_to_action(idx, n)
            candidate = apply_placement(board, r, c)
            length = self._max_run(candidate.grid, player, n)
            if length > best_len:
                best_len = length
                best_idx = int(idx)
        return best_idx

    @staticmethod
    def _max_run(grid: np.ndarray, player: int, n: int) -> int:
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        best = 0
        for dr, dc in directions:
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
                    best = max(best, length)
        return best
