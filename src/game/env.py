"""C_lines Gym-style environment. Manages episode state for the RL loop."""

from __future__ import annotations
from typing import Any

import numpy as np

from src.config import (
    MODE_FIRST_TO_FOUR, MODE_POINTS_FULL, DEFAULT_MODE,
    DEFAULT_BOARD_SIZE, WIN_REWARD, LOSS_REWARD, DRAW_REWARD,
    STEP_REWARD_SCALE, PLAYER_1, PLAYER_2, STATE_CHANNELS,
)
from src.engine.board import Board, setup_board
from src.engine.rules import (
    apply_placement, check_terminal_mode1, check_terminal_mode2,
    is_legal_placement,
)
from src.engine.scoring import compute_scores
from src.game.encoding import state_to_tensor, build_legal_mask, index_to_action


class GameEnv:
    """Single-game environment.

    Observations are (6, n, n) float32 arrays from the perspective of the
    current player (always ch0 = "mine"). Call reset() before each episode.
    """

    def __init__(
        self,
        board_size: int = DEFAULT_BOARD_SIZE,
        mode: str = DEFAULT_MODE,
        state_channels: int = STATE_CHANNELS,
    ):
        self.board_size = board_size
        self.mode = mode
        self.state_channels = state_channels
        self._board: Board = setup_board(board_size)
        self._prev_scores: tuple[float, float] = (0.0, 0.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        self._board = setup_board(self.board_size)
        self._prev_scores = (0.0, 0.0)
        return self._obs()

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Apply action, return (obs, reward, done, info).

        Reward is from the perspective of the player who just acted.
        After step() the board's current_player has already been toggled.
        """
        row, col = index_to_action(action_idx, self.board_size)
        if not is_legal_placement(self._board, row, col):
            # Illegal move — penalise and end episode
            return self._obs(), LOSS_REWARD, True, {"illegal": True}

        acting_player = self._board.current_player
        self._board = apply_placement(self._board, row, col)

        # Terminal check
        if self.mode == MODE_FIRST_TO_FOUR:
            done, winner = check_terminal_mode1(self._board)
        else:
            done, winner = check_terminal_mode2(self._board)

        reward = self._compute_reward(done, winner, acting_player)
        info: dict[str, Any] = {"winner": winner, "done": done}
        return self._obs(), reward, done, info

    def legal_mask(self) -> np.ndarray:
        """Boolean mask (n*n,) — legal placements for current player."""
        return build_legal_mask(self._board, phase="placement")

    @property
    def board(self) -> Board:
        return self._board

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _obs(self) -> np.ndarray:
        return state_to_tensor(self._board, n_channels=self.state_channels)

    def _compute_reward(
        self, done: bool, winner: int | None, acting_player: int
    ) -> float:
        if done:
            if winner is None:
                return DRAW_REWARD
            return WIN_REWARD if winner == acting_player else LOSS_REWARD

        # Delta-score shaping — applied to BOTH modes.
        # In first_to_four, the score delta (run formation) still gives the agent
        # a meaningful gradient on every step rather than only at the terminal.
        p1_now, p2_now = compute_scores(self._board)
        p1_prev, p2_prev = self._prev_scores
        self._prev_scores = (p1_now, p2_now)
        if acting_player == PLAYER_1:
            return (p1_now - p1_prev - (p2_now - p2_prev)) * STEP_REWARD_SCALE
        else:
            return (p2_now - p2_prev - (p1_now - p1_prev)) * STEP_REWARD_SCALE
