"""points_heuristic_agent.py
Purpose: Greedy one-ply heuristic opponent matched to the points-until-full
         objective. Scores each legal move by the convex line-score it gains
         across all four directions, minus the score it denies the opponent on
         that same (contested) cell. No search, no learning.
Author: Colile
"""

from __future__ import annotations
import numpy as np

from src.agents.base_agent import BaseAgent
from src.engine.board import Board
from src.engine.scoring import score_board
from src.game.encoding import index_to_action
from src.config import PLAYER_1, PLAYER_2

# How heavily denying the opponent a cell counts relative to scoring it yourself.
# 1.0 treats a contested cell symmetrically: in points-until-full the board fills
# completely, so a cell valuable to the opponent is value you lose by not taking it.
DEFAULT_DEFENSE_WEIGHT = 1.0

# Moves whose value is within this tolerance of the best are treated as ties.
_TIE_EPSILON = 1e-9


def _other(player: int) -> int:
    """Return the opposing player id."""
    return PLAYER_2 if player == PLAYER_1 else PLAYER_1


def marginal_score(grid: np.ndarray, row: int, col: int, player: int,
                   n: int, base: float) -> float:
    """marginal_score
    Purpose: Score gained by *player* from placing one stone at (row, col).
    Inputs:  grid (n×n int array), row/col (int cell), player (int id),
             n (int board size), base (float — player's score before the move).
    Output:  float — player's score after the hypothetical move minus base.
    """
    trial = grid.copy()
    trial[row, col] = player
    return score_board(trial, player, n) - base


class PointsHeuristicAgent(BaseAgent):
    """Objective-aligned greedy opponent for points-until-full mode.

    Needs the raw board via set_board() before select_action(), the same
    pattern as HeuristicAgent and AlphaBetaAgent. The obs argument is ignored
    and kept only for BaseAgent API compatibility.
    """

    def __init__(self, defense_weight: float = DEFAULT_DEFENSE_WEIGHT,
                 rng: np.random.Generator | None = None):
        """Store the defence weight and an injectable RNG for tie-breaking."""
        self._defense_weight = defense_weight
        self._rng = rng or np.random.default_rng()
        self._board: Board | None = None

    def set_board(self, board: Board) -> None:
        """Provide the current board so the agent can reason about it."""
        self._board = board

    def select_action(self, obs: np.ndarray, legal_mask: np.ndarray) -> int:
        """select_action
        Purpose: Pick the legal cell that maximises own score gain plus the
                 weighted opponent gain it denies.
        Inputs:  obs (ignored), legal_mask ((n*n,) bool array of legal cells).
        Output:  int — a flat, legal action index.
        """
        if self._board is None:
            raise RuntimeError(
                "PointsHeuristicAgent.set_board() must be called before select_action()"
            )

        board = self._board
        n = board.size
        legal_indices = list(np.where(legal_mask)[0])
        if not legal_indices:
            raise RuntimeError("PointsHeuristicAgent: no legal moves available")

        me = board.current_player
        opp = _other(me)
        base_me = score_board(board.grid, me, n)
        base_opp = score_board(board.grid, opp, n)

        best_value = -np.inf
        best_indices: list[int] = []
        for idx in legal_indices:
            row, col = index_to_action(int(idx), n)
            own_gain = marginal_score(board.grid, row, col, me, n, base_me)
            denied = marginal_score(board.grid, row, col, opp, n, base_opp)
            value = own_gain + self._defense_weight * denied

            if value > best_value + _TIE_EPSILON:
                best_value = value
                best_indices = [int(idx)]
            elif abs(value - best_value) <= _TIE_EPSILON:
                best_indices.append(int(idx))

        # Random tie-break (deterministic when an RNG is injected) avoids a
        # positional bias toward low-index cells.
        return int(self._rng.choice(best_indices))
