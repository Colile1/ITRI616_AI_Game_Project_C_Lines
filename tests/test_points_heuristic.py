"""test_points_heuristic.py
Purpose: Unit tests for PointsHeuristicAgent — objective alignment (convex
         score gain), contested-cell defence, legality, and determinism.
Author: Colile
"""

import numpy as np
import pytest

from src.game.env import GameEnv
from src.engine.board import setup_board
from src.game.encoding import build_legal_mask, state_to_tensor
from src.agents.points_heuristic_agent import PointsHeuristicAgent
from src.config import PLAYER_1, PLAYER_2


def _decide(board, defense_weight=1.0, seed=0):
    """Run the agent on a fixed board and return its chosen (row, col)."""
    agent = PointsHeuristicAgent(defense_weight=defense_weight,
                                 rng=np.random.default_rng(seed))
    agent.set_board(board)
    obs = state_to_tensor(board)
    mask = build_legal_mask(board)
    action = agent.select_action(obs, mask)
    return divmod(action, board.size)


# PH-01: set_board must be called before select_action
def test_PH01_requires_board():
    agent = PointsHeuristicAgent()
    with pytest.raises(RuntimeError):
        agent.select_action(np.zeros((10, 8, 8), dtype=np.float32),
                            np.ones(64, dtype=bool))


# PH-02: extends own run toward the higher-value length (convex objective)
def test_PH02_extends_for_points():
    n = 8
    board = setup_board(n)
    board.grid[0, 0] = PLAYER_1
    board.grid[0, 1] = PLAYER_1
    board.grid[0, 2] = PLAYER_1   # own open-3 (worth 0.25); extending to 4 → 1.0
    board.current_player = PLAYER_1
    assert _decide(board) == (0, 3)


# PH-03: contests the cell that is most valuable to the opponent (defence)
def test_PH03_contests_opponent_cell():
    n = 8
    board = setup_board(n)
    board.grid[7, 0] = PLAYER_2
    board.grid[7, 1] = PLAYER_2
    board.grid[7, 2] = PLAYER_2   # opponent open-3; (7,3) is the high-value cell
    board.current_player = PLAYER_1
    assert _decide(board, defense_weight=1.0) == (7, 3)


# PH-04: always returns a legal action across a full play-through
def test_PH04_always_legal():
    env = GameEnv(board_size=8)
    agent = PointsHeuristicAgent(rng=np.random.default_rng(1))
    obs = env.reset()
    for _ in range(64):
        mask = env.legal_mask()
        if not mask.any():
            break
        agent.set_board(env.board)
        action = agent.select_action(obs, mask)
        assert mask[action], f"chose illegal action {action}"
        obs, _, done, _ = env.step(action)
        if done:
            break


# PH-05: deterministic given an injected RNG seed
def test_PH05_deterministic():
    board = setup_board(8)
    board.current_player = PLAYER_1
    assert _decide(board, seed=123) == _decide(board, seed=123)
