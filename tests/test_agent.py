"""A-01 .. A-08: agent tests (A-05..A-08 added in Phase 6 for DQNAgent)."""

import numpy as np
import pytest
from src.game.env import GameEnv
from src.agents.random_agent import RandomAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.engine.board import setup_board
from src.engine.rules import apply_placement
from src.game.encoding import action_to_index
from src.config import MODE_FIRST_TO_FOUR, PLAYER_1, PLAYER_2


# ---------------------------------------------------------------------------
# A-01: RandomAgent always returns a legal action
# ---------------------------------------------------------------------------
def test_A01_random_always_legal():
    env = GameEnv(board_size=8)
    agent = RandomAgent()
    obs = env.reset()
    for _ in range(20):
        mask = env.legal_mask()
        if not mask.any():
            break
        action = agent.select_action(obs, mask)
        assert mask[action], f"RandomAgent returned illegal action {action}"
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()


# ---------------------------------------------------------------------------
# A-02: RandomAgent respects legal mask — never chooses masked-out cell
# ---------------------------------------------------------------------------
def test_A02_random_respects_mask():
    rng = np.random.default_rng(42)
    agent = RandomAgent(rng=rng)
    env = GameEnv(board_size=8)
    obs = env.reset()
    # Fill most of the board manually
    for idx in range(60):
        mask = env.legal_mask()
        if not mask.any():
            break
        action = agent.select_action(obs, mask)
        obs, _, done, _ = env.step(action)
        if done:
            break
    # With a nearly-full board, agent should still pick legal cell
    mask = env.legal_mask()
    if mask.any():
        action = agent.select_action(obs, mask)
        assert mask[action]


# ---------------------------------------------------------------------------
# A-03: HeuristicAgent blocks opponent's 4-in-a-row
# ---------------------------------------------------------------------------
def test_A03_heuristic_blocks_four():
    n = 8
    board = setup_board(n)
    # P2 has 3-in-a-row at cols 0,1,2 row 7 — P1 to move, should block col 3
    board.grid[7, 0] = PLAYER_2
    board.grid[7, 1] = PLAYER_2
    board.grid[7, 2] = PLAYER_2
    board.current_player = PLAYER_1

    agent = HeuristicAgent(rng=np.random.default_rng(0))
    agent.set_board(board)

    from src.game.encoding import build_legal_mask, state_to_tensor
    obs = state_to_tensor(board)
    mask = build_legal_mask(board)
    action = agent.select_action(obs, mask)
    r, c = divmod(action, n)
    # Blocking move must be (7,3) — the cell that would complete P2's 4-in-a-row
    assert (r, c) == (7, 3), f"Expected block at (7,3), got ({r},{c})"


# ---------------------------------------------------------------------------
# A-04: HeuristicAgent wins when it can
# ---------------------------------------------------------------------------
def test_A04_heuristic_takes_win():
    n = 8
    board = setup_board(n)
    # P1 has 3-in-a-row at row 0, cols 0,1,2; col 3 is empty
    board.grid[0, 0] = PLAYER_1
    board.grid[0, 1] = PLAYER_1
    board.grid[0, 2] = PLAYER_1
    board.current_player = PLAYER_1

    agent = HeuristicAgent(rng=np.random.default_rng(0))
    agent.set_board(board)

    from src.game.encoding import build_legal_mask, state_to_tensor
    obs = state_to_tensor(board)
    mask = build_legal_mask(board)
    action = agent.select_action(obs, mask)
    r, c = divmod(action, n)
    assert (r, c) == (0, 3), f"Expected win at (0,3), got ({r},{c})"
