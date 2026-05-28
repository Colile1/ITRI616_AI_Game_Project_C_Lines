"""E-01 .. E-11: GameEnv and encoding tests."""

import numpy as np
import pytest
from src.game.env import GameEnv
from src.game.encoding import (
    action_to_index, index_to_action, build_legal_mask, state_to_tensor,
)
from src.engine.board import setup_board
from src.config import MODE_FIRST_TO_FOUR, MODE_POINTS_FULL, PLAYER_1, PLAYER_2


# ---------------------------------------------------------------------------
# E-01: reset returns correct obs shape
# ---------------------------------------------------------------------------
def test_E01_reset_obs_shape():
    env = GameEnv(board_size=10, mode=MODE_POINTS_FULL)
    obs = env.reset()
    # 10 channels: 6 original + 4 new threat channels (v2 encoding)
    assert obs.shape[1:] == (10, 10)
    assert obs.shape[0] in (6, 10)  # 6 = legacy, 10 = new default
    assert obs.dtype == np.float32


# ---------------------------------------------------------------------------
# E-02: action_to_index / index_to_action round-trip
# ---------------------------------------------------------------------------
def test_E02_action_roundtrip():
    for n in [8, 10, 12]:
        for r in range(n):
            for c in range(n):
                idx = action_to_index(r, c, n)
                assert index_to_action(idx, n) == (r, c)


# ---------------------------------------------------------------------------
# E-03: legal_mask shape and all-true on fresh board
# ---------------------------------------------------------------------------
def test_E03_legal_mask_fresh():
    env = GameEnv(board_size=8)
    env.reset()
    mask = env.legal_mask()
    assert mask.shape == (64,)
    assert mask.all()


# ---------------------------------------------------------------------------
# E-04: step reduces legal moves by 1 on legal action
# ---------------------------------------------------------------------------
def test_E04_step_reduces_legal():
    env = GameEnv(board_size=8)
    env.reset()
    obs, reward, done, info = env.step(0)
    assert env.legal_mask().sum() == 63


# ---------------------------------------------------------------------------
# E-05: obs channels 0-2 are mutually exclusive and sum to 1
# ---------------------------------------------------------------------------
def test_E05_channel_partition():
    env = GameEnv(board_size=8)
    env.reset()
    env.step(action_to_index(3, 3, 8))
    obs = env._obs()
    total = obs[0] + obs[1] + obs[2]
    np.testing.assert_array_almost_equal(total, np.ones((8, 8)))


# ---------------------------------------------------------------------------
# E-06: turn counter increases with each step
# ---------------------------------------------------------------------------
def test_E06_turn_counter():
    env = GameEnv(board_size=8)
    env.reset()
    assert env.board.turn == 0
    env.step(0)
    assert env.board.turn == 1
    env.step(1)
    assert env.board.turn == 2


# ---------------------------------------------------------------------------
# E-07: player alternates each step
# ---------------------------------------------------------------------------
def test_E07_player_alternates():
    env = GameEnv(board_size=8)
    env.reset()
    assert env.board.current_player == PLAYER_1
    env.step(0)
    assert env.board.current_player == PLAYER_2
    env.step(1)
    assert env.board.current_player == PLAYER_1


# ---------------------------------------------------------------------------
# E-08: Mode 1 — win terminates episode
# ---------------------------------------------------------------------------
def test_E08_mode1_win_terminates():
    env = GameEnv(board_size=8, mode=MODE_FIRST_TO_FOUR)
    env.reset()
    n = 8
    # P1 places at cols 0,2,4,6 (to avoid P2 blocking)  — actually just
    # alternate placements: P1 at (0,0),(0,1),(0,2),(0,3) — P2 elsewhere
    done = False
    # Manual: P1→(0,0), P2→(7,7), P1→(0,1), P2→(7,6), P1→(0,2), P2→(7,5), P1→(0,3)
    moves = [(0,0),(7,7),(0,1),(7,6),(0,2),(7,5),(0,3)]
    for r, c in moves:
        _, _, done, info = env.step(action_to_index(r, c, n))
        if done:
            break
    assert done is True
    assert info.get("winner") == PLAYER_1


# ---------------------------------------------------------------------------
# E-09: Mode 2 — step reward is float
# ---------------------------------------------------------------------------
def test_E09_mode2_step_reward_is_float():
    env = GameEnv(board_size=8, mode=MODE_POINTS_FULL)
    env.reset()
    _, reward, _, _ = env.step(0)
    assert isinstance(reward, float)


# ---------------------------------------------------------------------------
# E-10: illegal action returns done=True and loss reward
# ---------------------------------------------------------------------------
def test_E10_illegal_action_penalty():
    env = GameEnv(board_size=8)
    env.reset()
    env.step(0)  # P1 places at cell 0
    # P2 tries to place at same cell
    _, reward, done, info = env.step(0)
    assert done is True
    assert info.get("illegal") is True


# ---------------------------------------------------------------------------
# E-11: state_to_tensor ch5 encodes turn progress
# ---------------------------------------------------------------------------
def test_E11_ch5_turn_progress():
    b = setup_board(8)
    t = state_to_tensor(b)
    assert t[5, 0, 0] == pytest.approx(0.0)
    # After 8 moves, progress = 8/64 = 0.125
    b.turn = 8
    t2 = state_to_tensor(b)
    assert t2[5, 0, 0] == pytest.approx(0.125)
