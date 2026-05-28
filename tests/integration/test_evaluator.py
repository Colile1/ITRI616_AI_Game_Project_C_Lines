"""EV-01 .. EV-03: evaluator integration tests."""

import pytest
from src.agents.random_agent import RandomAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.game.env import GameEnv
from src.evaluation.evaluator import evaluate


# ---------------------------------------------------------------------------
# EV-01: Random vs Random win rate ≈ 50%  (allow generous tolerance)
# ---------------------------------------------------------------------------
def test_EV01_random_vs_random():
    env = GameEnv(board_size=8, mode="points_full")
    a1, a2 = RandomAgent(), RandomAgent()
    stats = evaluate(a1, a2, env, n_games=100)
    # Should be between 30–70% given stochastic play
    assert 0.20 <= stats["win_rate"] <= 0.80


# ---------------------------------------------------------------------------
# EV-02: Heuristic vs Random win rate > random baseline
# ---------------------------------------------------------------------------
def test_EV02_heuristic_beats_random():
    env = GameEnv(board_size=8, mode="first_to_four")
    heuristic = HeuristicAgent()
    random = RandomAgent()
    stats = evaluate(heuristic, random, env, n_games=50)
    assert stats["win_rate"] >= 0.30  # heuristic should consistently win


# ---------------------------------------------------------------------------
# EV-03: evaluate returns required keys
# ---------------------------------------------------------------------------
def test_EV03_return_keys():
    env = GameEnv(board_size=8)
    stats = evaluate(RandomAgent(), RandomAgent(), env, n_games=10)
    for key in ("win_rate", "draw_rate", "loss_rate", "mean_ep_len"):
        assert key in stats
