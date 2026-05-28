"""Agent evaluation utilities."""

from __future__ import annotations
from typing import Optional

from src.agents.base_agent import BaseAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.game.env import GameEnv
from src.training.self_play import play_episode


def evaluate(
    agent: BaseAgent,
    opponent: BaseAgent,
    env: GameEnv,
    n_games: int = 200,
) -> dict:
    """Play *n_games* and return aggregate stats.

    agent always plays as P1 in the first half, P2 in the second half.
    """
    wins = 0
    draws = 0
    total_len = 0

    for g in range(n_games):
        if g < n_games // 2:
            _, _, winner = play_episode(agent, opponent, env)
            agent_player = 1
        else:
            _, _, winner = play_episode(opponent, agent, env)
            agent_player = 2

        ep_len = env.board.turn
        total_len += ep_len

        if winner is None:
            draws += 1
        elif winner == agent_player:
            wins += 1

    return {
        "win_rate": wins / n_games,
        "draw_rate": draws / n_games,
        "loss_rate": (n_games - wins - draws) / n_games,
        "mean_ep_len": total_len / n_games,
    }
