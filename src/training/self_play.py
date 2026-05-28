"""Self-play episode execution and curriculum opponent selection."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from src.agents.base_agent import BaseAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.game.env import GameEnv
from src.config import MODE_POINTS_FULL


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    legal_mask_next: np.ndarray


def play_episode(
    agent1: BaseAgent,
    agent2: BaseAgent,
    env: GameEnv,
) -> tuple[list[Transition], list[Transition], Optional[int]]:
    """Run one full episode and return (transitions_p1, transitions_p2, winner).

    Transitions are from the perspective of the acting agent at each step.
    winner: 1, 2, or None (draw / illegal)
    """
    obs = env.reset()
    agents = [agent1, agent2]
    agent_idx = 0  # P1 starts
    transitions_p1: list[Transition] = []
    transitions_p2: list[Transition] = []
    buffers = [transitions_p1, transitions_p2]

    prev_obs = obs
    done = False
    winner = None

    while not done:
        agent = agents[agent_idx]
        mask = env.legal_mask()
        if not mask.any():
            break

        # Agents that need board reference (HeuristicAgent, AlphaBetaAgent, MCTSAgent)
        if isinstance(agent, HeuristicAgent):
            agent.set_board(env.board)
        elif hasattr(agent, "set_board"):
            agent.set_board(env.board)

        action = agent.select_action(obs, mask)
        next_obs, reward, done, info = env.step(action)
        next_mask = env.legal_mask() if not done else np.zeros(env.board_size ** 2, dtype=bool)

        t = Transition(
            state=prev_obs if agent_idx == 0 else obs,
            action=action,
            reward=reward,
            next_state=next_obs,
            done=done,
            legal_mask_next=next_mask,
        )
        buffers[agent_idx].append(t)

        winner = info.get("winner")
        prev_obs = next_obs
        obs = next_obs
        agent_idx = 1 - agent_idx

    return transitions_p1, transitions_p2, winner


def linear_epsilon(eps_start: float, eps_end: float, game_idx: int, decay_games: int) -> float:
    frac = min(game_idx / max(decay_games, 1), 1.0)
    return eps_start + frac * (eps_end - eps_start)
