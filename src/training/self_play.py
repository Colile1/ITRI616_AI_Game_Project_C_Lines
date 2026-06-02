"""Self-play episode execution and curriculum opponent selection."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from src.agents.base_agent import BaseAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.game.env import GameEnv
from src.config import MODE_POINTS_FULL, GAMMA, N_STEP_RETURNS


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    legal_mask_next: np.ndarray
    gamma_n: float = field(default=GAMMA)   # bootstrap discount = GAMMA^n_actual


def _make_nstep_transitions(
    full_traj: list[tuple],
    player_idx: int,
    n: int,
    gamma: float,
) -> list[Transition]:
    """Build per-player Transition list with n-step returns.

    full_traj entries: (state, action, reward, next_state, done, legal_mask_next, player_idx)

    For each of this player's moves at position t in the trajectory, accumulate:
        G = r_t - γ·r_{t+1} + γ²·r_{t+2} - γ³·r_{t+3} + ...  (negamax alternating sign)
    and bootstrap from s_{t+n} with discount γ^n (negated by DQN update, so sign stays correct
    when n is odd; for even n the bootstrap state is from the same player's perspective,
    so we negate the accumulated reward to compensate — see note below).
    """
    T = len(full_traj)
    transitions: list[Transition] = []

    for t in range(T):
        _, _, _, _, _, _, pi = full_traj[t]
        if pi != player_idx:
            continue

        g = 0.0
        last_done = False
        steps = 0
        for k in range(n):
            ti = t + k
            if ti >= T:
                break
            _, _, rk, nsk, dk, msk, _ = full_traj[ti]
            sign = (-1) ** k  # negamax: reward alternates sign each step
            g += sign * (gamma ** k) * rk
            steps += 1
            if dk:
                last_done = True
                break

        # Bootstrap state: s_{t+steps}, from the perspective of the player
        # who acts at that step.  The DQN update always applies -γ * max Q(s'),
        # which is correct when s' is from the opponent's perspective (odd steps
        # from this player's move).  When steps is even, s' would be from our own
        # perspective; in that case negate g so the subtraction gives the right sign.
        boot_idx = t + steps
        if last_done or boot_idx >= T:
            _, _, _, last_ns, _, last_mask, _ = full_traj[t + steps - 1]
            boot_state = last_ns
            boot_mask  = last_mask
            boot_done  = True
            gamma_n    = 0.0
        else:
            boot_s, _, _, _, _, boot_mask_raw, _ = full_traj[boot_idx]
            boot_state = boot_s
            boot_mask  = boot_mask_raw
            boot_done  = False
            gamma_n    = gamma ** steps
            # If steps is even the bootstrap is from our own perspective; negate g
            # so that the DQN's implicit negation (target = g - γ^n * max Q) is correct.
            if steps % 2 == 0:
                g = -g

        s, a, _, _, _, _, _ = full_traj[t]
        transitions.append(Transition(
            state=s,
            action=a,
            reward=g,
            next_state=boot_state,
            done=boot_done,
            legal_mask_next=boot_mask,
            gamma_n=gamma_n,
        ))

    return transitions


def play_episode(
    agent1: BaseAgent,
    agent2: BaseAgent,
    env: GameEnv,
    n_step: int = N_STEP_RETURNS,
    gamma: float = GAMMA,
) -> tuple[list[Transition], list[Transition], Optional[int]]:
    """Run one full episode and return (transitions_p1, transitions_p2, winner).

    Transitions are from the perspective of the acting agent at each step.
    With n_step > 1, each transition's reward is an n-step accumulated return
    and gamma_n carries the correct bootstrap discount.
    winner: 1, 2, or None (draw / illegal)
    """
    obs = env.reset()
    agents = [agent1, agent2]
    agent_idx = 0  # P1 starts

    # Full interleaved trajectory for n-step computation
    full_traj: list[tuple] = []
    done = False
    winner = None

    while not done:
        agent = agents[agent_idx]
        mask = env.legal_mask()
        if not mask.any():
            break

        if isinstance(agent, HeuristicAgent):
            agent.set_board(env.board)
        elif hasattr(agent, "set_board"):
            agent.set_board(env.board)

        current_obs = obs
        action = agent.select_action(obs, mask)
        next_obs, reward, done, info = env.step(action)
        next_mask = env.legal_mask() if not done else np.zeros(env.board_size ** 2, dtype=bool)

        full_traj.append((current_obs, action, reward, next_obs, done, next_mask, agent_idx))

        winner = info.get("winner")
        obs = next_obs
        agent_idx = 1 - agent_idx

    if n_step <= 1:
        # Fast path: 1-step transitions (backward compatible)
        t1 = [
            Transition(s, a, r, ns, d, m, gamma)
            for s, a, r, ns, d, m, pi in full_traj if pi == 0
        ]
        t2 = [
            Transition(s, a, r, ns, d, m, gamma)
            for s, a, r, ns, d, m, pi in full_traj if pi == 1
        ]
    else:
        t1 = _make_nstep_transitions(full_traj, 0, n_step, gamma)
        t2 = _make_nstep_transitions(full_traj, 1, n_step, gamma)

    return t1, t2, winner


def linear_epsilon(eps_start: float, eps_end: float, game_idx: int, decay_games: int) -> float:
    frac = min(game_idx / max(decay_games, 1), 1.0)
    return eps_start + frac * (eps_end - eps_start)
