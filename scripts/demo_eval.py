"""Live-demo helper: load a trained C_lines snapshot and report its win rate
against Random, Heuristic, and (optionally) Alpha-Beta depth-4.

Usage:
    python -m scripts.demo_eval --gen 10 --games 50
    python -m scripts.demo_eval --gen 10 --games 30 --alphabeta

Prints numeric-first lines so the result is obvious on a projector.
"""

from __future__ import annotations
import argparse

from src.config import MODE_POINTS_FULL, STATE_CHANNELS_V2, NETWORK_ARCH, MODELS_DIR
from src.agents.dqn_agent import DQNAgent
from src.agents.random_agent import RandomAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.game.env import GameEnv
from src.evaluation.evaluator import evaluate
import torch


def _load_agent(board_size: int, gen: int, run_id: str) -> DQNAgent:
    weights = MODELS_DIR / f"size_{board_size:02d}" / run_id / f"gen_{gen:03d}" / "weights.pt"
    agent = DQNAgent(board_size, eval_only=True,
                     in_channels=STATE_CHANNELS_V2, network_arch=NETWORK_ARCH)
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    agent.load_state_dict(sd)
    agent.set_epsilon(0.0)  # greedy play for the demo
    return agent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--gen", type=int, default=10)
    p.add_argument("--run-id", type=str, default="run_pts_002")
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--alphabeta", action="store_true", default=False)
    args = p.parse_args()

    agent = _load_agent(args.size, args.gen, args.run_id)
    env = GameEnv(board_size=args.size, mode=MODE_POINTS_FULL)

    print(f"Snapshot: gen_{args.gen:03d}  (run {args.run_id}, {args.size}x{args.size}, points-full)")

    r = evaluate(agent, RandomAgent(), env, args.games)
    print(f"Win rate vs Random:    {r['win_rate']:.0%}   ({args.games} games)")

    h = evaluate(agent, HeuristicAgent(), env, args.games)
    print(f"Win rate vs Heuristic: {h['win_rate']:.0%}   ({args.games} games)")

    if args.alphabeta:
        from src.agents.alphabeta_agent import AlphaBetaAgent
        ab = AlphaBetaAgent(board_size=args.size, depth=4)
        n = min(args.games, 16)  # alpha-beta is slow; keep the live demo snappy
        a = evaluate(agent, ab, env, n)
        print(f"Win rate vs AlphaBeta d4: {a['win_rate']:.0%}   ({n} games)")


if __name__ == "__main__":
    main()
