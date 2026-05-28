"""Training CLI.

Usage:
    python -m src.training.train --games 10000 --size 8
    python -m src.training.train --games 10000 --size 9 --run-id run_001

Each invocation auto-increments the run ID unless --run-id is specified.
Outputs go to:
    results/size_NN/run_NNN/training_log.csv
    results/size_NN/run_NNN/figures/
    models/size_NN/run_NNN/gen_NNN/
"""

from __future__ import annotations
import argparse
import csv
import uuid
from pathlib import Path

import numpy as np

import src.config as _cfg
from src.agents.dqn_agent import DQNAgent
from src.agents.random_agent import RandomAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.config import (
    DEFAULT_BOARD_SIZE, DEFAULT_MODE,
    BATCH_SIZE, REPLAY_CAPACITY,
    EPS_START, EPS_END, EPS_DECAY_GAMES,
    TARGET_SYNC_STEPS, EVAL_INTERVAL, EVAL_GAMES,
    SNAPSHOT_INTERVAL, MAX_POOL_SIZE, WARMUP_GAMES,
    WARMUP_HEURISTIC_PROB, SELF_PLAY_MIX_PROB,
    RECENT_POOL_BIAS, RECENT_POOL_TOP_N,
    GRADIENT_STEPS_PER_GAME, TRAINING_GAMES,
    LR, LR_DECAY_MILESTONES, LR_DECAY_FACTOR,
)
from src.game.env import GameEnv
from src.training.replay_buffer import ReplayBuffer
from src.training.self_play import play_episode, linear_epsilon
from src.training.snapshot import freeze, clone_agent
from src.versioning.metadata import TrainingHistoryEntry
from src.versioning.registry import next_run_id


# ---------------------------------------------------------------------------
# Run-folder helpers
# ---------------------------------------------------------------------------

def _next_run_id_for_results(board_size: int) -> str:
    """Auto-increment run_NNN in results/size_NN/."""
    size_dir = _cfg.RESULTS_DIR / f"size_{board_size:02d}"
    if not size_dir.exists():
        return "run_001"
    existing = sorted(
        p.name for p in size_dir.iterdir()
        if p.is_dir() and p.name.startswith("run_")
    )
    if not existing:
        return "run_001"
    try:
        n = int(existing[-1].split("_")[1]) + 1
    except (IndexError, ValueError):
        n = len(existing) + 1
    return f"run_{n:03d}"


def resolve_run_id(board_size: int, requested: str | None) -> str:
    """Return a consistent run_id for both results/ and models/."""
    if requested:
        return requested
    # Take the max of what models and results folders suggest
    r_id = _next_run_id_for_results(board_size)
    m_id = next_run_id(board_size, _cfg.MODELS_DIR)
    # Pick the higher number so we never collide with either folder
    r_n = int(r_id.split("_")[1])
    m_n = int(m_id.split("_")[1])
    n = max(r_n, m_n)
    return f"run_{n:03d}"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a DQN agent for C_lines")
    p.add_argument("--games", type=int, default=TRAINING_GAMES)
    p.add_argument("--size",  type=int, default=DEFAULT_BOARD_SIZE)
    p.add_argument("--mode",  type=str, default=DEFAULT_MODE)
    p.add_argument("--run-id", type=str, default=None,
                   help="Override run ID (e.g. run_001). Auto-increments if omitted.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate(agent: DQNAgent, board_size: int, mode: str, n_games: int) -> dict:
    from src.evaluation.evaluator import evaluate
    env = GameEnv(board_size=board_size, mode=mode)
    stats_r = evaluate(agent, RandomAgent(), env, min(n_games, 100))
    stats_h = evaluate(agent, HeuristicAgent(), env, min(n_games, 50))
    return {
        "win_rate_vs_random":    stats_r["win_rate"],
        "win_rate_vs_heuristic": stats_h["win_rate"],
        "mean_ep_len":           stats_r["mean_ep_len"],
    }


# ---------------------------------------------------------------------------
# LR scheduling
# ---------------------------------------------------------------------------

def _apply_lr_schedule(optimizer, game_idx: int, n_games: int, base_lr: float) -> float:
    """Halve LR at each milestone. Returns current LR."""
    lr = base_lr
    for milestone in LR_DECAY_MILESTONES:
        if game_idx >= int(milestone * n_games):
            lr *= LR_DECAY_FACTOR
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ---------------------------------------------------------------------------
# Pool sampling with recency bias
# ---------------------------------------------------------------------------

def _sample_pool_opponent(pool: list) -> object:
    """Sample from pool, biasing toward the most-recent snapshots."""
    if not pool:
        return RandomAgent()
    top_n = min(RECENT_POOL_TOP_N, len(pool))
    if np.random.random() < RECENT_POOL_BIAS:
        return pool[-np.random.randint(1, top_n + 1)]
    return pool[np.random.randint(len(pool))]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    board_size: int = DEFAULT_BOARD_SIZE,
    mode: str = DEFAULT_MODE,
    n_games: int = TRAINING_GAMES,
    run_id: str | None = None,
) -> None:
    run_id = resolve_run_id(board_size, run_id)

    # Build run-specific output dirs
    results_run = _cfg.RESULTS_DIR / f"size_{board_size:02d}" / run_id
    figs_dir    = results_run / "figures"
    results_run.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    log_path = results_run / "training_log.csv"

    print(f"[train] size={board_size}  mode={mode}  games={n_games}  run={run_id}")
    print(f"        log  → {log_path}")
    print(f"        models → {_cfg.MODELS_DIR / f'size_{board_size:02d}' / run_id}/")

    # Agents & buffer
    agent  = DQNAgent(board_size)
    buffer = ReplayBuffer(REPLAY_CAPACITY)
    env    = GameEnv(board_size=board_size, mode=mode)

    warmup_opponents = [RandomAgent(), HeuristicAgent()]
    random_opp = RandomAgent()

    snapshot_pool: list = []
    history: list[TrainingHistoryEntry] = []
    parent_version_id: str | None = None
    parent_run_uuid = str(uuid.uuid4())
    global_step = 0
    last_loss = 0.0
    current_lr = LR

    log_file = log_path.open("w", newline="")
    writer = csv.writer(log_file)
    writer.writerow([
        "game", "epsilon", "lr", "loss",
        "win_rate_vs_random", "win_rate_vs_heuristic",
        "mean_ep_len", "snapshot",
    ])

    for game_idx in range(n_games):
        epsilon = linear_epsilon(EPS_START, EPS_END, game_idx, EPS_DECAY_GAMES)
        agent.set_epsilon(epsilon)

        # LR scheduling
        current_lr = _apply_lr_schedule(agent._optimizer, game_idx, n_games, LR)

        # ---- Curriculum opponent selection ----
        if game_idx < WARMUP_GAMES:
            # Mixed warmup: 30% heuristic, 70% random
            if np.random.random() < WARMUP_HEURISTIC_PROB:
                opponent = HeuristicAgent()
            else:
                opponent = random_opp
        elif not snapshot_pool:
            opponent = random_opp
        elif np.random.random() < SELF_PLAY_MIX_PROB:
            opponent = _sample_pool_opponent(snapshot_pool)
        else:
            opponent = clone_agent(agent)

        # ---- Play episode ----
        if game_idx % 2 == 0:
            t1, t2, _ = play_episode(agent, opponent, env)
            for t in t1:
                buffer.push(t.state, t.action, t.reward,
                            t.next_state, t.done, t.legal_mask_next)
        else:
            t1, t2, _ = play_episode(opponent, agent, env)
            for t in t2:
                buffer.push(t.state, t.action, t.reward,
                            t.next_state, t.done, t.legal_mask_next)

        # ---- Gradient steps (multiple per game) ----
        if len(buffer) >= BATCH_SIZE:
            for _ in range(GRADIENT_STEPS_PER_GAME):
                batch = buffer.sample(BATCH_SIZE)
                last_loss = agent.update(batch)
                global_step += 1
                if global_step % TARGET_SYNC_STEPS == 0:
                    agent.sync_target()

        # ---- Evaluation (every EVAL_INTERVAL games) ----
        eval_stats: dict = {}
        snap_id = ""
        if game_idx % EVAL_INTERVAL == 0 or game_idx == n_games - 1:
            eval_stats = _evaluate(agent, board_size, mode, EVAL_GAMES)
            writer.writerow([
                game_idx,
                f"{epsilon:.4f}",
                f"{current_lr:.2e}",
                f"{last_loss:.6f}",
                f"{eval_stats.get('win_rate_vs_random', 0):.4f}",
                f"{eval_stats.get('win_rate_vs_heuristic', 0):.4f}",
                f"{eval_stats.get('mean_ep_len', 0):.1f}",
                snap_id,
            ])
            log_file.flush()

        # ---- Snapshot (every SNAPSHOT_INTERVAL games) ----
        if game_idx % SNAPSHOT_INTERVAL == 0 and game_idx > 0:
            meta = freeze(
                agent, game_idx,
                {
                    "win_rate_vs_random":    eval_stats.get("win_rate_vs_random"),
                    "win_rate_vs_heuristic": eval_stats.get("win_rate_vs_heuristic"),
                    "mean_episode_length":   eval_stats.get("mean_ep_len"),
                },
                board_size, run_id,
                parent_run_id=parent_run_uuid,
                parent_version_id=parent_version_id,
                gradient_steps=global_step,
                history=history,
                models_dir=_cfg.MODELS_DIR,
            )
            parent_version_id = meta.version_id
            history.append(TrainingHistoryEntry(
                games_trained=game_idx,
                win_rate_vs_random=meta.win_rate_vs_random,
                elo_rating=meta.elo_rating,
            ))
            cloned = clone_agent(agent)
            snapshot_pool.append(cloned)
            if len(snapshot_pool) > MAX_POOL_SIZE:
                snapshot_pool.pop(0)
            snap_id = meta.version_id

        # ---- Progress print ----
        if game_idx % 100 == 0:
            wr_r = eval_stats.get("win_rate_vs_random", 0)
            wr_h = eval_stats.get("win_rate_vs_heuristic", 0)
            print(f"  game {game_idx:5d}  eps={epsilon:.3f}  lr={current_lr:.1e}"
                  f"  loss={last_loss:.4f}  wr_rand={wr_r:.2%}  wr_heur={wr_h:.2%}")

    log_file.close()

    # Final snapshot
    final_stats = _evaluate(agent, board_size, mode, EVAL_GAMES)
    freeze(
        agent, n_games, final_stats, board_size, run_id,
        parent_run_id=parent_run_uuid,
        parent_version_id=parent_version_id,
        gradient_steps=global_step, history=history,
        models_dir=_cfg.MODELS_DIR,
    )

    # Generate figures
    from src.evaluation.plots import generate_all_plots
    figs = generate_all_plots(log_path, board_size, figs_dir)
    print(f"\nFigures written to {figs_dir}/")
    for f in figs:
        print(f"  {f.name}")

    print(f"\nTraining complete — {n_games} games  size={board_size}  run={run_id}")
    print(f"Final WR vs random={final_stats['win_rate_vs_random']:.0%}"
          f"  vs heuristic={final_stats['win_rate_vs_heuristic']:.0%}")


def main() -> None:
    args = _parse_args()
    train(
        board_size=args.size,
        mode=args.mode,
        n_games=args.games,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
