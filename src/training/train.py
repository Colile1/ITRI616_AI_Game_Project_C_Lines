"""Training CLI: python -m src.training.train --games N --size S [--mode MODE]."""

from __future__ import annotations
import argparse
import csv
import uuid
from pathlib import Path

from src.agents.dqn_agent import DQNAgent
from src.agents.random_agent import RandomAgent
from src.config import (
    DEFAULT_BOARD_SIZE, DEFAULT_MODE,
    BATCH_SIZE, REPLAY_CAPACITY,
    EPS_START, EPS_END, EPS_DECAY_GAMES,
    TARGET_SYNC_STEPS, EVAL_INTERVAL, EVAL_GAMES,
    SNAPSHOT_INTERVAL, MAX_POOL_SIZE, WARMUP_GAMES,
    SELF_PLAY_MIX_PROB, TRAINING_LOG_PATH, LOGS_DIR,
)
from src.game.env import GameEnv
from src.training.replay_buffer import ReplayBuffer
from src.training.self_play import play_episode, linear_epsilon
from src.training.snapshot import freeze, clone_agent
from src.versioning.metadata import TrainingHistoryEntry


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a DQN agent for C_lines")
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--size", type=int, default=DEFAULT_BOARD_SIZE)
    p.add_argument("--mode", type=str, default=DEFAULT_MODE)
    p.add_argument("--run-id", type=str, default=None)
    return p.parse_args()


def _evaluate(agent: DQNAgent, board_size: int, mode: str, n_games: int) -> dict:
    from src.evaluation.evaluator import evaluate
    env = GameEnv(board_size=board_size, mode=mode)
    opponent = RandomAgent()
    return evaluate(agent, opponent, env, n_games)


def train(
    board_size: int = DEFAULT_BOARD_SIZE,
    mode: str = DEFAULT_MODE,
    n_games: int = 1000,
    run_id: str | None = None,
) -> None:
    run_id = run_id or str(uuid.uuid4())
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    agent = DQNAgent(board_size)
    buffer = ReplayBuffer(REPLAY_CAPACITY)
    env = GameEnv(board_size=board_size, mode=mode)
    random_opp = RandomAgent()

    snapshot_pool: list = []
    history: list[TrainingHistoryEntry] = []
    parent_version_id: str | None = None
    global_step = 0
    last_loss = 0.0

    log_path = Path(str(TRAINING_LOG_PATH).replace(
        "training_log.csv", f"training_log_size{board_size}.csv"
    ))
    write_header = not log_path.exists()
    log_file = log_path.open("a", newline="")
    writer = csv.writer(log_file)
    if write_header:
        writer.writerow(["game", "epsilon", "loss", "win_rate_vs_random",
                         "mean_ep_len", "snapshot"])

    for game_idx in range(n_games):
        epsilon = linear_epsilon(EPS_START, EPS_END, game_idx, EPS_DECAY_GAMES)
        agent.set_epsilon(epsilon)

        # Curriculum: warmup vs random, then mix self-play
        import numpy as np
        if game_idx < WARMUP_GAMES or not snapshot_pool:
            opponent = random_opp
        elif np.random.random() < SELF_PLAY_MIX_PROB and snapshot_pool:
            opponent = snapshot_pool[np.random.randint(len(snapshot_pool))]
        else:
            opponent = clone_agent(agent)

        # Alternate sides
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

        # Gradient steps
        if len(buffer) >= BATCH_SIZE:
            batch = buffer.sample(BATCH_SIZE)
            last_loss = agent.update(batch)
            global_step += 1
            if global_step % TARGET_SYNC_STEPS == 0:
                agent.sync_target()

        # Evaluation
        eval_stats: dict = {}
        snap_id = ""
        if game_idx % EVAL_INTERVAL == 0 or game_idx == n_games - 1:
            eval_stats = _evaluate(agent, board_size, mode, min(EVAL_GAMES, 50))
            writer.writerow([
                game_idx, f"{epsilon:.4f}", f"{last_loss:.6f}",
                f"{eval_stats.get('win_rate', 0):.4f}",
                f"{eval_stats.get('mean_ep_len', 0):.1f}",
                snap_id,
            ])
            log_file.flush()

        # Snapshot
        if game_idx % SNAPSHOT_INTERVAL == 0 and game_idx > 0:
            meta = freeze(
                agent, game_idx,
                {
                    "win_rate_vs_random": eval_stats.get("win_rate"),
                    "mean_episode_length": eval_stats.get("mean_ep_len"),
                },
                board_size,
                parent_run_id=run_id,
                parent_version_id=parent_version_id,
                gradient_steps=global_step,
                history=history,
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
            print(f"  [snap] {meta.version_id} at game {game_idx}")

        if game_idx % 100 == 0:
            wr = eval_stats.get("win_rate", 0)
            print(f"  game {game_idx:5d}  eps={epsilon:.3f}  loss={last_loss:.4f}  wr={wr:.3f}")

    log_file.close()

    # Final snapshot
    eval_stats = _evaluate(agent, board_size, mode, min(EVAL_GAMES, 50))
    freeze(
        agent, n_games, eval_stats, board_size,
        parent_run_id=run_id, parent_version_id=parent_version_id,
        gradient_steps=global_step, history=history,
    )
    print(f"Training complete. {n_games} games, size={board_size}.")


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
