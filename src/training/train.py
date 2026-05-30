"""Training CLI — unified upgrade plan (U1-U10).

Usage:
    # Simple auto-train (existing interface, backward compatible)
    python -m src.training.train --games 10000 --size 8

    # With benchmark logging against alpha-beta
    python -m src.training.train --games 5000 --size 8 --benchmark alphabeta_d4

    # Schedule-based session
    python -m src.training.train --size 8 --schedule "self:50,pool:200,self:100"

    # Schedule from JSON file
    python -m src.training.train --size 8 --schedule-file schedules/colile_evening.json

Outputs:
    results/size_NN/run_NNN/training_log.csv
    results/size_NN/run_NNN/benchmark_log.csv  (if --benchmark is active)
    results/size_NN/run_NNN/figures/
    models/size_NN/run_NNN/gen_NNN/
"""

from __future__ import annotations
import argparse
import csv
import time
import uuid
from datetime import timedelta
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
    STATE_CHANNELS_V2, NETWORK_ARCH,
    USE_SYMMETRY_AUGMENTATION,
    BENCHMARK_EVERY_N_GAMES_SMALL, BENCHMARK_EVERY_N_GAMES_LARGE,
)
from src.game.env import GameEnv
from src.training.replay_buffer import ReplayBuffer
from src.training.self_play import play_episode, linear_epsilon
from src.training.snapshot import freeze, clone_agent
from src.training.schedule import TrainingSchedule, make_source
from src.training.benchmark_logger import BenchmarkLogger
from src.versioning.metadata import TrainingHistoryEntry
from src.versioning.registry import next_run_id


# ---------------------------------------------------------------------------
# Run-folder helpers
# ---------------------------------------------------------------------------

def _next_run_id_for_results(board_size: int) -> str:
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
    if requested:
        return requested
    r_id = _next_run_id_for_results(board_size)
    m_id = next_run_id(board_size, _cfg.MODELS_DIR)
    r_n = int(r_id.split("_")[1])
    m_n = int(m_id.split("_")[1])
    return f"run_{max(r_n, m_n):03d}"


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

def _sample_pool_opponent(pool: list):
    if not pool:
        return RandomAgent()
    top_n = min(RECENT_POOL_TOP_N, len(pool))
    if np.random.random() < RECENT_POOL_BIAS:
        return pool[-np.random.randint(1, top_n + 1)]
    return pool[np.random.randint(len(pool))]


# ---------------------------------------------------------------------------
# Benchmark agent factory
# ---------------------------------------------------------------------------

def _make_benchmark_agent(benchmark_name: str, board_size: int):
    """Construct a fixed benchmark agent from a name string."""
    if benchmark_name.startswith("alphabeta"):
        depth = None
        if "_d" in benchmark_name:
            try:
                depth = int(benchmark_name.split("_d")[1])
            except ValueError:
                pass
        from src.agents.alphabeta_agent import AlphaBetaAgent
        return AlphaBetaAgent(board_size=board_size, depth=depth)
    if benchmark_name == "heuristic":
        return HeuristicAgent()
    return RandomAgent()


# ---------------------------------------------------------------------------
# Simple auto-train (backward-compatible)
# ---------------------------------------------------------------------------

def _load_resume_state(
    board_size: int, run_id: str, agent: DQNAgent
) -> int:
    """Load the latest snapshot into agent. Returns the game index to resume from."""
    import torch
    from src.versioning.registry import list_by_size_run
    snaps = list_by_size_run(board_size, run_id, _cfg.MODELS_DIR)
    if not snaps:
        print(f"  [resume] No snapshots found for {run_id} — starting from game 0.")
        return 0
    latest = snaps[-1]
    sd = torch.load(latest.weights_path, map_location="cpu", weights_only=True)
    agent.load_state_dict(sd)
    resume_from = latest.games_trained
    eps = linear_epsilon(EPS_START, EPS_END, resume_from, EPS_DECAY_GAMES)
    agent.set_epsilon(eps)
    print(f"  [resume] Loaded {latest.version_id} ({resume_from} games trained, eps={eps:.3f})")
    return resume_from


def train(
    board_size: int = DEFAULT_BOARD_SIZE,
    mode: str = DEFAULT_MODE,
    n_games: int = TRAINING_GAMES,
    run_id: str | None = None,
    benchmark_name: str | None = None,
    use_augmentation: bool = USE_SYMMETRY_AUGMENTATION,
    resume: bool = False,
) -> None:
    """Auto-train with all upgrades active — the simple existing interface."""
    run_id = resolve_run_id(board_size, run_id)

    results_run = _cfg.RESULTS_DIR / f"size_{board_size:02d}" / run_id
    figs_dir    = results_run / "figures"
    results_run.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    log_path = results_run / "training_log.csv"

    print(f"[train] size={board_size}  mode={mode}  games={n_games}  run={run_id}")
    print(f"        arch={NETWORK_ARCH}  channels={STATE_CHANNELS_V2}  augment={use_augmentation}")
    if benchmark_name:
        print(f"        benchmark={benchmark_name}")

    agent  = DQNAgent(board_size, in_channels=STATE_CHANNELS_V2, network_arch=NETWORK_ARCH)
    buffer = ReplayBuffer(REPLAY_CAPACITY, use_augmentation=use_augmentation)
    env    = GameEnv(board_size=board_size, mode=mode)

    # Fixed benchmark agent (constructed once, never updated)
    bm_logger: BenchmarkLogger | None = None
    if benchmark_name:
        bm_agent = _make_benchmark_agent(benchmark_name, board_size)
        bm_env   = GameEnv(board_size=board_size, mode=mode)
        bm_every = (BENCHMARK_EVERY_N_GAMES_SMALL if board_size <= 10
                    else BENCHMARK_EVERY_N_GAMES_LARGE)
        bm_logger = BenchmarkLogger(
            log_path=results_run / "benchmark_log.csv",
            benchmark_agent=bm_agent,
            env=bm_env,
            benchmark_name=benchmark_name,
        )

    warmup_opponents = [RandomAgent(), HeuristicAgent()]
    random_opp = RandomAgent()

    snapshot_pool: list = []
    history: list[TrainingHistoryEntry] = []
    parent_version_id: str | None = None
    parent_run_uuid = str(uuid.uuid4())
    global_step = 0
    last_loss = 0.0
    current_lr = LR

    # Resume: load latest snapshot weights and find start game
    start_game = 0
    if resume:
        start_game = _load_resume_state(board_size, run_id, agent)

    train_start = time.monotonic()

    # Open CSV — append if resuming so existing rows are preserved
    csv_mode = "a" if resume and log_path.exists() else "w"
    log_file = log_path.open(csv_mode, newline="")
    writer = csv.writer(log_file)
    if csv_mode == "w":
        writer.writerow([
            "game", "epsilon", "lr", "loss",
            "win_rate_vs_random", "win_rate_vs_heuristic",
            "mean_ep_len", "elapsed_sec", "games_per_hour", "snapshot",
        ])

    for game_idx in range(start_game, n_games):
        epsilon = linear_epsilon(EPS_START, EPS_END, game_idx, EPS_DECAY_GAMES)
        agent.set_epsilon(epsilon)
        current_lr = _apply_lr_schedule(agent._optimizer, game_idx, n_games, LR)

        # ---- Curriculum opponent selection ----
        if game_idx < WARMUP_GAMES:
            if np.random.random() < WARMUP_HEURISTIC_PROB:
                opponent = HeuristicAgent()
            else:
                opponent = random_opp
            source_weight = _cfg.DEFAULT_SOURCE_WEIGHTS.get("heuristic", 1.0)
        elif not snapshot_pool:
            opponent = random_opp
            source_weight = _cfg.DEFAULT_SOURCE_WEIGHTS.get("random", 0.5)
        elif np.random.random() < SELF_PLAY_MIX_PROB:
            opponent = _sample_pool_opponent(snapshot_pool)
            source_weight = _cfg.DEFAULT_SOURCE_WEIGHTS.get("pool", 1.0)
        else:
            opponent = clone_agent(agent)
            source_weight = _cfg.DEFAULT_SOURCE_WEIGHTS.get("self", 1.0)

        # ---- Play episode ----
        if game_idx % 2 == 0:
            t1, t2, _ = play_episode(agent, opponent, env)
            learner_transitions = t1
        else:
            t1, t2, _ = play_episode(opponent, agent, env)
            learner_transitions = t2

        for t in learner_transitions:
            buffer.push(
                t.state, t.action, t.reward,
                t.next_state, t.done, t.legal_mask_next,
                weight=source_weight,
            )

        # ---- Gradient steps ----
        if len(buffer) >= BATCH_SIZE:
            for _ in range(GRADIENT_STEPS_PER_GAME):
                batch = buffer.sample(BATCH_SIZE)
                last_loss = agent.update(batch)
                global_step += 1
                if global_step % TARGET_SYNC_STEPS == 0:
                    agent.sync_target()

        # ---- Benchmark check ----
        eval_stats: dict = {}
        snap_id = ""
        if bm_logger and game_idx % bm_every == 0:
            bm_logger.run(agent, game_idx, phase_name="auto")

        # ---- Evaluation ----
        if game_idx % EVAL_INTERVAL == 0 or game_idx == n_games - 1:
            eval_stats = _evaluate(agent, board_size, mode, EVAL_GAMES)
            elapsed = time.monotonic() - train_start
            games_done = game_idx - start_game + 1
            gph = games_done / elapsed * 3600 if elapsed > 0 else 0
            writer.writerow([
                game_idx,
                f"{epsilon:.4f}",
                f"{current_lr:.2e}",
                f"{last_loss:.6f}",
                f"{eval_stats.get('win_rate_vs_random', 0):.4f}",
                f"{eval_stats.get('win_rate_vs_heuristic', 0):.4f}",
                f"{eval_stats.get('mean_ep_len', 0):.1f}",
                f"{elapsed:.0f}",
                f"{gph:.0f}",
                snap_id,
            ])
            log_file.flush()

        # ---- Snapshot ----
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
            elapsed = time.monotonic() - train_start
            games_done = game_idx - start_game + 1
            gph = games_done / elapsed * 3600 if elapsed > 0 else 0
            remaining = (n_games - game_idx) / (gph / 3600) if gph > 0 else 0
            eta = str(timedelta(seconds=int(remaining)))
            print(f"  game {game_idx:5d}  eps={epsilon:.3f}  lr={current_lr:.1e}"
                  f"  loss={last_loss:.4f}  wr_rand={wr_r:.2%}  wr_heur={wr_h:.2%}"
                  f"  {gph:.0f} g/h  ETA {eta}")

    log_file.close()
    if bm_logger:
        bm_logger.close()

    final_stats = _evaluate(agent, board_size, mode, EVAL_GAMES)
    freeze(
        agent, n_games, final_stats, board_size, run_id,
        parent_run_id=parent_run_uuid,
        parent_version_id=parent_version_id,
        gradient_steps=global_step, history=history,
        models_dir=_cfg.MODELS_DIR,
    )

    from src.evaluation.plots import generate_all_plots, generate_benchmark_plot
    figs = generate_all_plots(log_path, board_size, figs_dir)
    if benchmark_name and (results_run / "benchmark_log.csv").exists():
        bm_fig = generate_benchmark_plot(
            results_run / "benchmark_log.csv", board_size, figs_dir
        )
        if bm_fig:
            figs.append(bm_fig)

    print(f"\nFigures written to {figs_dir}/")
    for f in figs:
        print(f"  {f.name}")

    print(f"\nTraining complete — {n_games} games  size={board_size}  run={run_id}")
    print(f"Final WR vs random={final_stats['win_rate_vs_random']:.0%}"
          f"  vs heuristic={final_stats['win_rate_vs_heuristic']:.0%}")


# ---------------------------------------------------------------------------
# Schedule-based train (U4 — human-in-the-loop sessions)
# ---------------------------------------------------------------------------

def train_schedule(
    schedule: TrainingSchedule,
    board_size: int = DEFAULT_BOARD_SIZE,
    mode: str = DEFAULT_MODE,
    run_id: str | None = None,
    use_augmentation: bool = USE_SYMMETRY_AUGMENTATION,
) -> None:
    """Run a full TrainingSchedule — supports mixed opponents and human phases."""
    run_id = resolve_run_id(board_size, run_id)

    results_run = _cfg.RESULTS_DIR / f"size_{board_size:02d}" / run_id
    figs_dir    = results_run / "figures"
    results_run.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    log_path = results_run / "training_log.csv"

    print(f"[train_schedule] size={board_size}  run={run_id}")
    print(f"  phases: {[(p.opponent_name, p.n_games) for p in schedule.phases]}")

    agent  = DQNAgent(board_size, in_channels=STATE_CHANNELS_V2, network_arch=NETWORK_ARCH)
    buffer = ReplayBuffer(REPLAY_CAPACITY, use_augmentation=use_augmentation)
    env    = GameEnv(board_size=board_size, mode=mode)

    # Fixed benchmark
    bm_agent  = _make_benchmark_agent(schedule.benchmark_name, board_size)
    bm_env    = GameEnv(board_size=board_size, mode=mode)
    bm_every  = schedule.benchmark_every_n_games
    bm_logger = BenchmarkLogger(
        log_path=results_run / "benchmark_log.csv",
        benchmark_agent=bm_agent,
        env=bm_env,
        benchmark_name=schedule.benchmark_name,
        games_per_check=schedule.benchmark_games_per_check,
    )

    snapshot_pool: list = []
    history: list[TrainingHistoryEntry] = []
    parent_version_id: str | None = None
    parent_run_uuid = str(uuid.uuid4())
    global_step = 0
    last_loss = 0.0
    current_lr = LR
    global_game_idx = 0
    human_games_seen = 0

    log_file = log_path.open("w", newline="")
    writer = csv.writer(log_file)
    writer.writerow([
        "game", "phase", "epsilon", "lr", "loss",
        "win_rate_vs_random", "win_rate_vs_heuristic",
        "mean_ep_len", "snapshot",
    ])

    for phase in schedule.phases:
        src = make_source(phase.opponent_name, pool=snapshot_pool, agent_factory=lambda: agent)
        print(f"\n[phase] {phase.name}  opponent={phase.opponent_name}  games={phase.n_games}  weight={phase.weight}")

        for phase_game in range(phase.n_games):
            epsilon = linear_epsilon(EPS_START, EPS_END, global_game_idx, EPS_DECAY_GAMES)
            agent.set_epsilon(epsilon)
            current_lr = _apply_lr_schedule(
                agent._optimizer, global_game_idx, schedule.total_games(), LR
            )

            opponent = src.next_agent()

            if global_game_idx % 2 == 0:
                t1, t2, _ = play_episode(agent, opponent, env)
                learner_transitions = t1
            else:
                t1, t2, _ = play_episode(opponent, agent, env)
                learner_transitions = t2

            if phase.train:
                for t in learner_transitions:
                    buffer.push(
                        t.state, t.action, t.reward,
                        t.next_state, t.done, t.legal_mask_next,
                        weight=phase.weight,
                    )

            if phase.opponent_name in ("human", "demo"):
                human_games_seen += 1

            # Gradient steps
            if phase.train and len(buffer) >= BATCH_SIZE:
                for _ in range(GRADIENT_STEPS_PER_GAME):
                    batch = buffer.sample(BATCH_SIZE)
                    last_loss = agent.update(batch)
                    global_step += 1
                    if global_step % TARGET_SYNC_STEPS == 0:
                        agent.sync_target()

            # Benchmark
            if global_game_idx % bm_every == 0:
                bm_logger.run(agent, global_game_idx, phase_name=phase.name)

            # Eval
            eval_stats: dict = {}
            snap_id = ""
            if global_game_idx % EVAL_INTERVAL == 0:
                eval_stats = _evaluate(agent, board_size, mode, EVAL_GAMES)
                writer.writerow([
                    global_game_idx, phase.name,
                    f"{epsilon:.4f}", f"{current_lr:.2e}", f"{last_loss:.6f}",
                    f"{eval_stats.get('win_rate_vs_random', 0):.4f}",
                    f"{eval_stats.get('win_rate_vs_heuristic', 0):.4f}",
                    f"{eval_stats.get('mean_ep_len', 0):.1f}", snap_id,
                ])
                log_file.flush()

            # Snapshot
            if global_game_idx % schedule.snapshot_every_n_games == 0 and global_game_idx > 0:
                meta = freeze(
                    agent, global_game_idx, eval_stats or {},
                    board_size, run_id,
                    parent_run_id=parent_run_uuid,
                    parent_version_id=parent_version_id,
                    gradient_steps=global_step,
                    history=history,
                    models_dir=_cfg.MODELS_DIR,
                    human_games_seen=human_games_seen,
                )
                parent_version_id = meta.version_id
                history.append(TrainingHistoryEntry(
                    games_trained=global_game_idx,
                    win_rate_vs_random=meta.win_rate_vs_random,
                    elo_rating=meta.elo_rating,
                ))
                cloned = clone_agent(agent)
                snapshot_pool.append(cloned)
                if len(snapshot_pool) > MAX_POOL_SIZE:
                    snapshot_pool.pop(0)
                snap_id = meta.version_id

            if phase_game % 10 == 0:
                print(f"    game {global_game_idx:5d}  eps={epsilon:.3f}"
                      f"  loss={last_loss:.4f}")

            global_game_idx += 1

    log_file.close()
    bm_logger.close()

    final_stats = _evaluate(agent, board_size, mode, EVAL_GAMES)
    freeze(
        agent, global_game_idx, final_stats, board_size, run_id,
        parent_run_id=parent_run_uuid,
        parent_version_id=parent_version_id,
        gradient_steps=global_step, history=history,
        models_dir=_cfg.MODELS_DIR,
        human_games_seen=human_games_seen,
    )

    from src.evaluation.plots import generate_all_plots, generate_benchmark_plot
    figs = generate_all_plots(log_path, board_size, figs_dir)
    bm_fig = generate_benchmark_plot(
        results_run / "benchmark_log.csv", board_size, figs_dir
    )
    if bm_fig:
        figs.append(bm_fig)

    print(f"\nSchedule complete — {global_game_idx} total games  run={run_id}")
    print(f"Final WR vs random={final_stats['win_rate_vs_random']:.0%}"
          f"  vs heuristic={final_stats['win_rate_vs_heuristic']:.0%}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a DQN agent for C_lines (unified upgrade)")
    p.add_argument("--games",          type=int, default=TRAINING_GAMES)
    p.add_argument("--size",           type=int, default=DEFAULT_BOARD_SIZE)
    p.add_argument("--mode",           type=str, default=DEFAULT_MODE)
    p.add_argument("--run-id",         type=str, default=None)
    p.add_argument("--benchmark",      type=str, default=None,
                   help="Fixed benchmark agent name, e.g. 'alphabeta_d4' or 'heuristic'")
    p.add_argument("--schedule",       type=str, default=None,
                   help="Inline schedule string, e.g. 'self:50,pool:200'")
    p.add_argument("--schedule-file",  type=str, default=None,
                   help="Path to a JSON schedule file")
    p.add_argument("--no-augment",     action="store_true", default=False,
                   help="Disable symmetry augmentation")
    p.add_argument("--resume",         action="store_true", default=False,
                   help="Resume from the latest snapshot saved for --run-id")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    use_aug = not args.no_augment

    if args.schedule or args.schedule_file:
        # Schedule-based run
        if args.schedule_file:
            text = Path(args.schedule_file).read_text()
            schedule = TrainingSchedule.from_json(text)
        else:
            schedule = TrainingSchedule.from_string(args.schedule, board_size=args.size)
        if args.benchmark:
            schedule.benchmark_name = args.benchmark
        train_schedule(schedule, board_size=args.size, mode=args.mode,
                       run_id=args.run_id, use_augmentation=use_aug)
    else:
        # Simple auto-train
        train(
            board_size=args.size,
            mode=args.mode,
            n_games=args.games,
            run_id=args.run_id,
            benchmark_name=args.benchmark,
            use_augmentation=use_aug,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
