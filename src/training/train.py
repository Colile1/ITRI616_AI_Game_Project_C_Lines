"""Training CLI — unified upgrade plan (U1-U10) + post-run improvements.

Usage:
    # Auto self-play training
    python -m src.training.train --games 10000 --size 8 --benchmark alphabeta_d4

    # Start with human play, switch to auto when tired
    python -m src.training.train --games 10000 --size 8 --human

    # Resume a previous run
    python -m src.training.train --games 10000 --size 8 --run-id run_003 --resume

    # Schedule-based session
    python -m src.training.train --size 8 --schedule "self:50,pool:200,self:100"

Human / auto switching (mid-run):
    During training, create a file called  mode.txt  in the run's results folder.
    Write one of:  auto | self | pool | heuristic | human
    The training loop reads it at each snapshot boundary and switches opponent.
    Delete the file to hold the current mode indefinitely.
    Typing 'q' during a human move also writes 'auto' to mode.txt automatically.

Outputs:
    results/size_NN/run_NNN/training_log.csv    — eval metrics every 100 games
    results/size_NN/run_NNN/game_log.csv        — every single game result
    results/size_NN/run_NNN/benchmark_log.csv   — benchmark checks (if --benchmark)
    results/size_NN/run_NNN/best/weights.pt     — best model seen during run
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
import torch

# Use all available CPU threads for intra-op parallelism
torch.set_num_threads(torch.get_num_threads())
torch.set_num_interop_threads(max(1, torch.get_num_interop_threads()))

import src.config as _cfg
from src.agents.base_agent import BaseAgent
from src.agents.dqn_agent import DQNAgent
from src.agents.random_agent import RandomAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.agents.points_heuristic_agent import PointsHeuristicAgent
from src.config import (
    DEFAULT_BOARD_SIZE, DEFAULT_MODE,
    MODE_FIRST_TO_FOUR,
    BATCH_SIZE, REPLAY_CAPACITY,
    EPS_START, EPS_END, EPS_DECAY_GAMES,
    TARGET_SYNC_STEPS, EVAL_INTERVAL,
    EVAL_GAMES_VS_RANDOM, EVAL_GAMES_VS_HEURISTIC,
    SNAPSHOT_INTERVAL, MAX_POOL_SIZE, WARMUP_GAMES,
    WARMUP_HEURISTIC_PROB, SELF_PLAY_MIX_PROB,
    RECENT_POOL_BIAS, RECENT_POOL_TOP_N,
    GRADIENT_STEPS_PER_GAME, TRAINING_GAMES,
    LR, LR_DECAY_FACTOR, PLATEAU_PATIENCE,
    LOSS_REWARD, FTF_EARLY_LOSS_TURNS, FTF_EARLY_LOSS_EXTRA,
    STATE_CHANNELS_V2, NETWORK_ARCH,
    USE_SYMMETRY_AUGMENTATION,
    BENCHMARK_EVERY_N_GAMES_SMALL, BENCHMARK_EVERY_N_GAMES_LARGE,
    BENCHMARK_GAMES_PER_CHECK, MIN_POOL_WR,
    N_STEP_RETURNS, GAMMA,
)
from src.game.env import GameEnv
from src.training.replay_buffer import ReplayBuffer
from src.training.self_play import play_episode, linear_epsilon
from src.training.snapshot import freeze, clone_agent
from src.training.schedule import TrainingSchedule, make_source
from src.training.benchmark_logger import BenchmarkLogger
from src.training.game_logger import GameLogger
from src.evaluation.elo import update_elo, expected_score
from src.versioning.metadata import TrainingHistoryEntry
from src.versioning.registry import next_run_id


# ---------------------------------------------------------------------------
# Parallel episode collection worker (module-level — required for Windows
# multiprocessing 'spawn' start method; cannot be a lambda or closure)
# ---------------------------------------------------------------------------

def _episode_worker(args: tuple) -> tuple:
    """Collect one episode in a worker process.

    Each worker has its own Python interpreter so game simulation runs
    in parallel without GIL contention.  Returns serialisable data only
    (lists of dicts, not Transition dataclasses).
    """
    # Workers only ever run single-state inference, never a gradient step, so a
    # per-worker CUDA context would cost VRAM and kernel-launch latency for no
    # gain.  Set before DQNAgent is imported/constructed — _best_device() reads
    # this at __init__ time.
    import os
    os.environ["FLAT4_DEVICE"] = "cpu"

    import io, torch, numpy as np
    torch.set_num_threads(1)   # N workers × N threads each would oversubscribe

    from src.agents.dqn_agent import DQNAgent
    from src.agents.random_agent import RandomAgent
    from src.agents.heuristic_agent import HeuristicAgent
    from src.game.env import GameEnv
    from src.training.self_play import play_episode
    from src.config import STATE_CHANNELS_V2, NETWORK_ARCH, WARMUP_GAMES

    (agent_bytes, board_size, mode, epsilon,
     opp_bytes, opp_type, game_idx, in_channels, network_arch) = args

    # Reconstruct training agent
    agent = DQNAgent(board_size, eval_only=True,
                     in_channels=in_channels, network_arch=network_arch)
    sd = torch.load(io.BytesIO(agent_bytes), map_location="cpu", weights_only=True)
    agent.load_state_dict(sd)
    agent.set_epsilon(float(epsilon))

    # Opponent
    if opp_type == "prev_best" and opp_bytes is not None:
        opp = DQNAgent(board_size, eval_only=True,
                       in_channels=in_channels, network_arch=network_arch)
        opp_sd = torch.load(io.BytesIO(opp_bytes), map_location="cpu", weights_only=True)
        opp.load_state_dict(opp_sd)
        opp.set_epsilon(0.0)
    elif opp_type == "heuristic":
        opp = HeuristicAgent()
    else:
        # Default to HeuristicAgent instead of Random — Random floods the buffer with
        # low-quality transitions that destroy policies trained against stronger opponents
        opp = HeuristicAgent()

    env = GameEnv(board_size, mode)
    flip = (game_idx % 2 == 1)   # alternate sides each game
    if not flip:
        t1, t2, winner = play_episode(agent, opp, env)
        learner_t = t1
        agent_player = 1
    else:
        t1, t2, winner = play_episode(opp, agent, env)
        learner_t = t2
        agent_player = 2

    # Serialise Transition objects → plain dicts (numpy arrays pickle fine)
    def _ser(trans):
        return {
            "state":          trans.state,
            "action":         int(trans.action),
            "reward":         float(trans.reward),
            "next_state":     trans.next_state,
            "done":           bool(trans.done),
            "legal_mask_next": trans.legal_mask_next,
            "gamma_n":        float(trans.gamma_n),
        }

    return (
        [_ser(t) for t in learner_t],
        winner,
        agent_player,
        len(t1) + len(t2),   # episode length
    )


# ---------------------------------------------------------------------------
# Run-folder helpers
# ---------------------------------------------------------------------------

import re as _re

_MODE_PREFIX: dict[str, str] = {
    "first_to_four": "ftf",
    "points_full":   "pts",
}


def _mode_prefix(mode: str) -> str:
    return _MODE_PREFIX.get(mode, "")


def _next_run_id_for_results(board_size: int, prefix: str = "") -> str:
    size_dir = _cfg.RESULTS_DIR / f"size_{board_size:02d}"
    if prefix:
        pattern = _re.compile(rf"^run_{_re.escape(prefix)}_(\d+)$")
        stem    = f"run_{prefix}_"
    else:
        pattern = _re.compile(r"^run_(\d+)$")
        stem    = "run_"

    if not size_dir.exists():
        return f"{stem}001"

    nums = []
    for p in size_dir.iterdir():
        if p.is_dir():
            m = pattern.match(p.name)
            if m:
                nums.append(int(m.group(1)))

    n = (max(nums) + 1) if nums else 1
    return f"{stem}{n:03d}"


def resolve_run_id(board_size: int, requested: str | None, mode: str = "") -> str:
    if requested:
        return requested
    prefix = _mode_prefix(mode)
    r_id   = _next_run_id_for_results(board_size, prefix)
    m_id   = next_run_id(board_size, _cfg.MODELS_DIR, prefix)

    # Extract trailing number from either format
    r_n = int(r_id.rsplit("_", 1)[-1])
    m_n = int(m_id.rsplit("_", 1)[-1])
    stem = f"run_{prefix}_" if prefix else "run_"
    return f"{stem}{max(r_n, m_n):03d}"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

_ELO_ANCHORS = {
    "random":    600.0,
    "heuristic": 900.0,
    "alphabeta": 1200.0,
}


def _elo_update(
    agent_elo: float,
    agent: DQNAgent,
    board_size: int,
    mode: str,
    n_games: int = 20,
) -> float:
    """Play n_games against each anchor and return the updated agent Elo."""
    from src.evaluation.evaluator import evaluate
    env = GameEnv(board_size=board_size, mode=mode)
    anchors = [
        (RandomAgent(),    _ELO_ANCHORS["random"]),
        (HeuristicAgent(), _ELO_ANCHORS["heuristic"]),
    ]
    elo = agent_elo
    for opp, opp_elo in anchors:
        stats = evaluate(agent, opp, env, n_games)
        wr    = stats["win_rate"]
        score = wr  # 1=win, 0.5=draw, 0=loss averaged over n_games
        elo, _ = update_elo(elo, opp_elo, score, k=16.0)
    return round(elo, 1)


def _evaluate(agent: DQNAgent, board_size: int, mode: str) -> dict:
    from src.evaluation.evaluator import evaluate
    env = GameEnv(board_size=board_size, mode=mode)
    stats_r = evaluate(agent, RandomAgent(),   env, EVAL_GAMES_VS_RANDOM)
    stats_h = evaluate(agent, HeuristicAgent(), env, EVAL_GAMES_VS_HEURISTIC)
    return {
        "win_rate_vs_random":    stats_r["win_rate"],
        "win_rate_vs_heuristic": stats_h["win_rate"],
        "mean_ep_len":           stats_r["mean_ep_len"],
    }


# ---------------------------------------------------------------------------
# Best-model checkpoint
# ---------------------------------------------------------------------------

def _save_best(agent: DQNAgent, best_dir: Path, wr_h: float, game_idx: int) -> None:
    best_dir.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), best_dir / "weights.pt")
    (best_dir / "info.txt").write_text(
        f"game={game_idx}  win_rate_vs_heuristic={wr_h:.4f}\n"
    )


# ---------------------------------------------------------------------------
# Plateau-based LR decay
# ---------------------------------------------------------------------------

def _maybe_decay_lr(
    optimizer,
    plateau_count: int,
    current_lr: float,
    min_lr: float,
) -> tuple[float, int]:
    """Halve LR if no improvement for PLATEAU_PATIENCE evals. Return (new_lr, reset_count)."""
    if plateau_count >= PLATEAU_PATIENCE and current_lr > min_lr:
        new_lr = max(current_lr * LR_DECAY_FACTOR, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr
        print(f"  [lr_decay] plateau {plateau_count} evals → LR {current_lr:.2e} → {new_lr:.2e}")
        return new_lr, 0
    return current_lr, plateau_count


# ---------------------------------------------------------------------------
# Pool helpers
# ---------------------------------------------------------------------------

def _sample_pool_opponent(pool: list):
    if not pool:
        return RandomAgent()
    top_n = min(RECENT_POOL_TOP_N, len(pool))
    if np.random.random() < RECENT_POOL_BIAS:
        return pool[-np.random.randint(1, top_n + 1)]
    return pool[np.random.randint(len(pool))]


def _opponent_label(opponent, is_human_mode: bool) -> str:
    """Human-readable label for the opponent used in game_log.csv."""
    if is_human_mode:
        return "human"
    pool_label = getattr(opponent, "_pool_label", None)
    if pool_label:
        return f"pool:{pool_label}"
    cls = type(opponent).__name__
    if cls == "DQNAgent":
        return "self"
    if cls == "RandomAgent":
        return "random"
    if cls == "HeuristicAgent":
        return "heuristic"
    if cls == "PointsHeuristicAgent":
        return "points_heuristic"
    if cls == "AlphaBetaAgent":
        depth = getattr(opponent, "depth", "?")
        return f"alphabeta_d{depth}"
    return cls.lower()


# ---------------------------------------------------------------------------
# Benchmark agent factory
# ---------------------------------------------------------------------------

def _make_benchmark_agent(benchmark_name: str, board_size: int):
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
    if benchmark_name in ("points_heuristic", "pheuristic"):
        return PointsHeuristicAgent()
    return RandomAgent()


# ---------------------------------------------------------------------------
# Mode signal file
# ---------------------------------------------------------------------------

def _read_mode_signal(run_dir: Path) -> str | None:
    """Read {run_dir}/mode.txt if it exists. Returns content or None."""
    sig = run_dir / "mode.txt"
    if sig.exists():
        return sig.read_text().strip().lower()
    return None


def _make_opponent_from_signal(
    signal: str,
    agent: DQNAgent,
    snapshot_pool: list,
    board_size: int,
    run_dir: Path,
) -> tuple:
    """Return (opponent_callable, source_weight, is_human) for a signal string."""
    from src.agents.terminal_human_agent import TerminalHumanAgent
    sig = signal.split(":")[0].strip()  # ignore optional :N suffix
    if sig in ("auto", "self"):
        return lambda: clone_agent(agent), _cfg.DEFAULT_SOURCE_WEIGHTS.get("self", 1.0), False
    if sig == "pool":
        return lambda: _sample_pool_opponent(snapshot_pool), _cfg.DEFAULT_SOURCE_WEIGHTS.get("pool", 1.0), False
    if sig == "heuristic":
        return lambda: HeuristicAgent(), _cfg.DEFAULT_SOURCE_WEIGHTS.get("heuristic", 1.0), False
    if sig in ("points_heuristic", "pheuristic"):
        return lambda: PointsHeuristicAgent(), _cfg.DEFAULT_SOURCE_WEIGHTS.get("heuristic", 1.0), False
    if sig.startswith("alphabeta"):
        depth = None
        if "_d" in sig:
            try:
                depth = int(sig.split("_d")[1])
            except ValueError:
                pass
        from src.agents.alphabeta_agent import AlphaBetaAgent
        return lambda: AlphaBetaAgent(board_size=board_size, depth=depth), _cfg.DEFAULT_SOURCE_WEIGHTS.get("alphabeta", 2.0), False
    if sig == "human":
        human = TerminalHumanAgent(board_size, switch_signal_path=run_dir / "mode.txt")
        return lambda: human, _cfg.DEFAULT_SOURCE_WEIGHTS.get("human", 5.0), True
    # Unknown signal — default to self
    return lambda: clone_agent(agent), 1.0, False


# ---------------------------------------------------------------------------
# Static previous-best opponent
# ---------------------------------------------------------------------------

def _load_static_opponent(weights_path: str, board_size: int) -> DQNAgent:
    """Load a frozen DQNAgent from a weights file. epsilon=0, no training."""
    agent = DQNAgent(board_size, eval_only=True,
                     in_channels=STATE_CHANNELS_V2, network_arch=NETWORK_ARCH)
    sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    agent.load_state_dict(sd)
    agent.set_epsilon(0.0)
    agent._pool_label = f"prev_best:{Path(weights_path).parts[-3]}"
    return agent


# ---------------------------------------------------------------------------
# Resume helper
# ---------------------------------------------------------------------------

def _load_resume_state(board_size: int, run_id: str, agent: DQNAgent) -> int:
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
    print(f"  [resume] Loaded {latest.version_id} ({resume_from} games, eps={eps:.3f})")

    # Clear any stale mode.txt that could cause immediate quit (e.g. copied from a previous run)
    results_run = _cfg.RESULTS_DIR / f"size_{board_size:02d}" / run_id
    mode_sig = results_run / "mode.txt"
    if mode_sig.exists():
        sig = mode_sig.read_text().strip().lower()
        if sig in ("quit", "q"):
            mode_sig.unlink()
            print(f"  [resume] Removed stale mode.txt (was '{sig}')")

    return resume_from


# ---------------------------------------------------------------------------
# Main auto-train loop
# ---------------------------------------------------------------------------

def train(
    board_size: int = DEFAULT_BOARD_SIZE,
    mode: str = DEFAULT_MODE,
    n_games: int = TRAINING_GAMES,
    run_id: str | None = None,
    benchmark_name: str | None = None,
    use_augmentation: bool = USE_SYMMETRY_AUGMENTATION,
    resume: bool = False,
    start_human: bool = False,
    human_agent_override=None,   # UIHumanAgent injected by _launch_with_ui
    load_weights: str | None = None,    # path to weights.pt to seed the training agent
    start_game: int | None = None,      # override the starting game counter
    prev_best_weights: str | None = None,  # path to frozen opponent replacing RandomAgent
    n_workers: int = 1,                 # parallel episode-collection workers (>1 uses multiprocessing)
    lr_start: float | None = None,      # override initial LR (use lower value when fine-tuning)
    elo_start: float | None = None,     # override starting Elo (carry forward from previous run)
    seed: int | None = None,            # fix torch/numpy/random seed for reproducibility
    benchmark_every: int | None = None, # override games between benchmark checks
) -> None:
    """Auto-train with all improvements active.

    start_human=True begins the session in human-play mode. The user can
    switch at any time using the UI button or by writing to {results_run}/mode.txt.
    human_agent_override: pass a UIHumanAgent to use the PyGame UI instead of
    the terminal agent (set automatically by _launch_with_ui).
    """
    # Reproducibility seed
    if seed is not None:
        import random as _random
        _random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"  [seed] Fixed random seed={seed}")

    run_id = resolve_run_id(board_size, run_id, mode)

    results_run = _cfg.RESULTS_DIR / f"size_{board_size:02d}" / run_id
    figs_dir    = results_run / "figures"
    best_dir    = results_run / "best"
    results_run.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    log_path = results_run / "training_log.csv"

    print(f"[train] size={board_size}  mode={mode}  games={n_games}  run={run_id}")
    print(f"        arch={NETWORK_ARCH}  channels={STATE_CHANNELS_V2}  augment={use_augmentation}")
    if benchmark_name:
        print(f"        benchmark={benchmark_name}  games_per_check={BENCHMARK_GAMES_PER_CHECK}")
    if start_human:
        print(f"        Starting in HUMAN mode. Type 'q' to switch to auto.")
        print(f"        Or write a new mode to:  {results_run / 'mode.txt'}")
    print(f"        mode.txt path: {results_run / 'mode.txt'}")

    agent  = DQNAgent(board_size, in_channels=STATE_CHANNELS_V2, network_arch=NETWORK_ARCH)
    buffer = ReplayBuffer(REPLAY_CAPACITY, use_augmentation=use_augmentation)
    env    = GameEnv(board_size=board_size, mode=mode)

    # Override initial LR to prevent catastrophic forgetting when fine-tuning (R3)
    if lr_start is not None:
        for pg in agent._optimizer.param_groups:
            pg["lr"] = lr_start
        current_lr_init = lr_start
        print(f"  [lr_start] LR overridden to {lr_start:.2e}")
    else:
        current_lr_init = LR

    # Seed agent weights from a previous run's best model (--load-weights)
    if load_weights:
        sd = torch.load(load_weights, map_location="cpu", weights_only=True)
        agent.load_state_dict(sd)
        print(f"  [load_weights] Loaded initial weights from {load_weights}")
        # Auto-read ELO from sibling metadata.json if --elo-start not explicitly given
        if elo_start is None:
            _meta_path = Path(load_weights).parent / "metadata.json"
            if _meta_path.exists():
                import json as _json
                _meta = _json.loads(_meta_path.read_text())
                _elo_from_meta = _meta.get("elo_rating")
                if isinstance(_elo_from_meta, (int, float)) and _elo_from_meta > 0:
                    elo_start = float(_elo_from_meta)
                    print(f"  [elo_start] Auto-read ELO={elo_start:.1f} from {_meta_path.name}")

    # Static frozen opponent replacing RandomAgent (--prev-best)
    if prev_best_weights:
        static_opp = _load_static_opponent(prev_best_weights, board_size)
        print(f"  [prev_best] Using frozen model as base opponent: {prev_best_weights}")
    else:
        static_opp = None

    # Effective random/base opponent — use frozen prev_best if provided, else RandomAgent
    random_opp: BaseAgent = static_opp if static_opp is not None else RandomAgent()

    # Parallel worker pool — bypass GIL for game simulation
    _pool = None
    _opp_bytes_cache: bytes | None = None
    if n_workers > 1:
        import io as _io
        import multiprocessing as _mp
        _mp.set_start_method("spawn", force=True)
        _pool = _mp.Pool(processes=n_workers)
        if prev_best_weights:
            _buf = _io.BytesIO()
            torch.save(static_opp.state_dict(), _buf)
            _opp_bytes_cache = _buf.getvalue()
        print(f"  [workers] Parallel episode collection with {n_workers} workers")

    # Benchmark
    bm_logger: BenchmarkLogger | None = None
    if benchmark_name:
        bm_agent = _make_benchmark_agent(benchmark_name, board_size)
        bm_env   = GameEnv(board_size=board_size, mode=mode)
        bm_every = benchmark_every or (BENCHMARK_EVERY_N_GAMES_SMALL if board_size <= 10
                                       else BENCHMARK_EVERY_N_GAMES_LARGE)
        bm_logger = BenchmarkLogger(
            log_path=results_run / "benchmark_log.csv",
            benchmark_agent=bm_agent,
            env=bm_env,
            benchmark_name=benchmark_name,
            games_per_check=BENCHMARK_GAMES_PER_CHECK,
        )

    game_logger = GameLogger(results_run / "game_log.csv")

    snapshot_pool: list = []
    history: list[TrainingHistoryEntry] = []
    parent_version_id: str | None = None
    parent_run_uuid = str(uuid.uuid4())
    global_step = 0
    last_loss = 0.0
    current_lr = current_lr_init   # respects --lr-start override
    plateau_count = 0
    best_wr_heuristic = 0.0
    best_elo_saved = 0.0           # Elo at the time the best checkpoint was saved
    best_ep_len = float("inf")     # F1: FTF primary metric — lower is better
    min_lr = current_lr_init * 0.125   # floor: three halving steps maximum
    agent_elo = elo_start if elo_start is not None else 800.0
    recent_ep_lens: list[float] = []   # F6: rolling window for adaptive threshold

    # Human mode state — use injected UI agent or fall back to terminal
    is_human_mode = start_human or (human_agent_override is not None)
    human_games_seen = 0
    if human_agent_override is not None:
        human_agent = human_agent_override
    elif start_human:
        from src.agents.terminal_human_agent import TerminalHumanAgent
        human_agent = TerminalHumanAgent(board_size, switch_signal_path=results_run / "mode.txt")
    else:
        human_agent = None

    # start_game resolution (param shadows the local variable):
    # Priority: explicit --start-game > resume snapshot > 0
    _sg_override = start_game   # save the param before any local shadows
    if resume and _sg_override is None:
        _sg_from_resume = _load_resume_state(board_size, run_id, agent)
    else:
        _sg_from_resume = 0
    start_game = _sg_override if _sg_override is not None else _sg_from_resume
    # If --load-weights given without --resume, still set epsilon for the offset
    if load_weights and not resume and start_game > 0:
        eps = linear_epsilon(EPS_START, EPS_END, start_game, EPS_DECAY_GAMES)
        agent.set_epsilon(eps)
        print(f"  [start_game] Starting at game {start_game}, eps={eps:.3f}")

    train_start = time.monotonic()
    csv_mode = "a" if (resume or start_game > 0) and log_path.exists() else "w"
    log_file = log_path.open(csv_mode, newline="")
    writer = csv.writer(log_file)
    if csv_mode == "w":
        writer.writerow([
            "game", "epsilon", "lr", "loss",
            "win_rate_vs_random", "win_rate_vs_heuristic",
            "elo_rating", "mean_ep_len", "elapsed_sec", "games_per_hour", "snapshot",
        ])

    for game_idx in range(start_game, n_games):
        epsilon = linear_epsilon(EPS_START, EPS_END, game_idx, EPS_DECAY_GAMES)
        agent.set_epsilon(epsilon)

        # ---- Mode signal check (every 10 games for fast response) ----
        if game_idx % 10 == 0:
            signal = _read_mode_signal(results_run)
            if signal and signal not in ("quit",):
                want_human = (signal == "human")
                if want_human != is_human_mode:
                    print(f"\n  [mode_switch] '{signal}' — switching to {'HUMAN' if want_human else 'AUTO'} mode.")
                    if want_human:
                        if human_agent is None:
                            from src.agents.terminal_human_agent import TerminalHumanAgent
                            human_agent = TerminalHumanAgent(
                                board_size, switch_signal_path=results_run / "mode.txt"
                            )
                        is_human_mode = True
                    else:
                        is_human_mode = False
            elif signal == "quit":
                print("\n  [mode_switch] Quit signal — finishing training early.")
                break

        # ---- Opponent selection ----
        # Post-warmup: 10% of games always go to a permanent anchor opponent
        # (random or heuristic) so the curriculum never collapses entirely into
        # weak-clone-vs-weak-clone self-play (P1.4 from improvement plan).
        _ANCHOR_PROB = 0.10

        if is_human_mode and human_agent is not None:
            opponent = human_agent
            source_weight = _cfg.DEFAULT_SOURCE_WEIGHTS.get("human", 5.0)
            human_games_seen += 1
        elif game_idx < WARMUP_GAMES:
            if np.random.random() < WARMUP_HEURISTIC_PROB:
                opponent = HeuristicAgent()
            else:
                opponent = random_opp
            source_weight = _cfg.DEFAULT_SOURCE_WEIGHTS.get("heuristic", 1.0)
        elif np.random.random() < _ANCHOR_PROB:
            # Permanent anchor: keeps the agent calibrated against known baselines
            opponent = HeuristicAgent() if np.random.random() < 0.5 else random_opp
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

        # ---- Pass board to human agent before episode ----
        if is_human_mode and hasattr(opponent, "set_board"):
            opponent.set_board(env.board.grid.copy())

        # ---- Play episode (sequential or parallel) ----
        if _pool is not None and not is_human_mode:
            # Parallel path: dispatch n_workers episodes simultaneously.
            # Only re-serialise weights every 50 games — pickle of 500KB on
            # every single step was the bottleneck eating the parallelism gain.
            import io as _io
            _SYNC_EVERY = 50
            if not hasattr(train, '_agent_bytes_cache') or game_idx % _SYNC_EVERY == 0:
                _abuf = _io.BytesIO()
                torch.save(agent.state_dict(), _abuf)
                train._agent_bytes_cache = _abuf.getvalue()
            _agent_bytes = train._agent_bytes_cache

            # Determine opponent type for workers
            # Use heuristic as default (not random) — random floods buffer with
            # low-quality data that destroys policies trained against stronger opponents
            _opp_type = "prev_best" if prev_best_weights else "heuristic"
            if game_idx < WARMUP_GAMES:
                _opp_type = "heuristic"

            _worker_args = [
                (_agent_bytes, board_size, mode, epsilon,
                 _opp_bytes_cache, _opp_type,
                 game_idx + _wi,
                 STATE_CHANNELS_V2, NETWORK_ARCH)
                for _wi in range(n_workers)
            ]
            _par_results = _pool.map(_episode_worker, _worker_args)

            # Process first result as the "main" episode for this game_idx
            _par_transitions_raw, ep_winner, agent_player, ep_len = _par_results[0]
            from src.training.self_play import Transition as _Tr
            learner_transitions = [
                _Tr(r["state"], r["action"], r["reward"],
                    r["next_state"], r["done"], r["legal_mask_next"], r["gamma_n"])
                for r in _par_transitions_raw
            ]
            human_player = 3 - agent_player  # unused in auto mode

            # Push extra parallel episodes directly to buffer (bonus data)
            for _par_t_raw, _par_winner, _par_ap, _par_ep in _par_results[1:]:
                _par_ts = [
                    _Tr(r["state"], r["action"], r["reward"],
                        r["next_state"], r["done"], r["legal_mask_next"], r["gamma_n"])
                    for r in _par_t_raw
                ]
                for _t in _par_ts:
                    buffer.push(_t.state, _t.action, _t.reward,
                                _t.next_state, _t.done, _t.legal_mask_next,
                                weight=source_weight, gamma_n=_t.gamma_n)

            t1, t2, human_transitions = [], [], []   # not used in auto parallel mode
        else:
            # Sequential path (default, also used for human mode)
            if game_idx % 2 == 0:
                t1, t2, ep_winner = play_episode(agent, opponent, env)
                learner_transitions = t1
                agent_player        = 1
                human_player        = 2
                human_transitions   = t2
            else:
                t1, t2, ep_winner = play_episode(opponent, agent, env)
                learner_transitions = t2
                agent_player        = 2
                human_player        = 1
                human_transitions   = t1

        # ---- Notify UI of game result ----
        if is_human_mode and hasattr(opponent, "notify_game_end"):
            ep_len = len(t1) + len(t2)
            if ep_winner == human_player:
                human_result   = "WIN"
                human_terminal = _cfg.WIN_REWARD
                agent_terminal = _cfg.LOSS_REWARD - (FTF_EARLY_LOSS_EXTRA if (mode == MODE_FIRST_TO_FOUR and ep_len <= FTF_EARLY_LOSS_TURNS) else 0.0)
            elif ep_winner is None:
                human_result   = "DRAW"
                human_terminal = _cfg.DRAW_REWARD
                agent_terminal = _cfg.DRAW_REWARD
            else:
                human_result   = "LOSS"
                human_terminal = _cfg.LOSS_REWARD - (FTF_EARLY_LOSS_EXTRA if (mode == MODE_FIRST_TO_FOUR and ep_len <= FTF_EARLY_LOSS_TURNS) else 0.0)
                agent_terminal = _cfg.WIN_REWARD
            opponent.notify_game_end({
                "grid":          env.board.grid.copy(),
                "human_player":  human_player,
                "winner_player": ep_winner,
                "human_result":  human_result,
                "human_reward":  human_terminal,
                "agent_reward":  agent_terminal,
                "game_idx":      game_idx,
            })

        # ---- Log every game result ----
        opp_label = _opponent_label(opponent, is_human_mode)
        if game_idx % 2 == 0:          # agent=P1, opponent=P2
            game_logger.log(game_idx, "dqn", opp_label, ep_winner, t1, t2)
        else:                           # opponent=P1, agent=P2
            game_logger.log(game_idx, opp_label, "dqn", ep_winner, t1, t2)

        # ---- Auto-play display update ----
        if not is_human_mode and human_agent is not None and hasattr(human_agent, "push_display"):
            human_agent.push_display(
                grid      = env.board.grid.copy(),
                winner    = ep_winner,
                p1_label  = "dqn" if agent_player == 1 else opp_label,
                p2_label  = opp_label if agent_player == 1 else "dqn",
                game_idx  = game_idx,
            )

        ep_len = len(t1) + len(t2)

        # ---- F6: Adaptive early-loss threshold ----
        # Track recent episode lengths to adjust the "early loss" threshold.
        recent_ep_lens.append(ep_len)
        if len(recent_ep_lens) > 500:
            recent_ep_lens.pop(0)
        if recent_ep_lens:
            recent_mean = float(np.mean(recent_ep_lens))
            adaptive_threshold = max(FTF_EARLY_LOSS_TURNS, int(recent_mean * 1.2))
        else:
            adaptive_threshold = FTF_EARLY_LOSS_TURNS

        # ---- FTF synthetic loss transition ----------------------------------------
        # In first_to_four mode the LOSER never makes the final move, so they never
        # receive a done=True terminal reward from the environment.  Without it the
        # early-loss penalty and the normal -1 signal are both invisible to the DQN.
        if (mode == MODE_FIRST_TO_FOUR
                and ep_winner is not None
                and ep_winner != agent_player
                and learner_transitions):
            loss_rew = (LOSS_REWARD - FTF_EARLY_LOSS_EXTRA
                        if ep_len <= adaptive_threshold
                        else LOSS_REWARD)
            last_t   = learner_transitions[-1]
            buffer.push(
                last_t.next_state,
                last_t.action,
                loss_rew,
                last_t.next_state,
                True,
                np.zeros(board_size ** 2, dtype=bool),
                weight=source_weight,
                gamma_n=GAMMA,
            )
        # -----------------------------------------------------------------------

        for t in learner_transitions:
            buffer.push(
                t.state, t.action, t.reward,
                t.next_state, t.done, t.legal_mask_next,
                weight=source_weight,
                gamma_n=t.gamma_n,
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
            eval_stats = _evaluate(agent, board_size, mode)
            elapsed = time.monotonic() - train_start
            games_done = game_idx - start_game + 1
            gph = games_done / elapsed * 3600 if elapsed > 0 else 0

            wr_h   = eval_stats.get("win_rate_vs_heuristic", 0.0)
            ep_len_eval = eval_stats.get("mean_ep_len", float("inf"))

            # F1: FTF uses mean episode length as primary metric (lower = better).
            # PTS mode still uses WR vs heuristic (higher = better).
            if mode == MODE_FIRST_TO_FOUR:
                improved = ep_len_eval < best_ep_len
                if improved:
                    best_ep_len = ep_len_eval
                    best_wr_heuristic = wr_h   # track for logging even if not primary
                    _save_best(agent, best_dir, wr_h, game_idx)
                    plateau_count = 0
                    print(f"  [best] New best ep_len: {ep_len_eval:.1f} (wr_h={wr_h:.2%}) at game {game_idx}")
                else:
                    plateau_count += 1
                    if wr_h > best_wr_heuristic:
                        best_wr_heuristic = wr_h
            else:
                _is_better = (wr_h > best_wr_heuristic) or (
                    wr_h == best_wr_heuristic and agent_elo > best_elo_saved
                )
                if _is_better:
                    best_wr_heuristic = wr_h
                    best_elo_saved = agent_elo
                    _save_best(agent, best_dir, wr_h, game_idx)
                    plateau_count = 0
                    print(f"  [best] New best WR={wr_h:.2%} ELO={agent_elo:.1f} at game {game_idx}")
                else:
                    plateau_count += 1

            # Plateau-based LR decay
            current_lr, plateau_count = _maybe_decay_lr(
                agent._optimizer, plateau_count, current_lr, min_lr
            )

            # Update Elo (lightweight: 20 games vs random + heuristic)
            agent_elo = _elo_update(agent_elo, agent, board_size, mode, n_games=20)

            # Push live stats to UI sidebar
            if human_agent is not None and hasattr(human_agent, "push_stats"):
                human_agent.push_stats({
                    "game":         game_idx,
                    "epsilon":      epsilon,
                    "wr_random":    eval_stats.get("win_rate_vs_random"),
                    "wr_heuristic": wr_h,
                    "best_wr":      best_wr_heuristic,
                    "elo":          agent_elo,
                    "mode":         "human" if is_human_mode else "auto",
                })

            writer.writerow([
                game_idx,
                f"{epsilon:.4f}",
                f"{current_lr:.2e}",
                f"{last_loss:.6f}",
                f"{eval_stats.get('win_rate_vs_random', 0):.4f}",
                f"{wr_h:.4f}",
                f"{agent_elo:.1f}",
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
                    "elo_rating":            agent_elo,
                },
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
                games_trained=game_idx,
                win_rate_vs_random=meta.win_rate_vs_random,
                elo_rating=meta.elo_rating,
            ))
            snap_id = meta.version_id

            # Pool quality gate — only add if WR vs random meets minimum
            wr_r = eval_stats.get("win_rate_vs_random", 0.0) or 0.0
            if wr_r >= MIN_POOL_WR:
                cloned = clone_agent(agent)
                cloned._pool_label = meta.version_id   # traceable in game_log
                snapshot_pool.append(cloned)
                if len(snapshot_pool) > MAX_POOL_SIZE:
                    snapshot_pool.pop(0)
            else:
                print(f"  [pool_gate] Snapshot skipped pool (WR_random={wr_r:.0%} < {MIN_POOL_WR:.0%})")

        # ---- Progress print ----
        if game_idx % 100 == 0:
            wr_r = eval_stats.get("win_rate_vs_random", 0)
            wr_h = eval_stats.get("win_rate_vs_heuristic", 0)
            ep_len_pr = eval_stats.get("mean_ep_len", 0)
            elapsed = time.monotonic() - train_start
            games_done = game_idx - start_game + 1
            gph = games_done / elapsed * 3600 if elapsed > 0 else 0
            remaining = (n_games - game_idx) / (gph / 3600) if gph > 0 else 0
            eta = str(timedelta(seconds=int(remaining)))
            mode_str = "HUMAN" if is_human_mode else "auto"
            if mode == MODE_FIRST_TO_FOUR:
                print(f"  game {game_idx:5d}  eps={epsilon:.3f}  lr={current_lr:.1e}"
                      f"  loss={last_loss:.4f}  ep_len={ep_len_pr:.1f}  best_len={best_ep_len:.1f}"
                      f"  wr_heur={wr_h:.2%}  [{mode_str}]  {gph:.0f} g/h  ETA {eta}")
            else:
                print(f"  game {game_idx:5d}  eps={epsilon:.3f}  lr={current_lr:.1e}"
                      f"  loss={last_loss:.4f}  wr_rand={wr_r:.2%}  wr_heur={wr_h:.2%}"
                      f"  best={best_wr_heuristic:.2%}  [{mode_str}]  {gph:.0f} g/h  ETA {eta}")

    log_file.close()
    game_logger.close()
    if bm_logger:
        bm_logger.close()
    if _pool is not None:
        _pool.close()
        _pool.join()

    final_stats = _evaluate(agent, board_size, mode)
    # R1: compute Elo before the final freeze so gen_N has a valid elo_rating
    agent_elo = _elo_update(agent_elo, agent, board_size, mode, n_games=40)
    final_stats["elo_rating"] = agent_elo
    freeze(
        agent, n_games, final_stats, board_size, run_id,
        parent_run_id=parent_run_uuid,
        parent_version_id=parent_version_id,
        gradient_steps=global_step, history=history,
        models_dir=_cfg.MODELS_DIR,
        human_games_seen=human_games_seen,
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
    if best_wr_heuristic > 0:
        print(f"Best WR vs heuristic during run: {best_wr_heuristic:.0%}"
              f"  (saved to {best_dir}/weights.pt)")


# ---------------------------------------------------------------------------
# PyGame UI launcher (--human flag)
# ---------------------------------------------------------------------------

def _launch_with_ui(
    board_size: int,
    mode: str,
    n_games: int,
    run_id: str | None,
    benchmark_name: str | None,
    use_augmentation: bool,
    resume: bool,
) -> None:
    """Run training in a background thread and the PyGame board in the main thread.

    PyGame requires the main thread for rendering on all platforms. The training
    loop runs behind it and communicates moves/stats via thread-safe queues.
    """
    import threading
    from queue import Queue
    from src.agents.ui_human_agent import UIHumanAgent

    resolved_run_id = resolve_run_id(board_size, run_id, mode)
    results_run     = _cfg.RESULTS_DIR / f"size_{board_size:02d}" / resolved_run_id
    results_run.mkdir(parents=True, exist_ok=True)
    switch_path = results_run / "mode.txt"

    board_q = Queue(maxsize=1)
    move_q  = Queue(maxsize=1)
    stats_q = Queue(maxsize=4)

    ui_agent = UIHumanAgent(
        board_queue=board_q,
        move_queue=move_q,
        stats_queue=stats_q,
        switch_signal_path=switch_path,
    )

    train_thread = threading.Thread(
        target=train,
        kwargs=dict(
            board_size=board_size,
            mode=mode,
            n_games=n_games,
            run_id=resolved_run_id,
            benchmark_name=benchmark_name,
            use_augmentation=use_augmentation,
            resume=resume,
            start_human=True,
            human_agent_override=ui_agent,
        ),
        daemon=True,
        name="TrainingLoop",
    )
    train_thread.start()

    from src.ui.training_board import TrainingBoard
    board_ui = TrainingBoard(
        board_size=board_size,
        mode=mode,
        board_queue=board_q,
        move_queue=move_q,
        stats_queue=stats_q,
        switch_signal_path=switch_path,
        training_thread=train_thread,
    )
    board_ui.run()   # blocks in main thread until window closed

    train_thread.join(timeout=10)


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
    run_id = resolve_run_id(board_size, run_id, mode)

    results_run = _cfg.RESULTS_DIR / f"size_{board_size:02d}" / run_id
    figs_dir    = results_run / "figures"
    best_dir    = results_run / "best"
    results_run.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    log_path = results_run / "training_log.csv"

    print(f"[train_schedule] size={board_size}  run={run_id}")
    print(f"  phases: {[(p.opponent_name, p.n_games) for p in schedule.phases]}")

    agent  = DQNAgent(board_size, in_channels=STATE_CHANNELS_V2, network_arch=NETWORK_ARCH)
    buffer = ReplayBuffer(REPLAY_CAPACITY, use_augmentation=use_augmentation)
    env    = GameEnv(board_size=board_size, mode=mode)

    bm_agent  = _make_benchmark_agent(schedule.benchmark_name, board_size)
    bm_env    = GameEnv(board_size=board_size, mode=mode)
    bm_every  = schedule.benchmark_every_n_games
    bm_logger = BenchmarkLogger(
        log_path=results_run / "benchmark_log.csv",
        benchmark_agent=bm_agent,
        env=bm_env,
        benchmark_name=schedule.benchmark_name,
        games_per_check=max(schedule.benchmark_games_per_check, BENCHMARK_GAMES_PER_CHECK),
    )

    snapshot_pool: list = []
    history: list[TrainingHistoryEntry] = []
    parent_version_id: str | None = None
    parent_run_uuid = str(uuid.uuid4())
    global_step = 0
    last_loss = 0.0
    current_lr = LR
    plateau_count = 0
    best_wr_heuristic = 0.0
    min_lr = LR * 0.125
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
        print(f"\n[phase] {phase.name}  opponent={phase.opponent_name}  games={phase.n_games}")

        for phase_game in range(phase.n_games):
            epsilon = linear_epsilon(EPS_START, EPS_END, global_game_idx, EPS_DECAY_GAMES)
            agent.set_epsilon(epsilon)

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

            if phase.train and len(buffer) >= BATCH_SIZE:
                for _ in range(GRADIENT_STEPS_PER_GAME):
                    batch = buffer.sample(BATCH_SIZE)
                    last_loss = agent.update(batch)
                    global_step += 1
                    if global_step % TARGET_SYNC_STEPS == 0:
                        agent.sync_target()

            if global_game_idx % bm_every == 0:
                bm_logger.run(agent, global_game_idx, phase_name=phase.name)

            eval_stats: dict = {}
            snap_id = ""
            if global_game_idx % schedule.eval_every_n_games == 0:
                eval_stats = _evaluate(agent, board_size, mode)
                wr_h = eval_stats.get("win_rate_vs_heuristic", 0.0)

                if wr_h > best_wr_heuristic:
                    best_wr_heuristic = wr_h
                    _save_best(agent, best_dir, wr_h, global_game_idx)
                    plateau_count = 0
                else:
                    plateau_count += 1

                current_lr, plateau_count = _maybe_decay_lr(
                    agent._optimizer, plateau_count, current_lr, min_lr
                )

                writer.writerow([
                    global_game_idx, phase.name,
                    f"{epsilon:.4f}", f"{current_lr:.2e}", f"{last_loss:.6f}",
                    f"{eval_stats.get('win_rate_vs_random', 0):.4f}",
                    f"{wr_h:.4f}",
                    f"{eval_stats.get('mean_ep_len', 0):.1f}", snap_id,
                ])
                log_file.flush()

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
                snap_id = meta.version_id

                wr_r = (eval_stats or {}).get("win_rate_vs_random", 0.0) or 0.0
                if wr_r >= MIN_POOL_WR:
                    cloned = clone_agent(agent)
                    snapshot_pool.append(cloned)
                    if len(snapshot_pool) > MAX_POOL_SIZE:
                        snapshot_pool.pop(0)

            global_game_idx += 1

    log_file.close()
    bm_logger.close()

    final_stats = _evaluate(agent, board_size, mode)
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
    if best_wr_heuristic > 0:
        print(f"Best WR vs heuristic: {best_wr_heuristic:.0%}  (saved to {best_dir}/)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a DQN agent for C_lines")
    p.add_argument("--games",          type=int, default=TRAINING_GAMES)
    p.add_argument("--size",           type=int, default=DEFAULT_BOARD_SIZE)
    p.add_argument("--mode",           type=str, default=DEFAULT_MODE)
    p.add_argument("--run-id",         type=str, default=None)
    p.add_argument("--benchmark",      type=str, default=None,
                   help="Fixed benchmark agent, e.g. 'alphabeta_d4' or 'heuristic'")
    p.add_argument("--schedule",       type=str, default=None,
                   help="Inline schedule string, e.g. 'self:50,pool:200'")
    p.add_argument("--schedule-file",  type=str, default=None,
                   help="Path to a JSON schedule file")
    p.add_argument("--no-augment",     action="store_true", default=False)
    p.add_argument("--resume",         action="store_true", default=False,
                   help="Resume from the latest snapshot for --run-id")
    p.add_argument("--human",          action="store_true", default=False,
                   help="Start training in human-play mode (terminal input)")
    p.add_argument("--load-weights",   type=str, default=None,
                   help="Path to weights.pt to seed the training agent (e.g. previous run's best/weights.pt)")
    p.add_argument("--start-game",     type=int, default=None,
                   help="Override the starting game counter (e.g. 10000 to continue from game 10001)")
    p.add_argument("--prev-best",      type=str, default=None,
                   help="Path to a frozen weights.pt used as the static base opponent instead of RandomAgent")
    p.add_argument("--workers",        type=int, default=1,
                   help="Number of parallel episode-collection workers (default 1). "
                        "Set to 4-6 to use idle CPU cores for game simulation.")
    p.add_argument("--lr-start",       type=float, default=None,
                   help="Override the initial learning rate (default: config LR=1e-3). "
                        "Use a lower value (e.g. 2e-4) when resuming from a converged model "
                        "to avoid catastrophic forgetting.")
    p.add_argument("--seed",           type=int, default=None,
                   help="Fix random seed for torch, numpy and random for reproducibility.")
    p.add_argument("--elo-start",      type=float, default=None,
                   help="Override starting Elo (e.g. 1097.4 to carry forward from a previous run). "
                        "Auto-read from metadata.json if --load-weights is given and this is omitted.")
    p.add_argument("--benchmark-every", type=int, default=None,
                   help="Games between benchmark checks (default 100 for size<=10). "
                        "One alphabeta_d4 check plays BENCHMARK_GAMES_PER_CHECK=128 "
                        "search-heavy games and can take 15+ minutes, so raise this "
                        "(e.g. 500) on slow hardware. Diagnostic only — it does not "
                        "affect what the agent learns.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    use_aug = not args.no_augment

    if args.schedule or args.schedule_file:
        if args.schedule_file:
            text = Path(args.schedule_file).read_text()
            schedule = TrainingSchedule.from_json(text)
        else:
            schedule = TrainingSchedule.from_string(args.schedule, board_size=args.size)
        if args.benchmark:
            schedule.benchmark_name = args.benchmark
        train_schedule(schedule, board_size=args.size, mode=args.mode,
                       run_id=args.run_id, use_augmentation=use_aug)
    elif args.human:
        # Open the PyGame training board — training runs in a background thread
        _launch_with_ui(
            board_size=args.size,
            mode=args.mode,
            n_games=args.games,
            run_id=args.run_id,
            benchmark_name=args.benchmark,
            use_augmentation=use_aug,
            resume=args.resume,
        )
        # Note: --load-weights / --start-game / --prev-best not yet wired into UI mode
        # (UI mode is interactive; these flags only affect auto training)
    else:
        train(
            board_size=args.size,
            mode=args.mode,
            n_games=args.games,
            run_id=args.run_id,
            benchmark_name=args.benchmark,
            use_augmentation=use_aug,
            resume=args.resume,
            start_human=False,
            load_weights=args.load_weights,
            start_game=args.start_game,
            prev_best_weights=args.prev_best,
            n_workers=args.workers,
            lr_start=args.lr_start,
            elo_start=args.elo_start,
            seed=args.seed,
            benchmark_every=args.benchmark_every,
        )


if __name__ == "__main__":
    main()
