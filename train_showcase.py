"""Showcase training script for ITRI 616 deliverables.

Trains a DQN agent for First-to-Four on 8x8 from scratch.
Optimised for:
  1. Clear, monotone improvement curves (episode length, WR, Elo)
  2. Fast wall-clock time (parallel workers, no UI overhead)
  3. Dense evidence output (eval every 50 games, benchmark every 50 games)
  4. Self-contained — runs in one command, produces all figures on completion

Usage:
    python train_showcase.py

Outputs:
    results/size_08/run_showcase/training_log.csv
    results/size_08/run_showcase/benchmark_log.csv
    results/size_08/run_showcase/game_log.csv
    results/size_08/run_showcase/best/weights.pt
    results/size_08/run_showcase/figures/*.png   (5 training + 1 benchmark)
    models/size_08/run_showcase/gen_*/           (snapshot every 500 games)
"""

import sys
import os

# Disable pygame display — no window needed for showcase training
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# ---- Patch config for showcase -----------------------------------------------
import src.config as _cfg

# Denser evaluation for clearer curves
_cfg.EVAL_INTERVAL              = 50    # eval every 50 games (default 100)
_cfg.EVAL_GAMES_VS_RANDOM       = 200   # tight estimate
_cfg.EVAL_GAMES_VS_HEURISTIC    = 100

# Denser snapshots for finer-grained difficulty ladder
_cfg.SNAPSHOT_INTERVAL          = 500   # every 500 games (default 1000)

# Denser benchmark — fires every 50 games with 32 games per check
_cfg.BENCHMARK_EVERY_N_GAMES_SMALL = 50
_cfg.BENCHMARK_GAMES_PER_CHECK     = 32  # keep fast; 32 is enough for trend

# Pool settings optimised for learning signal diversity
_cfg.MAX_POOL_SIZE   = 30
_cfg.RECENT_POOL_BIAS = 0.5

# Warmup: longer warmup so buffer is well-filled before self-play starts
_cfg.WARMUP_GAMES = 1500

# ---- Run training ------------------------------------------------------------
from src.training.train import train

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    print("=" * 60)
    print("ITRI 616 Showcase Training Run")
    print("Mode: First-to-Four | Board: 8x8 | Games: 10,000")
    print("Eval every: 50 games | Benchmark every: 50 games")
    print("=" * 60)

    train(
        board_size=8,
        mode="first_to_four",
        n_games=10_000,
        run_id="run_showcase",
        benchmark_name="alphabeta_d4",
        use_augmentation=True,
        resume=False,
        start_human=False,
        n_workers=1,
    )

    print()
    print("=" * 60)
    print("Showcase run complete.")
    print("Figures: results/size_08/run_showcase/figures/")
    print("Best model: results/size_08/run_showcase/best/weights.pt")
    print("Training log: results/size_08/run_showcase/training_log.csv")
    print("=" * 60)
