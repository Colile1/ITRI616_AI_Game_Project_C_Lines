"""Standalone progress plotter — safe to run while training is active.

Reads the existing CSV files and regenerates all plots from whatever data
exists at the moment you run it. Does not touch the running training process.

Usage (in a separate terminal):
    python plot_progress.py --size 8 --run run_002
    python plot_progress.py --size 8 --run run_002 --rolling 5
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"


def _rolling(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    w = min(window, len(values))
    kernel = np.ones(w) / w
    return np.convolve(values, kernel, mode="same")


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def plot_benchmark(bm_path: Path, out_dir: Path, rolling: int, board_size: int) -> Path | None:
    rows = _read_csv(bm_path)
    if not rows:
        print(f"  [skip] benchmark_log.csv is empty or missing: {bm_path}")
        return None

    games, results, score_diffs = [], [], []
    for r in rows:
        try:
            games.append(int(r["training_game"]))
            results.append(float(r["benchmark_result"]))
            score_diffs.append(float(r["benchmark_score_diff"]))
        except (KeyError, ValueError):
            continue

    if not games:
        return None

    g = np.array(games)
    r = np.array(results)
    sd = np.array(score_diffs)

    bm_name = rows[0].get("benchmark_name", "benchmark")
    win_pct  = r.mean() * 100
    roll_r   = _rolling(r, rolling)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # --- Win/loss ---
    ax = axes[0]
    ax.scatter(g, r, s=18, alpha=0.5, color="royalblue", label="result (1=win, 0=loss)")
    ax.plot(g, roll_r, color="royalblue", linewidth=2,
            label=f"rolling-{rolling} win rate")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Win rate vs benchmark")
    ax.set_title(
        f"Benchmark curve — {board_size}×{board_size}  vs {bm_name}\n"
        f"Games logged: {len(g)}   Overall win rate: {win_pct:.1f}%"
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)

    # --- Score difference ---
    ax2 = axes[1]
    ax2.scatter(g, sd, s=12, alpha=0.4, color="darkorange")
    ax2.plot(g, _rolling(sd, rolling), color="darkorange", linewidth=2,
             label=f"rolling-{rolling} score diff")
    ax2.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("Learner score − benchmark score")
    ax2.set_xlabel("Training game")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out = out_dir / "benchmark_curve_live.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  benchmark_curve_live.png  ({len(g)} data points, win rate={win_pct:.1f}%)")
    return out


def plot_training_log(log_path: Path, out_dir: Path, rolling: int, board_size: int) -> None:
    rows = _read_csv(log_path)
    if not rows:
        print(f"  [skip] training_log.csv empty: {log_path}")
        return

    games, wr_r, wr_h, losses, eps = [], [], [], [], []
    for r in rows:
        try:
            games.append(int(r["game"]))
            wr_r.append(float(r.get("win_rate_vs_random", 0) or 0))
            wr_h.append(float(r.get("win_rate_vs_heuristic", 0) or 0))
            losses.append(float(r.get("loss", 0) or 0))
            eps.append(float(r.get("epsilon", 0) or 0))
        except (KeyError, ValueError):
            continue

    if not games:
        return

    g = np.array(games)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    # Win rates
    axes[0].plot(g, np.array(wr_r), alpha=0.3, color="steelblue")
    axes[0].plot(g, _rolling(np.array(wr_r), rolling), color="steelblue",
                 linewidth=2, label=f"vs Random (rolling-{rolling})")
    axes[0].plot(g, np.array(wr_h), alpha=0.3, color="darkorange")
    axes[0].plot(g, _rolling(np.array(wr_h), rolling), color="darkorange",
                 linewidth=2, label=f"vs Heuristic (rolling-{rolling})")
    axes[0].axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Win rate")
    axes[0].set_title(f"Training progress — {board_size}×{board_size}  ({len(g)} eval points)")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)

    # Loss
    axes[1].plot(g, np.array(losses), alpha=0.3, color="tomato")
    axes[1].plot(g, _rolling(np.array(losses), rolling), color="tomato",
                 linewidth=2, label=f"TD loss (rolling-{rolling})")
    axes[1].set_ylabel("MSE TD loss")
    axes[1].legend(fontsize=8)

    # Epsilon
    axes[2].plot(g, np.array(eps), color="slateblue", linewidth=1.5, label="ε")
    axes[2].set_ylabel("Epsilon")
    axes[2].set_xlabel("Training game")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    out = out_dir / "training_progress_live.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  training_progress_live.png  ({len(g)} eval checkpoints)")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot training progress from existing CSVs")
    p.add_argument("--size",    type=int, default=8)
    p.add_argument("--run",     type=str, default=None, help="e.g. run_002")
    p.add_argument("--rolling", type=int, default=5,
                   help="Rolling average window (default 5; use 3 for very sparse data)")
    args = p.parse_args()

    size_dir = RESULTS / f"size_{args.size:02d}"
    if args.run:
        run_dirs = [size_dir / args.run]
    else:
        if not size_dir.exists():
            print(f"No results for size {args.size}")
            return
        run_dirs = sorted(size_dir.iterdir())

    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        out_dir = run_dir / "figures"
        out_dir.mkdir(exist_ok=True)
        print(f"\n[{run_dir.name}]")
        plot_benchmark(run_dir / "benchmark_log.csv", out_dir, args.rolling, args.size)
        plot_training_log(run_dir / "training_log.csv", out_dir, args.rolling, args.size)

    print("\nDone. Open the figures/ folder to see the plots.")


if __name__ == "__main__":
    main()
