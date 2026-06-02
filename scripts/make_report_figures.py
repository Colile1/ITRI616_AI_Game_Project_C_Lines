"""Regenerate the three report figures for run_pts_002 from its CSV logs.

Usage:
    python scripts/make_report_figures.py

Writes into results/size_08/run_pts_002/figures/:
    elo_curve.png, benchmark_vs_alphabeta.png, combined_progress.png
"""

from __future__ import annotations
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_DIR = Path("results/size_08/run_pts_002")
FIG_DIR = RUN_DIR / "figures"

BLUE = "#2563eb"; GREEN = "#059669"; RED = "#dc2626"; PURPLE = "#7c3aed"; GREY = "#6b7280"


def _style(ax) -> None:
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _load_training() -> dict[str, np.ndarray]:
    games, wr_rand, wr_heur, elo = [], [], [], []
    with (RUN_DIR / "training_log.csv").open() as f:
        for row in csv.DictReader(f):
            games.append(int(row["game"]))
            wr_rand.append(float(row["win_rate_vs_random"]))
            wr_heur.append(float(row["win_rate_vs_heuristic"]))
            elo.append(float(row["elo_rating"]))
    return {"games": np.array(games), "wr_rand": np.array(wr_rand),
            "wr_heur": np.array(wr_heur), "elo": np.array(elo)}


def _load_benchmark() -> dict[str, np.ndarray]:
    games, wr = [], []
    with (RUN_DIR / "benchmark_log.csv").open() as f:
        for row in csv.DictReader(f):
            games.append(int(row["training_game"]))
            wr.append(float(row["win_rate"]))
    return {"games": np.array(games), "wr": np.array(wr)}


def make_elo(t: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t["games"], t["elo"], color=PURPLE, lw=2.2)
    ax.fill_between(t["games"], t["elo"].min(), t["elo"], color=PURPLE, alpha=0.08)
    ax.set_title("Agent skill (Elo) rises monotonically with self-play experience",
                 fontsize=12, weight="bold")
    ax.set_xlabel("Training games"); ax.set_ylabel("Elo rating")
    _style(ax); fig.tight_layout()
    fig.savefig(FIG_DIR / "elo_curve.png", dpi=130); plt.close(fig)


def make_benchmark(b: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(b["games"], b["wr"], color=GREEN, lw=1.6, alpha=0.9,
            label="Win rate vs Alpha-Beta d4 (32 games/check)")
    k = 5
    ma = np.convolve(b["wr"], np.ones(k) / k, mode="valid")
    ax.plot(b["games"][k - 1:], ma, color=RED, lw=2.4, label=f"{k}-check moving average")
    ax.axhline(0.5, color=GREY, ls="--", lw=1, alpha=0.7)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("From losing every game to beating a 4-ply search",
                 fontsize=12, weight="bold")
    ax.set_xlabel("Training games"); ax.set_ylabel("Win rate vs Alpha-Beta depth-4")
    ax.legend(frameon=False, fontsize=9, loc="center right")
    _style(ax); fig.tight_layout()
    fig.savefig(FIG_DIR / "benchmark_vs_alphabeta.png", dpi=130); plt.close(fig)


def make_combined(t: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(t["games"], t["wr_rand"], color=BLUE, lw=1.8, label="Win rate vs Random")
    ax.plot(t["games"], t["wr_heur"], color=GREEN, lw=1.8, label="Win rate vs Heuristic")
    ax.set_xlabel("Training games"); ax.set_ylabel("Win rate"); ax.set_ylim(0, 1.05)
    ax2 = ax.twinx()
    ax2.plot(t["games"], t["elo"], color=PURPLE, lw=2.0, ls=(0, (4, 2)),
             label="Elo (right axis)")
    ax2.set_ylabel("Elo rating", color=PURPLE); ax2.tick_params(axis="y", colors=PURPLE)
    ax.spines["top"].set_visible(False); ax2.spines["top"].set_visible(False)
    ax.grid(True, alpha=0.25); ax.set_axisbelow(True)
    l1, la1 = ax.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, frameon=False, fontsize=9, loc="lower right")
    ax.set_title("C_lines 8x8 points-full - performance improves with experience",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "combined_progress.png", dpi=130); plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t = _load_training()
    b = _load_benchmark()
    make_elo(t)
    make_benchmark(b)
    make_combined(t)
    print("Figures written to:", FIG_DIR)
    print(f"Elo: {t['elo'][0]:.0f} -> {t['elo'][-1]:.0f}")
    print(f"WR vs random: {t['wr_rand'][0]:.0%} -> {t['wr_rand'][-1]:.0%}")
    print(f"WR vs heuristic: {t['wr_heur'][0]:.0%} -> {t['wr_heur'][-1]:.0%}")
    print(f"Benchmark WR: {b['wr'][0]:.0%} -> {b['wr'][-1]:.0%}")


if __name__ == "__main__":
    main()
