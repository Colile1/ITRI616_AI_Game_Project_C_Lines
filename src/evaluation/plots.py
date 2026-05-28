"""Generate the five required training figures from training_log.csv."""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _rolling(values: np.ndarray, window: int = 50) -> np.ndarray:
    if len(values) < window:
        window = max(1, len(values))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def generate_all_plots(
    log_csv_path: Path | str,
    board_size: int,
    out_dir: Path | str,
) -> list[Path]:
    """Read training_log CSV and write 5 PNG figures to out_dir."""
    import csv

    log_csv_path = Path(log_csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    games, epsilons, losses, win_rates, ep_lens = [], [], [], [], []
    with open(log_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                games.append(int(row["game"]))
                epsilons.append(float(row["epsilon"]))
                losses.append(float(row["loss"]))
                win_rates.append(float(row["win_rate_vs_random"]))
                ep_lens.append(float(row["mean_ep_len"]))
            except (KeyError, ValueError):
                continue

    games = np.array(games)
    tag = f"Board {board_size}×{board_size}"
    produced: list[Path] = []

    def _save(fig_name: str) -> Path:
        p = out_dir / fig_name
        plt.savefig(p, dpi=120, bbox_inches="tight")
        plt.close()
        produced.append(p)
        return p

    # 1. win_rate.png
    plt.figure()
    plt.plot(games, win_rates, alpha=0.3, label="raw")
    plt.plot(games, _rolling(np.array(win_rates)), label="rolling mean")
    plt.axhline(0.5, color="grey", linestyle="--", label="50% baseline")
    plt.xlabel("Training game")
    plt.ylabel("Win rate vs Random")
    plt.title(f"Win Rate — {tag}")
    plt.legend()
    _save("win_rate.png")

    # 2. reward_curve.png (using win_rate as proxy for cumulative reward)
    plt.figure()
    plt.plot(games, win_rates, alpha=0.3, label="win rate (proxy)")
    plt.plot(games, _rolling(np.array(win_rates)), label="rolling mean")
    plt.xlabel("Training game")
    plt.ylabel("Win rate (reward proxy)")
    plt.title(f"Reward Curve — {tag}")
    plt.legend()
    _save("reward_curve.png")

    # 3. episode_length.png
    plt.figure()
    plt.plot(games, ep_lens, alpha=0.3, label="raw")
    plt.plot(games, _rolling(np.array(ep_lens)), label="rolling mean")
    plt.xlabel("Training game")
    plt.ylabel("Transitions per episode")
    plt.title(f"Episode Length — {tag}")
    plt.legend()
    _save("episode_length.png")

    # 4. epsilon_decay.png
    plt.figure()
    plt.plot(games, epsilons)
    plt.xlabel("Training game")
    plt.ylabel("Epsilon")
    plt.title(f"Epsilon Decay — {tag}")
    _save("epsilon_decay.png")

    # 5. loss_curve.png
    plt.figure()
    plt.plot(games, losses, alpha=0.3, label="raw")
    plt.plot(games, _rolling(np.array(losses)), label="rolling mean")
    plt.xlabel("Gradient step (game index proxy)")
    plt.ylabel("MSE TD loss")
    plt.title(f"Loss Curve — {tag}")
    plt.legend()
    _save("loss_curve.png")

    return produced
