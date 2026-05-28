"""Generate the five required training figures from training_log.csv."""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _rolling(values: np.ndarray, window: int = 10) -> np.ndarray:
    if len(values) < window:
        window = max(1, len(values))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def _load_last_run(log_csv_path: Path) -> dict[str, np.ndarray]:
    """Read CSV and return only the last contiguous run (final game=0 onward)."""
    import csv

    all_rows: list[dict] = []
    with open(log_csv_path) as f:
        for row in csv.DictReader(f):
            try:
                int(row["game"])
                all_rows.append(row)
            except (KeyError, ValueError):
                continue

    if not all_rows:
        return {}

    # Find final run start
    last_start = 0
    for i, row in enumerate(all_rows):
        if int(row["game"]) == 0:
            last_start = i
    all_rows = all_rows[last_start:]

    def _col(key: str, default: float = 0.0) -> np.ndarray:
        return np.array([float(r.get(key, default) or default) for r in all_rows])

    return {
        "games":              _col("game"),
        "epsilons":           _col("epsilon"),
        "losses":             _col("loss"),
        "win_rate_vs_random": _col("win_rate_vs_random"),
        "win_rate_vs_heuristic": _col("win_rate_vs_heuristic"),
        "ep_lens":            _col("mean_ep_len"),
        "lr":                 _col("lr"),
    }


def generate_all_plots(
    log_csv_path: Path | str,
    board_size: int,
    out_dir: Path | str,
) -> list[Path]:
    """Read training_log CSV and write 5 PNG figures to out_dir."""
    log_csv_path = Path(log_csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_last_run(log_csv_path)
    if not data:
        return []

    games = data["games"]
    tag = f"Board {board_size}×{board_size}"
    produced: list[Path] = []

    def _save(fig_name: str) -> Path:
        p = out_dir / fig_name
        plt.savefig(p, dpi=120, bbox_inches="tight")
        plt.close()
        produced.append(p)
        return p

    # ------------------------------------------------------------------
    # 1. win_rate.png  — random AND heuristic on same chart
    # ------------------------------------------------------------------
    wr_r = data["win_rate_vs_random"]
    wr_h = data["win_rate_vs_heuristic"]
    plt.figure(figsize=(9, 5))
    plt.plot(games, wr_r,               alpha=0.25, color="steelblue")
    plt.plot(games, _rolling(wr_r),     color="steelblue",  label="vs Random (rolling)")
    plt.plot(games, wr_h,               alpha=0.25, color="darkorange")
    plt.plot(games, _rolling(wr_h),     color="darkorange", label="vs Heuristic (rolling)")
    plt.axhline(0.50, color="grey",    linestyle="--", linewidth=0.8, label="50% baseline")
    plt.axhline(0.75, color="green",   linestyle=":",  linewidth=0.8, label="75% target")
    plt.ylim(0, 1.05)
    plt.xlabel("Training game")
    plt.ylabel("Win rate")
    plt.title(f"Win Rate — {tag}")
    plt.legend()
    plt.tight_layout()
    _save("win_rate.png")

    # ------------------------------------------------------------------
    # 2. reward_curve.png
    # ------------------------------------------------------------------
    plt.figure(figsize=(9, 4))
    plt.plot(games, wr_r, alpha=0.25, color="steelblue")
    plt.plot(games, _rolling(wr_r), color="steelblue", label="WR vs Random (reward proxy)")
    plt.xlabel("Training game")
    plt.ylabel("Win rate (reward proxy)")
    plt.title(f"Reward Curve — {tag}")
    plt.legend()
    plt.tight_layout()
    _save("reward_curve.png")

    # ------------------------------------------------------------------
    # 3. episode_length.png
    # ------------------------------------------------------------------
    ep = data["ep_lens"]
    plt.figure(figsize=(9, 4))
    plt.plot(games, ep, alpha=0.25, color="mediumseagreen")
    plt.plot(games, _rolling(ep), color="mediumseagreen", label="rolling mean")
    plt.xlabel("Training game")
    plt.ylabel("Transitions per episode")
    plt.title(f"Episode Length — {tag}")
    plt.legend()
    plt.tight_layout()
    _save("episode_length.png")

    # ------------------------------------------------------------------
    # 4. epsilon_decay.png
    # ------------------------------------------------------------------
    plt.figure(figsize=(9, 4))
    plt.plot(games, data["epsilons"], color="slateblue", label="ε")
    plt.xlabel("Training game")
    plt.ylabel("Epsilon")
    plt.title(f"Epsilon Decay — {tag}")
    plt.legend()
    plt.tight_layout()
    _save("epsilon_decay.png")

    # ------------------------------------------------------------------
    # 5. loss_curve.png
    # ------------------------------------------------------------------
    loss = data["losses"]
    plt.figure(figsize=(9, 4))
    plt.plot(games, loss, alpha=0.25, color="tomato")
    plt.plot(games, _rolling(loss), color="tomato", label="rolling mean")
    plt.xlabel("Gradient-step proxy (game index)")
    plt.ylabel("MSE TD loss")
    plt.title(f"Loss Curve — {tag}")
    plt.legend()
    plt.tight_layout()
    _save("loss_curve.png")

    return produced


def generate_benchmark_plot(
    benchmark_csv_path: Path | str,
    board_size: int,
    out_dir: Path | str,
    rolling_window: int = 100,
) -> Path | None:
    """Read benchmark_log.csv and write benchmark_curve.png.

    x-axis: training game index
    y-axis: rolling win rate vs benchmark (one line per benchmark name)
    """
    import csv

    benchmark_csv_path = Path(benchmark_csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not benchmark_csv_path.exists():
        return None

    rows: list[dict] = []
    with open(benchmark_csv_path) as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "game":   int(row["training_game"]),
                    "result": float(row["benchmark_result"]),
                    "name":   row["benchmark_name"],
                })
            except (KeyError, ValueError):
                continue

    if not rows:
        return None

    # Group by benchmark name
    by_name: dict[str, tuple[list[int], list[float]]] = {}
    for r in rows:
        name = r["name"]
        if name not in by_name:
            by_name[name] = ([], [])
        by_name[name][0].append(r["game"])
        by_name[name][1].append(r["result"])

    plt.figure(figsize=(10, 5))
    colors = ["royalblue", "darkorange", "seagreen", "tomato"]
    for i, (name, (games, results)) in enumerate(by_name.items()):
        g = np.array(games)
        r = np.array(results)
        color = colors[i % len(colors)]
        plt.scatter(g, r, alpha=0.10, s=6, color=color)
        plt.plot(g, _rolling(r, rolling_window), color=color, label=f"vs {name} (rolling-{rolling_window})")

    plt.axhline(0.50, color="grey", linestyle="--", linewidth=0.8, label="50% baseline")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Training game")
    plt.ylabel(f"Win rate (rolling {rolling_window})")
    plt.title(f"Benchmark Curve — Board {board_size}×{board_size}")
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / "benchmark_curve.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path
