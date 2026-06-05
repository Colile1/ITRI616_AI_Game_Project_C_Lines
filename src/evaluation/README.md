# src/evaluation/

Measuring agent skill and generating training figures.

| File | Purpose |
|------|---------|
| `evaluator.py` | `evaluate(agent, opponent, env, n_games)` -> dict with win_rate, mean_ep_len |
| `elo.py` | `update_elo()`, `expected_score()` — standard Elo rating system |
| `plots.py` | `generate_all_plots()`, `generate_benchmark_plot()` -> writes PNG files |

### Regenerate figures
```python
from src.evaluation.plots import generate_all_plots, generate_benchmark_plot
generate_all_plots("results/size_08/run_ftf_004/training_log.csv", 8, "results/size_08/run_ftf_004/figures")
generate_benchmark_plot("results/size_08/run_ftf_004/benchmark_log.csv", 8, "results/size_08/run_ftf_004/figures")
```
Produces: `win_rate.png`, `reward_curve.png`, `episode_length.png`, `epsilon_decay.png`, `loss_curve.png`, `benchmark_curve.png`

### Elo anchors used in training
- RandomAgent: 600 Elo
- HeuristicAgent: 900 Elo
- AlphaBetaAgent: 1200 Elo

### Rolling mean
`_rolling()` uses a cumsum-based centred window that shrinks near edges rather than zero-padding, preventing the artificial dips that appeared at the start and end of earlier training curves.
