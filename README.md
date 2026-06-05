# C_lines — ITRI 616 Mini-Project

**Author:**  · North-West University · 2026

C_lines is a two-player strategy board game on a flat 8×8 grid where players freely place pieces and the first to form four in a row wins (*First-to-Four* mode). A Deep Q-Network (DQN) agent learns to play through self-play and is snapshotted at regular intervals — each snapshot becomes a selectable difficulty level in the UI.

---

## Quick Start

```powershell
# Install dependencies
pip install torch pygame matplotlib numpy

# Play vs trained agent
python -m src.ui.app

# Train a fresh agent (8x8, First-to-Four, 10,000 games)
python -m src.training.train --size 8 --mode first_to_four --games 10000 --benchmark alphabeta_d4

# Run the showcase training script (denser logging, parallel workers)
python train_showcase.py

# Run tests
python -m pytest tests/ -v
```

---

## Proven Results

| Metric | Game 0 | Game 7,000 | Game 9,000 |
|--------|--------|-----------|-----------|
| Episode length | 33.3 moves | 9.9 moves | **8.9 moves (−73%)** |
| Win rate vs random | 48.5% | **100%** | 100% |
| Win rate vs AlphaBeta-4 | 0% | 3% | **50%** |
| Elo rating | 790 | 744 | 783 |

---

## Project Structure

```
C_lines/
├── src/                    # All source code
│   ├── agents/             # DQN, AlphaBeta, MCTS, Heuristic, Random agents
│   ├── engine/             # Board, rules, scoring (pure functions, no ML)
│   ├── game/               # GameEnv (Gym-style), 10-channel state encoding
│   ├── training/           # Training loop, replay buffer, self-play, snapshots
│   ├── evaluation/         # Elo tracker, evaluator, plot generation
│   ├── ui/                 # PyGame interface, menus, level select, board view
│   ├── versioning/         # Snapshot metadata, model registry, migration
│   └── config.py           # Single source of truth for all hyperparameters
├── tests/                  # Unit + integration tests (72 tests, all passing)
├── scripts/                # Utility scripts (eval demo, figure generation)
├── docs/deliverables/      # Technical report, slides outline, project docs
├── results/size_08/        # Training logs, figures, best-model checkpoints
├── models/size_08/         # Snapshot weights and metadata per run
├── train_showcase.py       # One-command showcase training script
├── benchmark_workers.py    # Worker speedup benchmark
└── plot_progress.py        # Regenerate all training figures
```

---

## Training CLI Reference

```powershell
# Basic training
python -m src.training.train --size 8 --mode first_to_four --games 10000

# Resume a previous run
python -m src.training.train --size 8 --mode first_to_four --games 20000 --run-id run_ftf_004 --resume

# Resume with parallel workers (faster, uses multiple CPU cores)
python -m src.training.train --size 8 --mode first_to_four --games 20000 --run-id run_ftf_004 --resume --workers 4

# Use the best model from a previous run as the training opponent
python -m src.training.train --size 8 --mode first_to_four --games 20000 --run-id run_ftf_004 --resume --prev-best results/size_08/run_ftf_003/best/weights.pt --benchmark alphabeta_d4 --workers 4
```

Key flags:

| Flag | Description |
|------|-------------|
| `--size N` | Board size (8–12) |
| `--mode` | `first_to_four` or `points_full` |
| `--games N` | Total games from 0 (resume picks up from last snapshot) |
| `--run-id ID` | Run identifier; auto-assigned if omitted |
| `--resume` | Load latest snapshot and continue |
| `--prev-best PATH` | Frozen model used as static training opponent |
| `--workers N` | Parallel episode collection (4 recommended for single run) |
| `--benchmark NAME` | Fixed eval opponent e.g. `alphabeta_d4` |
| `--load-weights PATH` | Seed agent weights from a specific file |
| `--start-game N` | Override starting game counter |

---

## Regenerate Figures

```powershell
python -c "
from src.evaluation.plots import generate_all_plots, generate_benchmark_plot
generate_all_plots('results/size_08/run_ftf_004/training_log.csv', 8, 'results/size_08/run_ftf_004/figures')
generate_benchmark_plot('results/size_08/run_ftf_004/benchmark_log.csv', 8, 'results/size_08/run_ftf_004/figures')
"
```

---

## Deliverables

| Document | Location |
|----------|----------|
| Technical report | `docs/deliverables/technical_report.md` |
| Slides outline | `docs/deliverables/slides_outline.md` |
| run_ftf_004 analysis | `results/size_08/run_ftf_004/analysis_report.md` |

---

## Licence

MIT.
