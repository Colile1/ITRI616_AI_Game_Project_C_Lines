# C_lines — ITRI 616 Mini-Project

**Author:**  *___* · North-West University · 2026

C_lines is a two-player strategy board game on a flat 8×8 grid where players freely place pieces and the first to form four in a row wins (*First-to-Four* mode). A Deep Q-Network (DQN) agent learns to play through self-play and is snapshotted at regular intervals, each snapshot becomes a selectable difficulty level in the UI.

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


 
## 1. Requirements
•	Python 3.11+ (the project runs on CPU; no GPU is required).
•	Packages (in requirements.txt): pygame, torch, numpy, matplotlib, pytest.

## 2. Installation
From the project root, create a virtual environment and install the dependencies:
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS / Linux
pip install -r requirements.txt
Verify the install by running the test suite; a green run confirms the rules engine, environment, scoring, Elo, snapshotting, and a training smoke test all pass:
python -m pytest tests/ -q

## 3. Running the Code
### 3.1 Play the game (GUI)
python -m src.ui.app
This opens the PyGame window. Choose the 8x8 board and points-until-full mode, then the Level Select screen, which lists the trained snapshots (gen_001 ... gen_010) as difficulty cards read live from the model registry. Select a level to play against that checkpoint, or play hot-seat (human vs human) with no trained model.
## 3.2 Evaluate a trained snapshot (numeric)
python -m scripts.demo_eval --gen 10 --games 50

python -m scripts.demo_eval --gen 10 --games 16 --alphabeta   # include benchmark

Prints the final snapshot's win rate against Random, Heuristic, and (optionally) Alpha-Beta depth-4. Greedy play (epsilon = 0) is used so the result reflects the learned policy.

### 3.3 Train an agent
This is the exact command that produced the reported run. It trains for 10,000 self-play games on the 8x8 board in points-until-full mode, benchmarking against alpha-beta depth-4, and writes all logs, snapshots, and figures. A full run takes several hours on CPU (about 345-620 games/hour).
python -m src.training.train --games 10000 --size 8 --benchmark alphabeta_d4

Useful flags:
Flag	Meaning
--games N	Number of training games

--size 8	Board size (8x8)

--benchmark alphabeta_d4	Periodic benchmark opponent

--run-id NAME	Name the run's output folder

--no-augment	Disable D4 symmetry augmentation (ablation)

--resume	Continue from the latest snapshot of --run-id

## 4. Reproducing the Experiments
Every figure and table in the Technical Report comes from the run_pts_002 outputs, which are committed in the repository. To reproduce them:
•	(a) Regenerate the analysis figures from the logged data (fast, no training):
python scripts/make_report_figures.py
•	(b) Reproduce the numbers live from the trained model:
python -m scripts.demo_eval --gen 10 --games 50
•	(c) Reproduce the whole run from scratch (hours):
python -m src.training.train --games 10000 --size 8 --benchmark alphabeta_d4 --run-id reproduce_pts
A new run writes to results/size_08/reproduce_pts/ and models/size_08/reproduce_pts/, mirroring the structure of the reported run.

## 5. Output Files (Data Collection)
A training run writes a complete, traceable audit trail:
Path	Contents
results/size_08/<run>/training_log.csv	Per-100-game metrics: epsilon, learning rate, loss, win rate vs random/heuristic, Elo, elapsed time

results/size_08/<run>/benchmark_log.csv	Each alpha-beta benchmark check: win rate, mean score margin, games played

results/size_08/<run>/game_log.csv	Every individual game's result

results/size_08/<run>/best/weights.pt	Best model seen during the run

results/size_08/<run>/figures/	Generated plots

models/size_08/<run>/gen_NNN/weights.pt	Frozen snapshot weights (difficulty levels)

models/size_08/<run>/registry.json	Snapshot metadata: games trained, win rates, Elo, difficulty band

## 6. Code Map — Which Module Does What
The code is organised as Python packages with single-responsibility modules under src/.
### 6.1 The game
Module	Responsibility
src/engine/board.py	Board state and placement bookkeeping
src/engine/rules.py	Legal-move and terminal-condition logic
src/engine/scoring.py	Line detection and score / threat computation
src/game/env.py	Gym-style environment: reset(), step(), reward shaping, terminal checks
src/game/encoding.py	State -> 10-channel tensor; legal-move masks; action indexing
src/ui/	PyGame interface: menus, board view, level select, replay viewer
### 6.2 The learning algorithm
Module	Responsibility
src/training/network.py	Residual Q-network (and a legacy plain CNN)

src/agents/dqn_agent.py	DQN agent: epsilon-greedy action, negamax Double-DQN update, Huber loss, target sync

src/training/self_play.py	Episode execution; transition collection; epsilon schedule

src/training/replay_buffer.py	Experience replay with importance weighting and D4 augmentation at sample time

src/training/symmetry.py	Dihedral (D4) transforms for augmentation

src/training/snapshot.py	Freeze / clone agents into registered snapshots

src/training/train.py	The training loop / CLI that ties it together

src/agents/{heuristic,alphabeta,random}_agent.py	Opponents (warm-up, benchmark, baseline)

src/agents/mcts_agent.py	Monte-Carlo Tree Search agent (inference)

### 6.3 Data collection and analysis
Module	Responsibility
src/training/game_logger.py	Writes game_log.csv

src/training/benchmark_logger.py	Runs benchmark checks, writes benchmark_log.csv

src/versioning/registry.py, metadata.py	Snapshot registry and metadata (Elo, win rates, bands)

src/evaluation/evaluator.py	Plays N games of agent vs opponent; returns win/draw/loss rates

src/evaluation/elo.py	Elo rating computation

src/evaluation/plots.py	Per-run training / benchmark figures

scripts/make_report_figures.py	Regenerates the report figures from the CSV logs

scripts/demo_eval.py	Loads a snapshot and prints live win rates (used in the demo)
### 6.4 Configuration
src/config.py is the single source of truth for all hyperparameters: board size, scoring table, reward scales, DQN hyperparameters (learning rate, gamma, replay capacity, batch size, target-sync interval), epsilon schedule, snapshot/eval intervals, network architecture, and the difficulty bands. Changing an experiment means changing values here, not hunting through the code.
## 7. Entry Points at a Glance
python -m src.ui.app                                            # play / watch

python -m src.training.train --games 10000 --size 8 --benchmark alphabeta_d4   # train

python -m scripts.demo_eval --gen 10 --games 50                 # evaluate a snapshot

python -m pytest tests/ -q                                      # tests

