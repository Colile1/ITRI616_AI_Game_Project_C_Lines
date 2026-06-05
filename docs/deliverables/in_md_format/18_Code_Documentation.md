# C_lines — Code Documentation

**Author:** *----* *S----* · ITRI 616
**Date:** 2026-06-02
**Companion to:** `17_Project_Report_FINAL.md`
**Scope:** how to install and run the code, how to reproduce the experiments, and a map of which code does what (game, learning algorithm, data collection, analysis). All examples use the reported configuration: **8×8 board, points-until-full**.

---

## 1. Requirements

- **Python 3.11+**
- Packages (in `requirements.txt`): `pygame`, `torch`, `numpy`, `matplotlib`, `pytest`
- No GPU required — the project runs on CPU.

## 2. Installation

From the project root:

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

Verify the install by running the test suite:

```bash
python -m pytest tests/ -q
```

A green run confirms the rules engine, environment, scoring, Elo, snapshotting, and a training smoke test all pass.

## 3. Running the code

### 3.1 Play the game (GUI)

```bash
python -m src.ui.app
```

This opens the PyGame window. From the menus choose the **8×8** board and **points-until-full** mode, then the **Level Select** screen, which lists the trained snapshots (`gen_001` … `gen_010`) as difficulty cards — read live from the model registry. Select a level to play against that checkpoint, or play hot-seat (human vs human) without any trained model.

### 3.2 Evaluate a trained snapshot (numeric, for the demo)

```bash
python -m scripts.demo_eval --gen 10 --games 50
# include the slower alpha-beta benchmark:
python -m scripts.demo_eval --gen 10 --games 16 --alphabeta
```

Prints the final snapshot's win rate against Random, Heuristic, and (optionally) Alpha-Beta depth-4. Greedy play (ε = 0) is used so the result reflects the learned policy.

### 3.3 Train an agent

```bash
python -m src.training.train --games 10000 --size 8 --benchmark alphabeta_d4
```

This is the exact command that produced the reported run. It trains for 10,000 self-play games on the 8×8 board in points-until-full mode, benchmarking against alpha-beta depth-4, and writes all logs, snapshots, and figures (see §5). A full run takes several hours on CPU (~450–620 games/hour).

Useful flags:

| Flag | Meaning |
|---|---|
| `--games N` | Number of training games |
| `--size 8` | Board size (8×8) |
| `--benchmark alphabeta_d4` | Periodic benchmark opponent |
| `--run-id NAME` | Name the run's output folder |
| `--no-augment` | Disable D4 symmetry augmentation (ablation) |
| `--resume` | Continue from the latest snapshot of `--run-id` |

## 4. Reproducing the experiments in the report

Every figure and table in `17_Project_Report_FINAL.md` comes from the `run_pts_002` outputs, which are committed in the repository. To reproduce them:

**(a) Regenerate the analysis figures from the logged data** (fast, no training):

```bash
python scripts/make_report_figures.py        # writes elo_curve / benchmark_vs_alphabeta / combined_progress
```

(If you prefer the built-in plotting, `src/evaluation/plots.py` also generates per-run figures.)

**(b) Reproduce the numbers live** from the trained model:

```bash
python -m scripts.demo_eval --gen 10 --games 50
```

**(c) Reproduce the whole run from scratch** (hours):

```bash
python -m src.training.train --games 10000 --size 8 --benchmark alphabeta_d4 --run-id reproduce_pts
```

The new run writes to `results/size_08/reproduce_pts/` and `models/size_08/reproduce_pts/`, mirroring the structure of the reported run.

## 5. Output files (data collection)

A training run writes a complete, traceable audit trail:

| Path | Contents |
|---|---|
| `results/size_08/run_pts_002/training_log.csv` | Per-100-game metrics: epsilon, learning rate, loss, win rate vs random/heuristic, **Elo**, elapsed time |
| `results/size_08/run_pts_002/benchmark_log.csv` | Each alpha-beta benchmark check: win rate, mean score margin, games played |
| `results/size_08/run_pts_002/game_log.csv` | Every individual game's result |
| `results/size_08/run_pts_002/best/weights.pt` | Best model seen during the run |
| `results/size_08/run_pts_002/figures/` | Generated plots |
| `models/size_08/run_pts_002/gen_NNN/weights.pt` | Frozen snapshot weights (difficulty levels) |
| `models/size_08/run_pts_002/registry.json` | Snapshot metadata: games trained, win rates, Elo, difficulty band |

## 6. Code map — which module does what

The code is organised as Python packages with single-responsibility modules under `src/`.

### 6.1 Code for the game

| Module | Responsibility |
|---|---|
| `src/engine/board.py` | Board state and placement bookkeeping |
| `src/engine/rules.py` | Legal-move and terminal-condition logic |
| `src/engine/scoring.py` | Line detection and score / threat computation |
| `src/game/env.py` | Gym-style environment: `reset()`, `step()`, reward shaping, terminal checks |
| `src/game/encoding.py` | State → 10-channel tensor; legal-move masks; action indexing |
| `src/ui/` | PyGame interface: menus, board view, level select, replay viewer |

### 6.2 Code for the learning algorithm

| Module | Responsibility |
|---|---|
| `src/training/network.py` | Residual Q-network (and a legacy plain CNN) |
| `src/agents/dqn_agent.py` | DQN agent: ε-greedy action, **negamax Double-DQN update**, Huber loss, target sync |
| `src/training/self_play.py` | Episode execution; transition collection; epsilon schedule |
| `src/training/replay_buffer.py` | Experience replay with importance weighting and D4 augmentation at sample time |
| `src/training/symmetry.py` | Dihedral (D4) transforms for augmentation |
| `src/training/snapshot.py` | Freeze / clone agents into registered snapshots |
| `src/training/train.py` | The training loop / CLI that ties it all together |
| `src/agents/heuristic_agent.py`, `alphabeta_agent.py`, `random_agent.py` | Opponents (warm-up, benchmark, baseline) |

### 6.3 Code for data collection

| Module | Responsibility |
|---|---|
| `src/training/game_logger.py` | Writes `game_log.csv` |
| `src/training/benchmark_logger.py` | Runs benchmark checks, writes `benchmark_log.csv` |
| `src/versioning/registry.py`, `metadata.py` | Snapshot registry and metadata (Elo, win rates, bands) |
| `training_log.csv` writer in `train.py` | Per-eval metrics |

### 6.4 Code for analysis

| Module | Responsibility |
|---|---|
| `src/evaluation/evaluator.py` | Plays N games of agent vs opponent; returns win/draw/loss rates |
| `src/evaluation/elo.py` | Elo rating computation |
| `src/evaluation/plots.py` | Per-run training / benchmark figures |
| `scripts/make_report_figures.py` | Regenerates the three report figures from the CSV logs |
| `scripts/demo_eval.py` | Loads a snapshot and prints live win rates (used in the demo) |

### 6.5 Configuration

`src/config.py` is the single source of truth for all hyperparameters: board size, scoring table, reward scales, DQN hyperparameters (learning rate, gamma, replay capacity, batch size, target-sync interval), epsilon schedule, snapshot/eval intervals, network architecture, and the difficulty bands. Changing an experiment means changing values here, not hunting through the code.

## 7. Entry points at a glance

```bash
python -m src.ui.app                                              # play / watch
python -m src.training.train --games 10000 --size 8 --benchmark alphabeta_d4   # train
python -m scripts.demo_eval --gen 10 --games 50                   # evaluate a snapshot
python -m pytest tests/ -q                                        # tests
```
