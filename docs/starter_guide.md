# Starter Guide — C_lines AI (ITRI616)

This guide takes you from a fresh clone to a working game, trained agent, and rendered figures. It assumes Python 3.11+ and a working `pip`.

---

## 1. Install

```bash
git clone <repo-url> ITRI616_AI_Game_Project_Flat_4_in_Row
cd ITRI616_AI_Game_Project_Flat_4_in_Row

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

A GPU is optional. Training defaults are sized for a typical student-laptop CPU.

## 2. Run the test suite

```bash
python -m pytest tests/ -v
```

All unit and integration tests must pass before training begins. Expected runtime: ~30 seconds on CPU.

## 3. Play (hot-seat, no training required)

The UI works as soon as the game engine is implemented; it does not depend on any trained snapshot for hot-seat play.

```bash
python -m src.ui.app
```

From the main menu choose **Hot-seat**, then board size, then game mode. Two human players share the keyboard and mouse.

## 4. Train an agent (one board size)

```bash
# Quick smoke test (50 games, takes ~1 min on CPU)
python -m src.training.train --games 50 --size 8

# Real run (10_000 games, takes 4–8 hours on CPU for size 8)
python -m src.training.train --games 10000 --size 8
```

The training script:

* Writes per-episode rows to `results/logs/training_log.csv`.
* Freezes a snapshot every `SNAPSHOT_INTERVAL` games into `models/size_NN/gen_NNN/`.
* Updates `models/registry.json` after each snapshot.
* Generates the five PNG figures in `results/figures/` when the run completes.

To train multiple board sizes, run the command above with `--size 9`, `--size 10`, `--size 11`, `--size 12` (sequentially or in parallel terminals).

## 5. Play against a trained agent

```bash
python -m src.ui.app
```

From the main menu choose **Play vs AI**, then board size, then game mode, then a level (snapshot). The level-select screen reads `models/registry.json` and shows one card per available snapshot for the chosen board size.

## 6. Inspect the results

* `results/figures/win_rate.png` — primary "learning curve" plot.
* `results/figures/reward_curve.png` — cumulative reward per episode.
* `results/figures/episode_length.png` — game length over training.
* `results/figures/epsilon_decay.png` — exploration schedule.
* `results/figures/loss_curve.png` — TD loss.
* `results/logs/training_log.csv` — raw per-episode log; the figures are derived from this file.

## 7. Component-level testing

```bash
# A single unit test
python -m pytest tests/test_scoring.py::test_horizontal_5line -v

# All engine tests
python -m pytest tests/test_rules.py tests/test_scoring.py -v

# Integration tests only
python -m pytest tests/integration/ -v
```

## 8. Common gotchas

* **PyGame fonts on Linux** — if Inter is not installed, the UI falls back to `pygame.font.SysFont(None, …)` automatically. No error, just a different visual font.
* **PyTorch CPU-only install** — the default `pip install torch` works. If you need a smaller wheel, use the `+cpu` build from `https://download.pytorch.org/whl/torch_stable.html`.
* **`models/registry.json` is empty** — that is normal before training. The level-select screen will show an empty grid and a hint to train first.
* **Disk space** — a full training run across all five board sizes uses ~500 MB. Drop `SNAPSHOT_INTERVAL` or `MAX_POOL_SIZE` in `config.py` to reduce.
* **CTRL-C during training** — the script flushes the current row of `training_log.csv` and writes a partial snapshot if one is in progress; otherwise existing snapshots remain valid.

## 9. Where things live

| You want to … | Look in … |
|---------------|-----------|
| Change a hyperparameter | `src/config.py` |
| Change the scoring schedule | `src/config.py` → `SCORE_FOR_LENGTH` |
| Change the network architecture | `src/training/network.py` |
| Change the UI theme | `src/ui/theme.py` |
| Add a new agent type | `src/agents/` |
| Read the rules in detail | `docs/deliverables/in_md_format/03_game_description.md` |
| Read the design rationale | `docs/deliverables/in_md_format/02_ml_methods_research.md` and `04_implementation_plan.md` |

## 10. Reproducing the report

The `docs/report.md` numbers are produced from `results/logs/training_log.csv` and `results/figures/*.png` of a single training run per board size. The recipe is:

```bash
python -m src.training.train --games 10000 --size 8
# (repeat for 9, 10, 11, 12 if you have the time)
python -m src.evaluation.plots          # regenerates figures from the CSV
```

Then edit `docs/report.md` and fill in the numbers from `training_log.csv` (the head and tail rows tell most of the story).
