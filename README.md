# C_lines

**ITRI 616 mini-project — original open-placement four-in-a-row variant with a self-play DQN agent.**

Author: Colile Sibanda · NWU · 2026

C_lines is a two-player strategy board game on a flat NxN grid (N in {8, 9, 10, 11, 12}) where players freely place pieces anywhere and score by forming lines of length 3 to 8 in any direction. Two modes (first-to-four and points-until-full), a custom no-draw tie-break, a frosted-glass PyGame UI, and a Deep Q-Network agent that learns to play by self-play and is snapshotted at intervals — each snapshot becomes a selectable difficulty level.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
# Play (hot-seat is available without any trained snapshot)
python -m src.ui.app

# Train (one board size, 10 000 games, ~4–8 h on CPU)
python -m src.training.train --games 10000 --size 8

# Test
python -m pytest tests/ -v
```

## Documentation

* Project brief: [`docs/deliverables/in_md_format/01_project_brief.md`](docs/deliverables/in_md_format/01_project_brief.md)
* ML methods research: [`docs/deliverables/in_md_format/02_ml_methods_research.md`](docs/deliverables/in_md_format/02_ml_methods_research.md)
* Game description: [`docs/deliverables/in_md_format/03_game_description.md`](docs/deliverables/in_md_format/03_game_description.md)
* Implementation plan: [`docs/deliverables/in_md_format/04_implementation_plan.md`](docs/deliverables/in_md_format/04_implementation_plan.md)
* UI/UX spec: [`docs/deliverables/in_md_format/05_ui_ux_spec.md`](docs/deliverables/in_md_format/05_ui_ux_spec.md)
* Versioning spec: [`docs/deliverables/in_md_format/06_versioning_spec.md`](docs/deliverables/in_md_format/06_versioning_spec.md)
* Test plan: [`docs/deliverables/in_md_format/07_test_plan.md`](docs/deliverables/in_md_format/07_test_plan.md)
* TEP definitions: [`docs/tep_definitions.md`](docs/tep_definitions.md)
* Starter guide: [`docs/starter_guide.md`](docs/starter_guide.md)
* Report: [`docs/report.md`](docs/report.md)

## Licence

MIT.
