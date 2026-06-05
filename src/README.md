# src/

All source code for C_lines.

| Sub-package | Purpose |
|-------------|---------|
| `agents/` | All agent implementations (DQN, AlphaBeta, MCTS, Heuristic, Random) |
| `engine/` | Pure game logic — board, rules, scoring (no ML dependencies) |
| `game/` | Gym-style `GameEnv`, 10-channel state encoding |
| `training/` | Training loop, replay buffer, self-play, network, snapshots |
| `evaluation/` | Elo tracker, head-to-head evaluator, plot generation |
| `ui/` | PyGame interface — menus, board view, level select |
| `versioning/` | Snapshot metadata, model registry, v1->v2 migration |
| `config.py` | **Single source of truth** for all hyperparameters and file paths |

`config.py` is the first file to read when tuning training.
Set `UI_BOARD_SIZES = [8, 9, ...]` in config to add board sizes back to the UI once agents are trained.
