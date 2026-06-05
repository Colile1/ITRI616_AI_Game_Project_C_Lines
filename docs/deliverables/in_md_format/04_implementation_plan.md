# C_lines — Implementation Plan

**Author:** *----* *S----*
**Module:** ITRI 616
**Date:** 2026-05-28
**Status:** Plan locked; execution phase pending

---

## 1. Overview

C_lines is implemented in Python 3.11+ using PyGame for the user interface, PyTorch for the learning agent, and a small set of scientific-Python libraries (NumPy, Matplotlib) for evaluation and plotting. The directory layout follows the existing `game_design_skeleton.md` exactly, with C_lines-specific adaptations to the engine, encoding, reward shaping, and UI theme. The agent is a Deep Q-Network trained by self-play with a snapshot pool, with five separately-trained agent families — one per board size in {8, 9, 10, 11, 12}.

The build follows a strict phase order (Section 4) so that each phase can be unit-tested in isolation before the next phase depends on it. The implementation is sized to run end-to-end on a CPU-only student laptop within 4 to 8 hours of wall-clock training per board size.

## 2. Module map

```
ITRI616_AI_Game_Project_Flat_4_in_Row/
├── src/
│   ├── __init__.py
│   ├── config.py                      # All tunable constants. Single source of truth.
│   │
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── board.py                   # Board dataclass; setup_board(N); clone_board()
│   │   ├── rules.py                   # PURE: legal_placements, apply_placement,
│   │   │                              #   apply_removal, score_board, get_winner,
│   │   │                              #   is_terminal_mode1, is_terminal_mode2,
│   │   │                              #   apply_tiebreak_removal, count_lines_for_player
│   │   └── scoring.py                 # Pure helpers for the line-counting algorithm
│   │
│   ├── game/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── encoding.py                # state_to_tensor; action_to_index;
│   │   │                              #   index_to_action; build_legal_mask
│   │   └── env.py                     # GameEnv: reset, step, legal_mask
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── base_agent.py              # ABC: select_action(obs, legal_mask) -> int
│   │   ├── random_agent.py            # Uniform over legal actions
│   │   ├── heuristic_agent.py         # Hand-coded: complete-own-4 > block-opp-4 > extend
│   │   └── dqn_agent.py               # epsilon-greedy DQN wrapper
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── network.py                 # DQNNetwork (Conv-based)
│   │   ├── replay_buffer.py           # ReplayBuffer.push / sample / __len__
│   │   ├── self_play.py               # run_episode; self_play_loop
│   │   ├── train.py                   # get_training_config; train(); main()
│   │   └── snapshot.py                # freeze; load_snapshot; clone_agent
│   │
│   ├── versioning/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── metadata.py                # SnapshotMetadata dataclass; save; load
│   │   └── registry.py                # registry.json: register, list, get, evict
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── evaluator.py               # evaluate(agent, opponent, n_games) -> dict
│   │   ├── elo.py                     # update_elo; expected_score
│   │   └── plots.py                   # generate_all_plots() -> 5 PNGs
│   │
│   └── ui/
│       ├── __init__.py
│       ├── README.md
│       ├── app.py                     # PyGame main loop; entry point
│       ├── theme.py                   # Frosted-glass colour tokens + surface helpers
│       ├── board_view.py              # render_board; render_sidebar
│       ├── menus.py                   # main_menu, board_size_picker, mode_picker
│       ├── level_select.py            # level_select_screen -> SnapshotMetadata
│       └── replay_view.py             # post-game move-by-move stepper (v1.1)
│
├── tests/
│   ├── __init__.py
│   ├── test_rules.py                  # engine correctness
│   ├── test_env.py                    # env contract
│   ├── test_agent.py                  # baselines + dqn legal-action
│   ├── test_snapshot.py               # versioning round-trip
│   └── test_scoring.py                # line-counting edge cases
│
├── models/
│   └── registry.json                  # [] initially
│
├── results/
│   ├── figures/
│   └── logs/
│
├── docs/
│   ├── plan.md
│   ├── todo.md
│   ├── log.md
│   ├── report.md
│   ├── starter_guide.md
│   ├── tep_definitions.md
│   └── deliverables/in_md_format/
│       ├── 01_project_brief.md
│       ├── 02_ml_methods_research.md
│       ├── 03_game_description.md
│       ├── 04_implementation_plan.md
│       ├── 05_ui_ux_spec.md
│       ├── 06_versioning_spec.md
│       └── 07_test_plan.md
│
├── requirements.txt
└── README.md
```

## 3. Coding conventions (recap, binding)

Every file in `src/` honours the conventions in the project `rules.md`: 120-line soft cap, single-responsibility, package `__init__.py` + `README.md`, snake_case, pure functions in `engine/` with zero I/O, input-wrapper and output-handler separation, file-level docstring, one-line public-function docstrings, `main()` orchestrates only. `config.py` is the sole home of constants. No code executes at import time.

## 4. Build phases

The build proceeds in eleven phases. Each phase has an acceptance test; do not advance until it passes.

| Phase | Component | Acceptance test |
|-------|-----------|-----------------|
| 1 | `config.py` + `engine/board.py` | `setup_board(N)` returns empty N×N board for N in {8..12} |
| 2 | `engine/rules.py` + `engine/scoring.py` | `tests/test_rules.py` and `tests/test_scoring.py` pass |
| 3 | `game/encoding.py` + `game/env.py` | `tests/test_env.py` passes; legal-mask shape correct |
| 4 | `agents/base_agent.py`, `random_agent.py`, `heuristic_agent.py` | `tests/test_agent.py` passes for baselines |
| 5 | `training/network.py` + `replay_buffer.py` | Forward pass returns shape `(B, N*N)`; buffer round-trips |
| 6 | `agents/dqn_agent.py` | DQNAgent always selects legal action at eps=0 and eps=1 |
| 7 | `training/self_play.py` + `train.py` + `snapshot.py` | `python -m src.training.train --games 10 --size 8` exits cleanly with a CSV |
| 8 | `versioning/` | `tests/test_snapshot.py` passes (round-trip, registry add/list/get) |
| 9 | `evaluation/` | `evaluate(dqn, random_agent, 50)` returns dict with `win_rate` >= 0 |
| 10 | `ui/` (theme, menus, board_view, level_select, app) | Hot-seat game playable end-to-end on 10×10 |
| 11 | Full training run (per board size) + plots | `results/figures/` contains five PNGs; `training_log.csv` populated |

Build order is strict; phases 5–9 in particular must precede phase 10 because the UI loads agents from the registry and the registry is empty until training has produced snapshots.

## 5. State encoding

For a board of side N, the state is encoded as a float32 tensor of shape `(C, N, N)` with C = 6 channels:

| ch | Content                                              | Range     |
|----|------------------------------------------------------|-----------|
| 0  | Current player's pieces (binary mask)                | {0.0, 1.0}|
| 1  | Opponent's pieces (binary mask)                      | {0.0, 1.0}|
| 2  | Empty cells (binary mask)                            | {0.0, 1.0}|
| 3  | Current player's open-3 threats (cells that would complete a 3-line; binary) | {0.0, 1.0}|
| 4  | Opponent's open-3 threats (mirror of ch3)            | {0.0, 1.0}|
| 5  | Turn-number normaliser (constant plane = turn/N²)    | [0.0, 1.0]|

Channels 3 and 4 are an inductive bias — they pre-compute the "what would happen if I placed here" features for the most strategically important sub-pattern (3-in-row threats), which lets the convolutional layers focus capacity elsewhere. They are computed by the same scanner that scores lines, restricted to L = 3, on the union of (existing pieces + the candidate cell).

Channel 5 supplies the network with a notion of how late in the game it is, which matters because the optimal policy shifts as fewer cells remain.

## 6. Action space

A single, flat integer index per cell:

```
ACTION_SPACE_SIZE = N * N

action_to_index((r, c)) -> r * N + c
index_to_action(i)      -> (i // N, i % N)
```

There is no second action type during regular play (placement is the only move). During the Mode-2 tie-break, the same flat index is reused but the legal mask is constructed differently (opponent-only cells), and the engine routes the action to `apply_removal` instead of `apply_placement`. This keeps the network output layer fixed at N×N for a given board-size agent.

`build_legal_mask(board, player, phase) -> np.ndarray(bool, shape=(N*N,))`:

* `phase == "placement"` — True where `board[r, c] == EMPTY`.
* `phase == "tiebreak_removal"` — True where `board[r, c] == OPPONENT_OF(player)`.

The mask is passed into both the agent's action-selection (sets illegal Q-values to −∞ before argmax) and the network's training loss (illegal Q-values are not used as targets).

## 7. Reward function

```
PER STEP (Mode 1):
    r_t = +STEP_REWARD_SCALE     if the placement completed a line >= 4 of current player
        =  0.0                   otherwise

PER STEP (Mode 2):
    delta_score = (own_score_after - own_score_before)
                - (opp_score_after - opp_score_before)
    r_t = delta_score * STEP_REWARD_SCALE     # delta-score shaping

TERMINAL (both modes):
    r_T += WIN_REWARD     if current player wins
    r_T += LOSS_REWARD    if current player loses
    r_T += 0.0            on draw (Mode 1 only — Mode 2 draws are vanishingly rare with tie-break)

OPTIONAL (off by default):
    +0.01 per opponent-line-broken in Mode 2 tie-break removal phase.
```

Defaults: `STEP_REWARD_SCALE = 0.05`, `WIN_REWARD = +1.0`, `LOSS_REWARD = -1.0`. Per-step shaping is mode-aware because Mode 1 has no scoring during play.

## 8. Network architecture

For a board of side N, with C = 6 input channels:

```
Input  : (B, 6, N, N)

Conv2d( 6 -> 32, kernel=3, padding=1)  ->  ReLU
Conv2d(32 -> 64, kernel=3, padding=1)  ->  ReLU
Conv2d(64 -> 64, kernel=3, padding=1)  ->  ReLU
Flatten
Linear(64 * N * N -> 256)              ->  ReLU
Linear(256 -> N * N)                       # Q-values, one per cell

Output : (B, N*N)
```

Total parameter count grows mainly through the flattened linear layer. For N=12: `64 * 144 * 256 + 256 * 144 ≈ 2.4M` parameters — well within CPU-feasible bounds.

The same architecture is used for all five board-size agents, parameterised by N at construction time.

## 9. Hyperparameter defaults

```python
# Learning
LR                  = 1e-3
GAMMA               = 0.99
REPLAY_CAPACITY     = 100_000
BATCH_SIZE          = 64
TARGET_SYNC_STEPS   = 1_000
GRADIENT_CLIP       = 10.0

# Exploration
EPS_START           = 1.0
EPS_END             = 0.05
EPS_DECAY_GAMES     = 5_000      # linear schedule over games (not steps)

# Curriculum
TRAINING_GAMES      = 10_000     # per board size
WARMUP_GAMES        = 1_000      # vs RandomAgent before self-play
SELF_PLAY_MIX_PROB  = 0.5        # P(opponent is a snapshot vs current agent)

# Snapshotting
SNAPSHOT_INTERVAL   = 1_000      # games between snapshots
MAX_POOL_SIZE       = 20         # cap on snapshots used as opponents

# Evaluation
EVAL_INTERVAL       = 500        # games between eval passes
EVAL_GAMES          = 200        # games per eval pass
EVAL_SEEDS          = list(range(EVAL_GAMES))   # fixed seeds for comparability

# Reward shaping
STEP_REWARD_SCALE   = 0.05
WIN_REWARD          = 1.0
LOSS_REWARD         = -1.0
```

All of the above live in `src/config.py`. They are the experimental defaults; the lecturer may adjust `TRAINING_GAMES` for the assessed run.

## 10. Self-play curriculum

```
GAMES 0 to WARMUP_GAMES:
    DQN agent  vs  RandomAgent
    Purpose: bootstrap a usable policy without the moving-target problem of
             pure self-play from random initialisation.

GAMES WARMUP_GAMES to TRAINING_GAMES:
    With probability SELF_PLAY_MIX_PROB, opponent is a snapshot drawn
        uniformly from the snapshot pool (max MAX_POOL_SIZE).
    Otherwise, opponent is a frozen copy of the current agent.
    Each episode randomises whether the DQN plays as P1 or P2 (to mitigate
        first-mover bias on the win-rate metric).
```

Every `SNAPSHOT_INTERVAL` games the current agent is frozen, evaluated against the random and heuristic baselines, registered in `registry.json`, and added to the pool.

## 11. Training loop pseudo-code

```
init dqn_agent (online + target nets)
init replay_buffer
init snapshot_pool

for game_idx in 0..TRAINING_GAMES:
    pick opponent per the curriculum (random | self | pool snapshot)
    play one episode (alternate which side is P1)
    push all transitions into replay_buffer

    for k in 0..NUM_GRAD_STEPS_PER_EPISODE:
        if len(replay_buffer) < BATCH_SIZE: break
        batch = replay_buffer.sample(BATCH_SIZE)
        loss  = td_loss(online, target, batch, gamma=GAMMA, legal_mask)
        loss.backward()
        clip_grad_norm_(online.params, GRADIENT_CLIP)
        optimizer.step()

        if global_step % TARGET_SYNC_STEPS == 0:
            target.load_state_dict(online.state_dict())

    epsilon = linear_decay(EPS_START, EPS_END, game_idx, EPS_DECAY_GAMES)
    dqn_agent.set_epsilon(epsilon)

    if game_idx % EVAL_INTERVAL == 0:
        eval_stats = evaluator.evaluate(dqn_agent, RandomAgent(), EVAL_GAMES)
        log_row(training_log.csv, game_idx, epsilon, loss, eval_stats)

    if game_idx % SNAPSHOT_INTERVAL == 0:
        meta = freeze(dqn_agent, game_idx, eval_stats)
        registry.register(meta)
        snapshot_pool.add(meta)
        evict_oldest_if_full(snapshot_pool, MAX_POOL_SIZE)

after loop: generate_all_plots()
```

`NUM_GRAD_STEPS_PER_EPISODE` is set so total gradient steps stay roughly proportional to total transitions; a default of 4 is a sensible starting value.

## 12. Required output figures

After training, `evaluation/plots.py` produces the five PNGs in `results/figures/`:

| File                  | X axis         | Y axis                          | Notes |
|-----------------------|----------------|---------------------------------|-------|
| `win_rate.png`        | Training game  | Win rate (rolling 200) vs random| 50% baseline horizontal line |
| `reward_curve.png`    | Training game  | Cumulative reward per episode   | Rolling mean overlay |
| `episode_length.png`  | Training game  | Transitions per episode         | Rolling mean overlay |
| `epsilon_decay.png`   | Training game  | Epsilon value                   | Linear schedule |
| `loss_curve.png`      | Gradient step  | MSE TD loss                     | Rolling mean overlay |

All figures: labelled axes, title naming the board size, dpi=120, saved as PNG. The `report.md` references the actual numbers from `training_log.csv`.

## 13. Versioning integration

See `06_versioning_spec.md` for the full schema. In short: every snapshot writes `models/<size>/<gen_id>/weights.pt` and `metadata.json`; the global index lives at `models/registry.json` and is loaded by the UI's level-select screen, which filters by board size to show the right agent family.

## 14. Frosted-glass UI summary

See `05_ui_ux_spec.md` for the full design. Key implementation notes for the plan:

* Use `pygame.SRCALPHA` surfaces for every panel.
* Pre-render blurred backdrops once at start-up; cache them per resolution.
* For blur, use `pygame.transform.gaussian_blur` (PyGame-CE 2.5+); fallback to a downsample-upsample trick if running on classic PyGame.
* Panel borders: 1-px stroke at 30% white alpha to suggest the glass edge.
* Player accents: P1 = light cyan/blue `(127, 223, 255)`, P2 = warm amber `(255, 184, 92)`.
* Background: deep navy `(10, 20, 40)` with a soft radial gradient to a slightly lighter centre.

## 15. requirements.txt (proposed)

```
pygame>=2.5.0
torch>=2.2.0
numpy>=1.26.0
matplotlib>=3.8.0
pytest>=8.0.0
```

No PyTorch GPU build is required — the spec runs on CPU. If a CUDA-enabled torch is installed, training will use it automatically through the standard `device = "cuda" if torch.cuda.is_available() else "cpu"` pattern in `dqn_agent.py`.

## 16. Time budget per phase (indicative, one developer)

| Phase | Effort | Notes |
|-------|--------|-------|
| 1     | 1 hr   | Trivial scaffolding |
| 2     | 4 hr   | Line-counting + tie-break logic is the only non-trivial pure-function block |
| 3     | 2 hr   | Encoding plus env contract |
| 4     | 2 hr   | Random + heuristic baselines |
| 5     | 2 hr   | Network + buffer |
| 6     | 1 hr   | DQN wrapper |
| 7     | 3 hr   | Self-play + train + snapshot |
| 8     | 1 hr   | Versioning metadata + registry |
| 9     | 2 hr   | Evaluator + Elo + plots |
| 10    | 6 hr   | UI; frosted-glass theme is the long pole |
| 11    | wall-clock 4–8 hr per board size; engineer-time minimal | Five board sizes possible if time permits |

Total engineer time: roughly 25 hours of focused work to "code complete", plus the training wall-clock.

---

*End of implementation plan.*
