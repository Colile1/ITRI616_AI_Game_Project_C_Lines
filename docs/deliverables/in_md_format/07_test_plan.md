# C_lines — Test Plan

**Author:** *----* *S----*
**Module:** ITRI 616
**Date:** 2026-05-28
**Status:** Plan locked; tests to be written alongside their target phase

---

## 1. Approach

Tests are written **as each phase is built**, not after. The build-order table in the implementation plan defines the acceptance test for each phase; this document expands those acceptance tests into a per-file inventory plus a small set of integration tests, training smoke tests, and a manual-UI checklist. Every test file lives in `tests/` and is discoverable by `pytest`. The full suite must pass from a fresh clone after `pip install -r requirements.txt`.

The C_lines test suite is split into three layers:

* **Unit tests** — every pure function in `engine/` and `evaluation/elo.py`, plus the agent classes and the snapshot round-trip.
* **Integration tests** — the env + agent loop, the training-loop smoke test, the registry write-read cycle.
* **Manual tests** — the PyGame UI (no automated visual testing in scope).

## 2. Unit tests

### 2.1 `tests/test_rules.py` — engine correctness

| Test ID | Description |
|---------|-------------|
| R-01 | `setup_board(N)` returns an N×N board of `EMPTY` for every N in {8..12} |
| R-02 | `apply_placement` on a legal cell switches its value to the placing player and leaves all others unchanged |
| R-03 | `apply_placement` raises `IllegalMoveError` on a cell that is already occupied |
| R-04 | `apply_placement` raises on (r, c) with r or c out of range |
| R-05 | `legal_placements(board, player)` returns exactly the empty cells |
| R-06 | Mode 1: `is_terminal_mode1` returns True iff a 4-line (length >= 4) exists for the just-placed player |
| R-07 | Mode 1: a 5-line counts as a win (length-4 win is `>=`, not `==`) |
| R-08 | Mode 1: a fully empty board is not terminal |
| R-09 | Mode 2: `is_terminal_mode2` returns True iff the board is full |
| R-10 | `get_winner` after `is_terminal_mode2` returns the higher-scoring player, or `None` if tied |
| R-11 | `apply_removal(board, opponent_cell)` empties the cell |
| R-12 | `apply_removal` raises on an empty cell or own-player cell |
| R-13 | `apply_tiebreak_removal` runs at most TIEBREAK_MAX_ROUNDS rounds and exits when scores differ |
| R-14 | `apply_tiebreak_removal` returns `winner=None` after 3 rounds with equal scores |

### 2.2 `tests/test_scoring.py` — line counting and score schedule

| Test ID | Description |
|---------|-------------|
| S-01 | A horizontal run of length 3 scores exactly 0.25 |
| S-02 | A horizontal run of length 4 scores exactly 1.0 |
| S-03 | A horizontal run of length 5 scores exactly 2.0 (NOT counted as 4+5 overlapping) |
| S-04 | A horizontal run of length 8 scores exactly 5.0 |
| S-05 | A horizontal run of length 10 scores exactly 5.0 (length cap at 8) |
| S-06 | A run of length 2 scores 0.0 |
| S-07 | Vertical and both diagonal runs score identically to horizontal at every length |
| S-08 | Two separate runs in the same row (e.g. `XXXX.XXXXXX`) score independently as 1.0 + 3.0 = 4.0 |
| S-09 | A fulcrum piece sitting on lines in 4 directions contributes points in all 4 directions |
| S-10 | Boundary lines (against board edge) score the same as interior lines |
| S-11 | Mixed-player rows score only the active player's runs and ignore opponent's pieces (which act as interruptions) |
| S-12 | `score_board(board, player)` is the sum of all player runs across all 4 directions, unique per direction |
| S-13 | An empty board scores 0.0 for both players |

### 2.3 `tests/test_env.py` — env contract

| Test ID | Description |
|---------|-------------|
| E-01 | `reset()` returns a tensor of shape `(C, N, N)` with dtype float32 |
| E-02 | `legal_mask()` after reset has shape `(N*N,)` and at least one True |
| E-03 | `legal_mask()` after reset has False at every cell currently occupied |
| E-04 | `step(legal_action)` returns `(obs, reward, done, info)` without raising |
| E-05 | After `step`, the turn counter increments by exactly 1 |
| E-06 | After `step`, the active player flips (P1 -> P2 -> P1 -> ...) |
| E-07 | Mode 1: an episode that completes a 4-line sets `done=True` with `info['winner']` correct |
| E-08 | Mode 2: an episode that fills the board sets `done=True` with the higher-scoring player as winner |
| E-09 | `step(illegal_action)` raises `IllegalMoveError` (or returns a sentinel reward, per implementation choice — pick one and stick with it) |
| E-10 | Game terminates within `N*N` total steps in Mode 2 (no infinite loops) |
| E-11 | `state_to_tensor` round-trip via `tensor_to_state` (if implemented) preserves piece positions |

### 2.4 `tests/test_agent.py` — agent contracts

| Test ID | Description |
|---------|-------------|
| A-01 | `RandomAgent.select_action(obs, mask)` returns an integer i such that `mask[i] is True` |
| A-02 | Over 1000 calls with multiple legal actions, `RandomAgent` selects at least 3 distinct actions |
| A-03 | `HeuristicAgent` completes its own 4-line when available (1-ply test) |
| A-04 | `HeuristicAgent` blocks the opponent's 4-line when no own-win is available (1-ply test) |
| A-05 | `DQNAgent(eps=0).select_action(obs, mask)` returns a legal index |
| A-06 | `DQNAgent(eps=1).select_action(obs, mask)` returns a legal index (epsilon=1 must still mask) |
| A-07 | `DQNAgent.set_epsilon(x)` updates the internal epsilon to x |
| A-08 | Two `DQNAgent`s with same seed produce identical Q-values on the same input |

### 2.5 `tests/test_snapshot.py` — versioning round-trip

| Test ID | Description |
|---------|-------------|
| V-01 | `snapshot.freeze(agent, games_trained, stats)` writes `weights.pt` and `metadata.json` |
| V-02 | `snapshot.load_snapshot(size, version_id)` returns a DQNAgent whose Q-values match the frozen agent (atol 1e-5) |
| V-03 | `registry.register` appends to the correct board-size bucket |
| V-04 | `registry.list_by_size` returns entries sorted by `games_trained` ascending |
| V-05 | `registry.get` raises `KeyError` on unknown version_id |
| V-06 | Re-running `freeze` for the same training run picks the next sequential gen_NNN ID |
| V-07 | Atomic write: simulating a crash mid-write leaves the prior `registry.json` intact |
| V-08 | `evict_oldest(size, keep_n=3)` removes all but the newest 3 snapshots and returns the evicted IDs |
| V-09 | Loaded snapshot plays a single legal action on a fresh obs |

### 2.6 `tests/test_elo.py` — rating arithmetic

| Test ID | Description |
|---------|-------------|
| L-01 | `expected_score(R, R)` returns 0.5 |
| L-02 | `expected_score(R+400, R)` returns ≈ 0.909 (textbook Elo) |
| L-03 | `update_elo(R, R, win)` increases the winner's rating |
| L-04 | `update_elo` is symmetric: the loser loses exactly what the winner gains |
| L-05 | K-factor is read from config and applied correctly |

## 3. Integration tests

### 3.1 `tests/integration/test_env_agent_loop.py`

| Test ID | Description |
|---------|-------------|
| I-01 | `RandomAgent` vs `RandomAgent`, 50 games on 8×8, all terminate within `N*N` steps |
| I-02 | `RandomAgent` vs `HeuristicAgent`, 200 games — heuristic wins >= 60% of decided games |
| I-03 | `DQNAgent(eps=1)` vs `RandomAgent` plays 50 full games without raising |

### 3.2 `tests/integration/test_train_smoke.py`

| Test ID | Description |
|---------|-------------|
| T-01 | `python -m src.training.train --games 50 --size 8` exits with code 0 |
| T-02 | The smoke run writes a non-empty `training_log.csv` |
| T-03 | The smoke run produces at least one snapshot in `models/size_08/` |
| T-04 | The snapshot loads successfully via `snapshot.load_snapshot` |

### 3.3 `tests/integration/test_evaluator.py`

| Test ID | Description |
|---------|-------------|
| EV-01 | `evaluate(RandomAgent(), RandomAgent(), 100)` returns win-rate ≈ 0.5 (±0.10) |
| EV-02 | `evaluate(HeuristicAgent(), RandomAgent(), 100)` returns win-rate >= 0.65 |
| EV-03 | `evaluate` populates all dict keys: win_rate, draw_rate, mean_reward, mean_length |

## 4. Training validation tests (post-run, not part of CI)

These run after a full training pass to confirm the learning curve is sensible. They are diagnostic, not pass/fail in CI.

| Check | Acceptance threshold |
|-------|----------------------|
| Final `win_rate_vs_random` on the trained board size | >= 0.75 |
| Final `win_rate_vs_heuristic` on the trained board size | >= 0.55 |
| Rolling-200 win-rate is higher in the final 1000 games than in the first 1000 | strictly greater |
| Loss curve drops by >= 50% from its peak rolling mean to its final rolling mean | yes |
| At least 5 snapshots registered in `models/size_NN/` | yes |
| All five required PNGs present and non-empty | yes |

## 5. Manual UI tests (Phase 10 acceptance)

Performed by hand once the UI is built. Each item is a pass/fail checkbox the developer ticks before moving on.

```
[ ] Main menu opens; all four buttons are visible and clickable
[ ] Board-size picker cycles 8 -> 9 -> 10 -> 11 -> 12 -> 8 with arrow keys
[ ] Mode picker toggles between First-to-Four and Points-Until-Full
[ ] Level-select shows all snapshots for the chosen board size; latest is flagged
[ ] In-game: clicking a legal empty cell places a piece in the current player's colour
[ ] In-game: clicking an illegal cell does nothing (no crash, no place)
[ ] In-game: turn counter and scores update after every placement
[ ] In-game: last-move ring is visible for the most recent placement
[ ] Mode 1: completing a 4-line ends the game with the correct winner overlay
[ ] Mode 2: filling the board ends the game; scores match a manual recount
[ ] Mode 2 tie-break: a constructed tie correctly enters tiebreak removal
[ ] Tie-break draw: 3 rounds of equal scores produce the draw overlay
[ ] ESC during a game pauses; "Resign" works
[ ] Replay viewer: arrow keys step through moves; final state matches end-of-game
[ ] Settings: toggles persist across screens within a session
[ ] Frosted-glass theme is visibly applied (panels have blur tint, not flat fills)
[ ] Window resizes correctly when the board size changes from the menu
[ ] Reduce-motion setting disables fade/spring animations
```

## 6. Running the suite

From the project root:

```bash
# Unit + integration tests, verbose
python -m pytest tests/ -v

# Just the unit tests (fast)
python -m pytest tests/test_*.py -v

# Just the integration tests
python -m pytest tests/integration/ -v

# A single test by ID
python -m pytest tests/test_scoring.py::test_horizontal_5line -v
```

CI: a single `pytest tests/ -v` invocation must return 0. The training-validation thresholds in Section 4 are checked manually after the full run by inspecting `results/figures/` and the tail of `training_log.csv`.

## 7. Coverage targets

The project does not enforce a numeric coverage threshold, but `engine/rules.py`, `engine/scoring.py`, and `versioning/registry.py` should all approach 100% line coverage given how cheap they are to exercise. Network and replay-buffer code is exercised by the training smoke test rather than by isolated unit tests, which is sufficient for the project's scale.

---

*End of test plan.*
