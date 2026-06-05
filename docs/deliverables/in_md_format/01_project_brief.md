# C_lines — Project Brief

**Module:** ITRI 616 — Artificial Intelligence 1
**Institution:** North-West University (NWU)
**Student / Author:** *----* *S----*
**Date:** 2026-05-28
**Project type:** Semester mini-project — well-posed learning problem for an intelligent game agent
**Status:** Planning complete, build pending

---

## 1. One-sentence summary

C_lines is an original two-player strategy board game played on a flat 8x8 to 12x12 grid where players freely place pieces anywhere on the board and score by forming uninterrupted lines of length 3 to 8 in any of the four directions (horizontal, vertical, and both diagonals), packaged with a PyGame graphical interface, a hot-seat multiplayer mode, and a learning AI opponent whose successive training snapshots are saved as discrete, immutable "levels" the player can challenge in single-player mode.

## 2. Problem statement

The ITRI 616 mini-project requires a Southern-African-themed, not-yet-digitised game with a well-posed learning agent that demonstrably improves with experience. Existing four-in-a-row variants (Connect Four, Gomoku, Renju, Five-in-a-row) all impose constraints C_lines does not: they either use gravity-fed columns, restrict the placement direction, hard-cap the line length at 5, or use forbidden-move rules. C_lines is an original variant that combines free placement on a square grid with a tunable board size, a variable line-length scoring schedule (3 to 8 in a row), and a custom piece-removal tie-break, producing a game with no published rule-set, no existing dataset, and therefore no opportunity for the agent to learn from human data — it must learn from self-play alone.

## 3. Project objectives

The project will deliver, in order of importance:

1. A playable PyGame implementation of C_lines supporting hot-seat multiplayer and human-vs-AI single-player on board sizes 8x8 through 12x12, with both game modes (first-to-four and points-until-full), the custom three-round piece-removal tie-break, and a frosted-glass / sheer / light-blue UI theme.
2. A learning AI agent (Deep Q-Network with self-play — motivation in deliverable 02) that improves measurably with training experience, evaluated by win rate against the random and heuristic baselines and Elo against past snapshots.
3. A versioning system that freezes the agent at periodic checkpoints (every `SNAPSHOT_INTERVAL` games), writes each snapshot's weights plus a metadata record (date trained, games-trained count, training-history summary, win-rates, Elo) to `models/<version_id>/`, and exposes the snapshots as selectable difficulty levels in the single-player menu — frozen snapshots never train further once registered.
4. The full ITRI 616 deliverable set: formal TEP definitions (Mitchell, Chapter 1), training curves (`win_rate`, `reward`, `episode_length`, `epsilon_decay`, `loss`), critical analysis, and a starter guide.

## 4. Scope

### In scope

The complete game engine (board, rules, terminal detection, scoring with per-direction uniqueness, custom tie-break); both game modes; five separately-trained AI agent families (one per board size from 8x8 to 12x12); a frosted-glass PyGame UI with main menu, board-size picker, mode picker, level-select screen, hotseat play, vs-AI play, end-of-game summary, replay viewer; a self-play training loop with snapshotting; an evaluation harness producing the five required figures; and full project documentation (plan, todo, log, report, TEP definitions, starter guide, README).

### Out of scope

Online networked multiplayer, mobile or web ports, tournament/match-making, persistent player accounts, custom board shapes (only square N×N for N in 8..12), AI move explanations or saliency maps, multi-language localisation beyond English, and training a single generalised agent that plays all board sizes (deliberately rejected during clarification in favour of five separate agent families for cleaner network shapes and faster convergence).

### Explicit non-goals

Beating a state-of-the-art Connect-Four solver, replicating AlphaZero's level of play, or producing a publishable research contribution. The goal is a well-posed learning problem that demonstrably improves with experience — not state-of-the-art game-AI performance.

## 5. Stakeholders

| Stakeholder | Role | Interest |
|-------------|------|----------|
| *----* *S----* | Student, sole developer | Pass the module with a strong mark and a maintainable code base reusable for future ITRI projects |
| ITRI 616 module lecturer | Examiner | TEP rigour, learning-curve evidence, code quality, critical analysis |
| External markers (if any) | Examiner | Same as lecturer |
| Future *----* (post-submission) | Re-user | Reuse the game-design skeleton and snapshot system for a follow-on ITRI 626 or M.Sc. project |

## 6. Success criteria

The project succeeds if **all** of the following are true on submission day:

1. `python -m pytest tests/ -v` passes from a fresh clone with `pip install -r requirements.txt`.
2. A single coherent training run produces a `training_log.csv` and a `results/figures/` folder containing the five required PNGs; the `win_rate.png` curve trends upward (not necessarily monotonically) and ends materially above the 50% random-baseline line for at least one board size.
3. The trained agent at the latest snapshot beats the `RandomAgent` baseline at >= 75% win rate and the `HeuristicAgent` at >= 55% win rate on its trained board size (over >= 200 evaluation games).
4. At least 5 distinct snapshot versions exist in `models/registry.json` per board size, each with valid metadata, and the level-select screen can load any of them as a frozen opponent.
5. The PyGame UI runs without crashing for a full game in both modes, on at least the 10x10 board, with the frosted-glass theme visibly applied.
6. The report addresses both critical-analysis questions ("Did performance improve with experience?" and "Was the problem well-posed?") with reference to actual numbers from the training log.

## 7. TEP frame (formal definitions live in `docs/tep_definitions.md`)

* **Task (T):** Sequential decision-making — at each turn, choose a placement (and in tie-break sub-mode, choose a removal) on the C_lines board so as to maximise expected end-of-game reward.
* **Experience (E):** Self-play simulated episodes, with the early `WARMUP_GAMES` episodes played against `RandomAgent` and the remainder played against a mix of the current agent and a pool of past frozen snapshots; all transitions stored in a replay buffer.
* **Performance (P):** Primary — win rate against `RandomAgent` measured every `EVAL_INTERVAL` games on the same fixed seed set. Secondary — Elo against the snapshot pool, average reward per episode, average episode length.

## 8. Deliverables (artefacts produced by this project)

The deliverables fall into four buckets:

**Code** — `src/` (engine, game, agents, training, evaluation, versioning, UI), `tests/`, `requirements.txt`. **Models** — `models/registry.json` and the per-snapshot folders containing weights and metadata. **Results** — `results/figures/` (five PNGs minimum) and `results/logs/training_log.csv`. **Documentation** — `docs/plan.md`, `docs/todo.md`, `docs/log.md`, `docs/report.md`, `docs/tep_definitions.md`, `docs/starter_guide.md`, the seven deliverable Markdown files in `docs/deliverables/in_md_format/`, and the root `README.md`.

## 9. Constraints and assumptions

The implementation is constrained to Python with PyGame for graphics and PyTorch for the neural network (confirmed during clarification). The course explicitly allows supervised learning, a simple neural network, or reinforcement learning — the chosen approach (DQN with self-play) is RL. Hardware: a typical student laptop with no dedicated GPU is assumed — the network architecture and training budget are sized accordingly. The agent must train within roughly 4 to 8 hours of wall-clock time per board size on CPU; if a GPU is available, this is upside, not a requirement. Pieces are immutable once placed except during the tie-break sub-mode. The board always starts empty. Player 1 always moves first.

## 10. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| DQN fails to learn on 12x12 due to action-space size (144 cells) | Medium | High | Start with 8x8 (64 actions); validate learning curve before scaling up; budget more training games for larger boards |
| First-player advantage dominates and inflates win rates artificially | Medium | Medium | Always randomise which side the agent plays during evaluation; report per-side win rate; consider a pie/swap rule (see deliverable 03) |
| Mode 2 reward shaping is too sparse (most reward arrives only at terminal) | Medium | Medium | Use per-step delta-score reward shaping (`STEP_REWARD_SCALE` in config) so each line completed gives immediate signal |
| Snapshot pool grows unbounded and slows self-play | Low | Low | Cap pool at `MAX_POOL_SIZE`; evict oldest or sample uniformly from a fixed-size reservoir |
| Frosted-glass effect is hard to achieve in raw PyGame | Medium | Low | Pre-render blurred surfaces with `pygame.transform.gaussian_blur` (Pygame-CE) or PIL fallback; cache them; degrade gracefully to flat semi-transparent panels if blur is slow |
| Tie-break removal logic introduces bugs (multi-round, recomputation, draw fallback) | Medium | Medium | Isolate in `engine/rules.py` as a pure function `apply_tiebreak_removal()`; exhaustive unit tests covering 0/1/2/3 rounds and draw fallback |
| Five separate agent families inflate training time five-fold | High | Medium | Train smallest (8x8) first as a vertical slice; only scale up board sizes after the smallest succeeds; remaining sizes can be left as future work if time-boxed |

## 11. Timeline (indicative)

The phases follow the build order in `04_implementation_plan.md`. At the current rate (one to two evenings per phase) the project completes inside three weeks of focused part-time work; padding is included for the training run, which is wall-clock-bound rather than developer-bound.

| Week | Phases | Outputs |
|------|--------|---------|
| 1 | Phases 1 to 4: engine, rules, env, baseline agents | Hot-seat hotseat play works in CLI; baselines pass `test_agent.py` |
| 2 | Phases 5 to 9: network, replay buffer, DQN agent, self-play, snapshot, evaluation | First full 8x8 training run produces all five figures |
| 3 | Phases 10 to 11: PyGame UI with frosted-glass theme, level-select, full training of remaining board sizes, write-up | Submission-ready repo |

## 12. Out of brief, into deliverable 02 onwards

The remaining deliverables in this folder go into the depth this brief intentionally avoids:

* `02_ml_methods_research.md` — comparative survey of learning algorithms with motivated recommendation.
* `03_game_description.md` — exhaustive ruleset for C_lines plus suggested rule improvements.
* `04_implementation_plan.md` — phased build order, module map, encoding, hyperparameters.
* `05_ui_ux_spec.md` — frosted-glass theme, screen flows, accessibility.
* `06_versioning_spec.md` — snapshot metadata schema and level-select mapping.
* `07_test_plan.md` — unit, integration, training, and UI tests.

---

*End of project brief.*
