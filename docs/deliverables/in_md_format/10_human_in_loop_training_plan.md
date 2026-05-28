# Human-in-the-Loop Training and Fixed-Benchmark Plan

**Author:** Colile Sibanda
**Module:** ITRI 616
**Date:** 2026-05-28
**Status:** Plan locked, implementation pending
**Related:** `09_algorithm_upgrade_plan.md`, `11_unified_upgrade_plan.md`

---

## 1. Goal

Two coupled capabilities are added to the project:

1. **Switchable, scriptable training opponents.** The user can compose a training session like "50 games agent-vs-agent, 10 games me-vs-agent, 100 games agent-vs-snapshot-pool, 5 games me-vs-agent" and the system executes the schedule end-to-end, training on transitions from all phases.
2. **A fixed benchmark agent measured after every training game.** Regardless of who the training opponent was, after each training game the agent plays one (or more) game against an unchanging benchmark agent and the result is logged. This produces objective, monotonically-comparable evidence of improvement that does not depend on the changing opponent pool.

Together these turn the project from "let it train for hours and trust the win-rate curve" into a controllable, measurable, interactive training experience.

## 2. Design constraints

* Backward compatible — the existing `train --games N --size 8` CLI keeps working with default settings.
* No assumption about whether the algorithm upgrade in `09_algorithm_upgrade_plan.md` is in place. Both plans are independent on paper; `11_unified_upgrade_plan.md` discusses the conflicts.
* Human game capture reuses the existing PyGame UI; no parallel UI is built.
* All session state is persisted, so the user can stop the program mid-session and resume.

## 3. Core abstractions

### 3.1 OpponentSource — who the agent plays in a given game

A unified interface; every existing and new agent already conforms or can be wrapped.

```python
class OpponentSource(Protocol):
    name: str                 # e.g. "random", "heuristic", "self", "snapshot:gen_007",
                              #      "human", "alphabeta_d4", "benchmark"
    def next_agent(self) -> BaseAgent: ...
    def is_blocking(self) -> bool:    # True for HumanSource (input wait), else False
        ...
```

Implementations:

* `RandomSource`, `HeuristicSource`, `AlphaBetaSource(depth)`, `BenchmarkSource` — return their fixed agent.
* `SelfSource(agent_factory)` — returns a frozen copy of the current learner.
* `SnapshotPoolSource(pool, weights="recent")` — samples from the registered snapshot pool.
* `HumanSource(ui)` — wraps the PyGame UI's human player; blocking on mouse input.

### 3.2 TrainingPhase — one chunk of a session

```python
@dataclass
class TrainingPhase:
    name: str                  # human-readable label
    opponent: OpponentSource
    n_games: int               # how many games to play in this phase
    train: bool = True         # whether to update the learner on transitions from this phase
    weight: float = 1.0        # replay-buffer weight multiplier for transitions from this phase
```

The `weight` field is the key to handling human-game scarcity: human transitions can be pushed with weight 5.0 or 10.0 so each game contributes more to gradient updates than a self-play game.

### 3.3 TrainingSchedule — the whole session

```python
@dataclass
class TrainingSchedule:
    phases: list[TrainingPhase]
    benchmark_opponent: OpponentSource         # the fixed reference (Section 5)
    benchmark_every_n_games: int = 1           # play benchmark game(s) after every N training games
    benchmark_games_per_check: int = 1         # how many benchmark games per check
    snapshot_every_n_games: int = 1000         # standard snapshotting cadence
    eval_every_n_games: int = 500              # standard eval cadence
```

A schedule serialises to JSON for replay/audit and can be loaded from a CLI argument.

## 4. CLI and schedule files

### 4.1 Inline schedule (quick interactive use)

```bash
python -m src.training.train \
    --size 8 \
    --schedule "self:50,human:10,self:100,human:5,pool:200,human:10" \
    --benchmark alphabeta_d4 \
    --benchmark-every 1
```

The `--schedule` string is parsed left-to-right; each phase is `<source>:<n_games>` with sensible default `train=True, weight=1.0` (humans default to weight 5.0).

### 4.2 Schedule from JSON file

```bash
python -m src.training.train --schedule-file schedules/colile_evening.json --size 8
```

Example file:

```json
{
  "phases": [
    { "name": "warmup",   "opponent": "random",          "n_games": 200, "weight": 1.0 },
    { "name": "self1",    "opponent": "self",            "n_games": 50,  "weight": 1.0 },
    { "name": "me1",      "opponent": "human",           "n_games": 10,  "weight": 5.0 },
    { "name": "pool",     "opponent": "snapshot_pool",   "n_games": 100, "weight": 1.0 },
    { "name": "me2",      "opponent": "human",           "n_games": 5,   "weight": 5.0 },
    { "name": "bossfight","opponent": "alphabeta_d4",    "n_games": 20,  "weight": 1.5 }
  ],
  "benchmark": "alphabeta_d4",
  "benchmark_every_n_games": 1,
  "benchmark_games_per_check": 1,
  "snapshot_every_n_games": 100,
  "eval_every_n_games": 50
}
```

### 4.3 Resume

```bash
python -m src.training.train --resume runs/2026-05-29_colile_evening
```

Loads the schedule, the last-saved learner state, the partial CSV log, and continues from the next un-played game.

## 5. Fixed benchmark agent

### 5.1 What "fixed" means

The benchmark agent is constructed **once at the start of the session** and is **never updated**, **never re-initialised**, and is the **same agent across every training game** in the session (and ideally across sessions). It is the experimental control variable. Any change in the learner's win-rate against the benchmark is a real measurement, not a side-effect of a drifting opponent.

### 5.2 What it should be

Recommendations, in preference order:

1. **`AlphaBetaAgent(depth=4)` with the hand-engineered evaluation from Phase E of the upgrade plan.** This is deterministic for a given seed, computationally tractable (sub-second per move on 8×8), and uncontroversial — alpha-beta with a sensible eval is a textbook benchmark.
2. **`HeuristicAgent`** as a secondary fallback if alpha-beta is not yet built. Slightly weaker but always available.
3. **`RandomAgent`** as a sanity-floor benchmark — only useful for the very early phase of learning.

For real evidence collection, run **two** benchmarks per training game — one against alpha-beta (the "moving the needle" measurement) and one against random (the "have we regressed catastrophically?" sentinel). The cost is small because both opponents are cheap and games on 8×8 finish in tens of ms.

### 5.3 What the benchmark logs

A new CSV file `results/logs/benchmark_log_size{N}.csv` with one row per training game:

```
training_game, phase_name, benchmark_name, benchmark_result, benchmark_score_diff,
benchmark_episode_length, benchmark_seed, wall_clock_sec
```

* `benchmark_result` — 1 if the learner won, 0.5 if drawn, 0 if lost.
* `benchmark_score_diff` — Mode 2 only; `learner_score − benchmark_score`.
* `benchmark_seed` — fixed for the row's game index, so a re-run produces identical results.

Rolling windows (last 100 games, last 500 games) are computed at plot time, not at log time, so the raw log stays append-only.

### 5.4 New plot

`results/figures/benchmark_curve.png`:
* x-axis: training game index
* y-axis: rolling-100 win rate vs benchmark
* one line per benchmark opponent
* horizontal reference line at 0.5

Added to the standard `generate_all_plots()` output set. The submission report will reference this figure as **the** evidence of improvement-with-experience.

## 6. Human game capture

### 6.1 Flow

When a training phase has `opponent="human"`, the system switches into UI mode:

1. The PyGame window opens (if not already), shows the current learner as the AI opponent.
2. The learner side is randomised per game (half as P1, half as P2).
3. The user plays one game against the current learner. Every move (both sides) is captured as a transition with the standard `(s, a, r, s', done, legal_mask')`. Reward shaping is identical to self-play.
4. The transitions are pushed into the replay buffer with weight `phase.weight` (default 5.0 for human phases).
5. The benchmark check runs after the game ends (the human does not have to wait — it runs in the background while the next prompt is shown).
6. Loop until `phase.n_games` is reached, then advance to the next phase.

### 6.2 What is captured

Every transition the human generates is treated as ground-truth for the algorithm — the human's move is the action taken, and the next state, reward, and legal mask are computed by the engine as normal. From the learner's perspective these are simply transitions from a stronger opponent.

If the user wants to **also** be the agent's controller (i.e., the user moves on the *learner*'s turn, generating expert demonstrations), that is a separate mode:

```bash
python -m src.training.train --schedule "demo:30" --size 8
```

In demo mode, the user plays on the learner's side, the opponent is a fixed agent (random, snapshot, alpha-beta — configurable), and every human action is treated as the action label. Transitions are pushed with a higher weight (10.0) into the buffer. This is behavioural-cloning-flavoured training.

### 6.3 UI minimum changes

The PyGame UI already has hot-seat and vs-AI modes. The training-mode capture is a thin wrapper that:

* Adds a small "Training mode — phase X of Y, game M of N" status banner.
* Suppresses the post-game victory screen (just shows a one-line "Result: <W/L/D>" and auto-advances after 2 seconds).
* Adds a Pause button that suspends the schedule and returns to the main menu (resume from `--resume`).

## 7. Buffer-weighting strategy for mixed-quality opponents

The replay buffer becomes a **multi-source** store. Sources contribute at different rates:

| Source | Rate of generation | Quality of transition | Default weight |
|--------|--------------------|-----------------------|----------------|
| Self-play | Fast (~100/min on 8×8 CPU) | Medium (improves with agent) | 1.0 |
| Snapshot pool | Fast | Variable (per snapshot) | 1.0 |
| Random | Fast | Low | 0.5 |
| Heuristic | Fast | Medium | 1.0 |
| Alpha-beta | Slow (~10/min on 8×8 CPU) | High | 2.0 |
| **Human (opponent)** | **Very slow (~1/min)** | **High** | **5.0** |
| **Human (demonstrator)** | **Very slow** | **Highest** | **10.0** |

Weights are realised at sampling time via importance-weighted minibatch sampling:

```python
# in ReplayBuffer.sample(batch_size)
probabilities = self.weights / self.weights.sum()
indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=True)
```

This is mathematically equivalent to duplicating each transition `weight` times in the buffer (which would waste memory).

### 7.1 Safeguards

* `max_replay_weight` cap (default 20.0) to prevent any single transition from dominating gradient updates.
* Optional **demonstration buffer** as a second buffer with a fixed mixing ratio (e.g., 25% of every minibatch sampled from the human buffer, 75% from the self-play buffer). This is the DQfD (Deep Q-learning from Demonstrations, Hester et al. 2018) recipe and is the cleanest way to keep small-quantity high-quality data influential without distorting the main buffer.

The default in v1 is the single-buffer importance-weighted approach; the demonstration-buffer alternative ships in v1.1 if results require it.

## 8. Session bookkeeping

Each training session produces a single directory under `runs/<timestamp>_<session_name>/`:

```
runs/2026-05-29_colile_evening/
├── schedule.json                  # the schedule, copied verbatim
├── learner_initial.pt             # snapshot of the learner at session start
├── learner_latest.pt              # rolling latest snapshot (overwritten)
├── learner_final.pt               # written on session-end / clean shutdown
├── training_log.csv               # per-training-game row (unchanged from existing logger)
├── benchmark_log.csv              # per-training-game benchmark result (new)
├── human_games.jsonl              # one JSON object per human game (move list + final outcome)
└── plots/                         # all figures generated at session-end
```

The benchmark CSV is the artefact that "proves" improvement. It is short, append-only, and trivially auditable.

## 9. Modes the user can run

### 9.1 Pure auto-train (existing)

```bash
python -m src.training.train --games 10000 --size 8
```

Equivalent under the new system to a single phase `[("self", 10000)]` with default benchmark `alphabeta_d4` and `benchmark_every_n_games=1`.

### 9.2 Quick "play 10 games against the learner to nudge it"

```bash
python -m src.training.train --schedule "human:10" --size 8 --resume runs/last
```

10 games of human-vs-learner; benchmarks logged after each.

### 9.3 Interleaved evening training

```bash
python -m src.training.train --schedule-file schedules/colile_evening.json --size 8
```

The mixed schedule from Section 4.2.

### 9.4 Pure demonstration

```bash
python -m src.training.train --schedule "demo:30" --size 8
```

User generates 30 demonstration games on the learner's side against a fixed opponent; learner trains on demonstrations only. Useful as a warm start before self-play.

## 10. Tests

`tests/test_training_schedule.py`:

* Parsing a schedule string `"self:50,human:10"` produces the expected `TrainingSchedule`.
* JSON round-trip on a schedule preserves all fields.
* Empty schedule raises a clear error.

`tests/test_benchmark_logger.py`:

* After every training game, exactly one row is appended to `benchmark_log.csv`.
* Benchmark seed is reproducible across re-runs of the same schedule.
* `generate_benchmark_plot()` produces the new figure with axes and title.

`tests/test_buffer_weighting.py`:

* High-weight transitions are sampled disproportionately often (statistical check over 10 000 samples).
* Weight cap is enforced.

`tests/integration/test_human_loop_stub.py`:

* With a stubbed `HumanSource` that returns a fixed move list, a 5-game human phase completes and pushes 5 × (N × 0.5) average transitions into the buffer.
* Session directory is created with all expected files.

## 11. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Benchmark game after every training game inflates wall-clock time | High | Medium | `benchmark_every_n_games` is tunable; default 1 on 8×8 (cheap), 5 on 12×12 (more expensive); benchmark depth is shallow by design |
| Human games are too few to influence learning | High | Medium | Weighted sampling (default 5x) + optional demonstration buffer with fixed mix ratio |
| User forgets to set a benchmark and runs blind | Low | Medium | If `--benchmark` is omitted, default to `heuristic`; warn loudly if `random` |
| Human game session is interrupted (user fatigue) | Medium | Low | `--resume` from the partial session directory |
| Benchmark agent itself "learns" by accident | Low | High | Defensive guard: `BenchmarkSource` wraps its agent in a `frozen=True` flag; any attempt to call `.update()` raises |
| Human is much stronger than the learner — losing every benchmark check is uninformative | Medium | Low | Track score-difference (`benchmark_score_diff`) in Mode 2 so the *margin* of loss is the early signal of improvement; the win/loss flip comes later |
| User wants to bias the agent toward their own play style | Medium | Low | This is by design when human weight is high; document and surface a warning in the report template |

## 12. What this plan does **not** include

* Multiple concurrent users (only one human at a time).
* Online/networked benchmark agents.
* Automatic curriculum design (the user composes the schedule manually).
* Reward learning from preferences (RLHF-style). The closest provision is the demonstration-buffer mode in Section 7.

---

*End of human-in-the-loop training plan.*
