# Improvement Plan — Lessons from `run_pts_001`

**Written:** 2026-05-30
**Based on:** `results/size_08/run_pts_001/analysis_report.md`
**Applies to:** Next training run (`run_pts_002` or equivalent)

These are observations and intended changes only. No code has been modified yet.

---

## I1 — Fix Difficulty Band Assignment (Critical)

**Problem:** The difficulty band is assigned by generation ordinal (game count), not by measured performance. gen_010 calls itself "Champion / master" while winning only 15% vs random — the worst model in the run.

**Fix:** Band assignment must read the actual `win_rate_vs_random` and `win_rate_vs_heuristic` values and map them through `DIFFICULTY_BANDS` in `config.py`. The friendly name and band written to the registry should reflect what the model can actually do, not how long it trained.

---

## I2 — Multi-Game Benchmark Per Check (Critical)

**Problem:** Each benchmark check ran a single game. `benchmark_result` is always exactly 0.0 or 1.0 — binary noise, not a skill signal. A 40% and a 60% win-rate agent are statistically indistinguishable on one game.

**Fix:** Run at least 20–50 games per benchmark check and log the win fraction. The `BenchmarkLogger` call in the simple `train()` path must pass an explicit `games_per_check` argument (e.g. 32). This produces a real number between 0 and 1 that shows a trend over time.

---

## I3 — Best-Model Checkpointing (High Priority)

**Problem:** Training overwrites good policies in late training with no way to recover. gen_004 (the best model) was produced at game 4,000 but the run continued for another 6,000 games and the weights were never separately preserved as "best seen so far."

**Fix:** Track `best_wr_vs_heuristic` throughout training. Whenever a new best is set, save the weights to a dedicated `best/` subfolder inside the run directory (e.g. `models/size_08/run_pts_001/best/weights.pt`). This checkpoint is never overwritten by later snapshots — it is the gold standard for that run regardless of what happens afterward.

---

## I4 — Early-Stop / Rollback Trigger (High Priority)

**Problem:** The run had no automatic safeguard. Once degradation began at game ~7,400, training continued for 2,600 more games producing progressively worse models with no intervention.

**Fix:** Add a degradation detector. If `win_rate_vs_heuristic` drops more than 15 percentage points below the best-ever value for three consecutive evaluation intervals, either stop training and report the best checkpoint, or optionally reload the best checkpoint weights and resume from there. This would have saved the run at approximately game 7,500.

---

## I5 — More Evaluation Games (Medium Priority)

**Problem:** `EVAL_GAMES = 100` vs random and 50 vs heuristic produces too much noise. Win rates swung from 6% to 98% in consecutive 100-game evaluations after epsilon hit its floor — this cannot be real skill change between 100 training games.

**Fix:** Increase to 200 games vs random and 100 vs heuristic. The statistical error on a win-rate estimate at p=0.5 drops from ±5% (n=100) to ±3.5% (n=200), which is still not tight but significantly reduces false signals in the learning curve.

---

## I6 — Plateau-Based LR Decay Instead of Fixed Milestones (Medium Priority)

**Problem:** LR decays at fixed fractions of total game count (40% and 75%). The second decay at game 7,500 coincided with the onset of catastrophic forgetting and appears to have accelerated it — the lower LR could not correct bad transitions fast enough.

**Fix:** Consider decaying LR based on a plateau condition (e.g. no improvement in `win_rate_vs_heuristic` for N consecutive evaluation intervals) rather than a fixed schedule. If the plateau condition is never triggered, LR stays at its initial value longer, which preserves adaptability in late training when the policy is under stress.

---

## I7 — Stronger Intermediate Reward Shaping (Medium Priority)

**Problem:** `STEP_REWARD_SCALE = 0.05` is too weak for a 64-move game. With `GAMMA = 0.99`, the terminal reward discounted back 63 steps is approximately 0.53 — half the terminal signal is lost by the time it reaches the opening moves. Early moves receive almost no learning signal from the game outcome.

**Fix:** Consider increasing `STEP_REWARD_SCALE` to 0.10–0.15, or investigate a richer shaping function that explicitly rewards blocking opponent open-4 threats (not just delta-score). This does not change what winning or losing means — it only densifies the signal the agent receives on the path to that outcome.

---

## I8 — Extend Epsilon Decay Across Full Training (Medium Priority)

**Problem:** `EPS_DECAY_GAMES = 5,000` means epsilon hits its floor exactly at the halfway point. The second 5,000 games are run with almost no exploration (ε=0.05). Low exploration means the buffer fills with similar states, increasing the risk of overfitting and catastrophic forgetting.

**Fix:** Extend decay to cover 70–80% of training (e.g. `EPS_DECAY_GAMES = 7,000` for a 10,000-game run). This keeps the agent exploring more varied states deeper into training and reduces buffer homogeneity in the late phase.

---

## I9 — Clarify and Fix Benchmark Agent Depth (Low Priority)

**Problem:** `config.py` sets `BENCHMARK_DEPTH_BY_SIZE = {8: 2}` (depth 2 for 8×8), but this run used `--benchmark alphabeta_d4` (depth 4) from the command line. Depth-4 alpha-beta is substantially stronger than depth 2. The run's ~42% peak win rate against depth 4 is a respectable result, but the target is ambiguous.

**Fix:** Before the next run, decide on a consistent benchmark depth and document it. Options:
- Depth 2 — achievable ceiling, clearer progress signal
- Depth 4 — ambitious, harder ceiling (what this run used)
- Log both in parallel (separate columns) for a fuller picture

---

## I10 — Prioritized Experience Replay (Low Priority)

**Problem:** The replay buffer uses weighted sampling by source type but not prioritized experience replay (PER). In late training, most buffer transitions are "easy" (TD-error near zero, already well-predicted). The network wastes gradient steps on transitions it has already learned.

**Fix:** Implement PER so transitions with high TD-error are sampled more frequently. This focuses learning on the cases the network currently gets wrong, which is especially valuable in late training when the agent's policy is fragile.

---

## Priority Order for Next Run

| Priority | Item | Expected Impact |
|:---:|------|:---:|
| 1 | I3 — Best-model checkpointing | Prevents losing the best weights to late-run collapse |
| 2 | I2 — Multi-game benchmark | Turns the benchmark log into an actual skill signal |
| 3 | I1 — Fix band assignment | Fixes the difficulty system for players |
| 4 | I4 — Early-stop trigger | Stops wasting compute after degradation begins |
| 5 | I5 — More eval games | Cleaner learning curves |
| 6 | I8 — Extend epsilon decay | More exploration in late training |
| 7 | I6 — Plateau-based LR decay | More adaptive, less harmful in late training |
| 8 | I7 — Stronger reward shaping | Better credit assignment for early moves |
| 9 | I9 — Benchmark depth decision | Cleaner comparison target |
| 10 | I10 — PER | Lower priority; other fixes address the same late-training fragility |
