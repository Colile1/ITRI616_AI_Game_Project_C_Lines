# Run `run_pts_001` — Post-Run Analysis Report

**Date:** 2026-05-30
**Board size:** 8×8 | **Mode:** `points_full` | **Total training games:** 10,000
**Benchmark opponent:** Alpha-Beta depth-4 (`alphabeta_d4`)
**Network:** ResNet v1 — 4 residual blocks, 64 channels, 10-channel state encoding
**Run duration:** ~16 hours (2026-05-29 09:25 → 2026-05-30 01:06 UTC)

---

## 1. What the Numbers Say

### 1.1 Snapshot Performance Summary

| Gen | Games | Label | Band | WR vs Random | WR vs Heuristic |
|-----|-------|-------|------|:---:|:---:|
| gen_001 | 1,000 | Apprentice | novice | 66% | 62% |
| gen_002 | 2,000 | Beginner | novice | 52% | 70% |
| gen_003 | 3,000 | Improver | easy | 79% | 80% |
| gen_004 | 4,000 | Cadet | easy | **88%** | **90%** |
| gen_005 | 5,000 | Tactician | medium | 78% | 90% |
| gen_006 | 6,000 | Adept | medium | 44% | **92%** |
| gen_007 | 7,000 | Tactician | medium | 57% | **100%** |
| gen_008 | 8,000 | Veteran | hard | 24% | 66% |
| gen_009 | 9,000 | Strategist | hard | 29% | 42% |
| gen_010 | 10,000 | **Champion** | **master** | **15%** | **10%** |

**The final model is the worst model produced in this run.** gen_010 (labeled "Champion") has lower win rates than even gen_001. The run went backwards in its final 3,000 games.

### 1.2 Benchmark Win Rate vs Alpha-Beta depth-4 (per phase)

| Phase | Games | Wins / Checks | Win Rate |
|-------|-------|:---:|:---:|
| Phase 1 — Early warmup | 0–999 | 2/11 | 18% |
| Phase 2 — Mid warmup | 1,000–1,999 | 3/11 | 27% |
| Phase 3 — LR=1e-3 active | 2,000–3,999 | 8/21 | 38% |
| Phase 4 — LR=5e-4 active | 4,000–7,499 | 15/36 | **42%** |
| Phase 5 — LR=2.5e-4 active | 7,500–9,999 | 5/25 | **20%** |

The agent improved steadily through phase 4, then **collapsed in the final phase** after the second LR decay.

### 1.3 Training Curve Highlights

**Win rate vs random:**

```
Games 0–4000  (epsilon decaying 1.0→0.24):  roughly 46–88%, trending up
Games 4000–7400 (epsilon near floor):        peak performance, 80–95%+ some checks
Games 5300–onward (epsilon = 0.05):          EXTREME variance begins
Games 7500–9999:                             swings 6% → 97% → 24% → 87% → 6%
Game 9999 (final):                           6% vs random  !!
```

**Win rate vs heuristic:**

```
Best stretch:  games 2800–7300 — consistently 80–100%
Collapse:      games 7500–9999 — drops to as low as 10–20% repeatedly
Final:         10% vs heuristic (gen_010)
```

**Loss:**
Stays in the narrow band 0.001–0.04 throughout — no clear trend up or down. The loss appears converged and uninformative from around game 1,000 onward.

**Episode length:**
Always exactly **64.0** — every single game plays to a full board. This is expected in `points_full` mode on 8×8, but has important implications (see Section 3).

---

## 2. What Actually Happened — Root Cause Analysis

### Problem 1: Catastrophic Forgetting in Late Training

**What:** After game ~7,400, performance collapsed dramatically. The final snapshot is catastrophically bad.

**Why:** This is textbook catastrophic forgetting in a DQN with a fixed-size replay buffer. At game 7,500, the LR dropped to 2.5e-4 AND epsilon had long been at its floor (0.05). The replay buffer (100,000 transitions) was now dominated by transitions collected under a mature policy with very low exploration. When the network overfits to that narrow distribution of states, gradient updates overwrite earlier, more general knowledge. There is no mechanism to protect or restore high-quality earlier weights.

Additionally, `SELF_PLAY_MIX_PROB = 0.5` means 50% of opponents are clones of the current agent. Once the current agent degrades, self-play against it generates poor-quality transitions that corrupt the buffer further — a feedback loop of degradation.

### Problem 2: Single-Game Benchmark is Too Noisy to Guide Decisions

**What:** Every benchmark check in the log is a single game — `benchmark_result` is exactly 0.0 or 1.0 throughout. A single game on an 8×8 board is a coin flip worth of signal.

**Why:** The `BenchmarkLogger` was called without a `games_per_check` override in the simple `train()` path, so it defaulted to one game per check. A 40% win-rate agent and a 60% win-rate agent are statistically indistinguishable on a single result. The benchmark log looks like progress noise rather than a true skill curve.

### Problem 3: Difficulty Band Labels Are Detached from Reality

**What:** gen_010 is labeled "master / Champion" but is the weakest model in the run. gen_004 is labeled "easy / Cadet" but has 88% win rate vs random.

**Why:** The difficulty band appears to be assigned based on the generation ordinal (game count progression) rather than from measured win rates. The `DIFFICULTY_BANDS` dictionary in `config.py` correctly maps performance ranges to bands, but the code writing the registry is not using those ranges — it uses snapshot order. This means the difficulty system players see in the game is completely reversed for this run.

### Problem 4: Extreme Policy Variance at Low Epsilon

**What:** After epsilon reaches its floor of 0.05 around game 5,000, the win-rate-vs-random column becomes wildly unstable — swinging 0.06 → 0.98 → 0.27 → 1.00 → 0.11 across consecutive 100-game evaluations.

**Why:** Three compounding factors:
1. `EVAL_GAMES = 100` for random and only 50 for heuristic introduces measurement noise, but not enough to explain swings from 6% to 98%.
2. The real cause is a brittle policy. With ε=0.05, the agent exploits a narrow set of learned patterns. When those patterns align with the evaluation games it scores near 100%; when they don't, it collapses.
3. The heuristic and random agents are deterministic enough that exact seed states matter greatly — the evaluation is not diversified enough to smooth this out.

### Problem 5: Reward Signal is Maximally Delayed

**What:** `mean_ep_len = 64.0` in every single row — games always run to completion.

**Why:** In `points_full` mode there is no early termination. The terminal WIN/LOSS reward only arrives on move 64. With `GAMMA = 0.99`, the terminal reward discounted back to move 1 is `0.99^63 ≈ 0.53` — half the value of winning is invisible to early moves. Intermediate delta-score shaping exists (`STEP_REWARD_SCALE = 0.05`) but is scaled too small to compensate. Credit assignment is hard — the agent cannot clearly link early-game decisions to the final outcome.

### Problem 6: LR Decay Timing Appears Counterproductive

**What:** The second LR decay (to 2.5e-4 at game 7,500) coincides almost exactly with the onset of catastrophic forgetting.

**Why:** The LR milestones are fixed fractions of total training games (40% and 75%). At game 7,500 the replay buffer contains mostly near-converged, low-exploration transitions of decreasing quality. A lower LR makes the network slower to correct mistakes AND cannot maintain good policies learned earlier. The timing amplified the collapse rather than stabilising it.

---

## 3. What Went Right

1. **Early training was effective.** By game 3,000–4,000, the agent was genuinely competitive: 88% vs random, 90% vs heuristic. The curriculum (warmup → self-play) worked for the first two-thirds of training.

2. **The 10-channel encoding and ResNet architecture caused no problems.** Loss stayed low and stable. The network had sufficient capacity and the state representation was usable throughout.

3. **Benchmark logging infrastructure works correctly.** CSV logging, wall-clock timing, and periodic checks all functioned. The issue is signal quality (one game per check), not the logging system itself.

4. **gen_004 through gen_007 are genuinely useful models** covering roughly 80–100% vs heuristic. These are the correct candidates for the mid-to-upper difficulty bands in a player-facing difficulty system.

5. **Self-play with snapshot pool was stable in the middle phase.** The recency-bias pool mechanism appeared to help in games 2,000–7,000 before degradation set in.

---

## 4. Summary Table

| What | Finding |
|------|---------|
| **Best model produced** | gen_004 (game 4,000): 88% / 90% vs random / heuristic |
| **Worst model produced** | gen_010 (game 10,000): 15% / 10% vs random / heuristic |
| **Primary failure mode** | Catastrophic forgetting in final 3,000 games |
| **Secondary failure mode** | Single-game benchmark too noisy to measure real progress |
| **Mislabeled models** | gen_004 labeled "easy" though it's the best; gen_010 labeled "master" though it's the worst |
| **Peak alpha-beta win rate** | ~42% vs depth-4 alpha-beta (games 4,000–7,400) |
| **Root causes** | No best-model checkpoint, no early-stop, LR decay timing, buffer pollution from degraded self-play |
| **What to keep for next run** | Architecture, curriculum structure, benchmark logging infrastructure, weights from gen_004–gen_007 |
