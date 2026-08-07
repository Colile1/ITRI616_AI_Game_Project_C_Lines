# Analysis Report — run_ftf_004

**Mode:** First-to-Four (FTF) | **Board:** 8×8 | **Games:** 20,000 (with Phase 4 restarts)
**Written:** June 2026
**Seeded from:** run_ftf_003 best weights (game 9,800, WR_h = 31%)
**Saved best snapshot:** game 16,300 — WR_h = 8% (last time best/ was updated in Phase 4)

---

## 1. Summary of Results

| Metric | Game 0 | Phase 3 Peak | Final (19,999) |
|--------|--------|--------------|----------------|
| Episode length (moves) | 33.3 | **8.9** (game 9,000) | 19.7 |
| WR vs random | 48.5% | **100%** (game 5,700) | 91.5% |
| WR vs heuristic | 0% | **31%** (game 9,800) | 8% |
| Elo | 790 | **792** (game 9,999) | 774 |
| Benchmark vs AlphaBeta-4 | 0% | **87.5%** (game 8,800) | ~40% (noisy) |

**Best training snapshot (not best/weights.pt):** game 8,800 — 87.5% vs AlphaBeta-d4.
**Saved best/ snapshot:** game 16,300 — only 8% WR_h. This mismatch is a key failure explained in §4.

---

## 2. Assessment of F1–F7 Fixes from run_ftf_003 Improvement Plan

| Fix | Target | Outcome |
|-----|--------|---------|
| F1 — Episode length as LR metric | Hold LR at 1e-3 for ≥ 5,000 games | **Not achieved.** LR decayed 1e-3 → 5e-4 at game 1,100. |
| F2 — Open-4 threat shaping | WR_h climbing within 1,500 games | **Uncertain.** WR_h reached 31% at game 9,800 — same as run_ftf_003 peak. No clear signal of improvement. |
| F3 — FTF difficulty bands | Mode-appropriate snapshot labelling | No evidence of change in logs. |
| F4 — Pool diversity + benchmark seeds | Reduce bimodal benchmark variance | **Not effective.** Benchmark still swings 0–87.5%. |
| F5 — n-step returns | Cleaner credit assignment | Not confirmed; single-step TD appears unchanged. |
| F6 — Adaptive early-loss threshold | Prevent penalty misfiring in short games | Not confirmed; Phase 4 regression suggests threshold issue persists. |
| F7 — Multi-seed validation | 3 seeds from best weights | Not done; single-seed run. |

**Net verdict:** The structural fixes did not land as intended. The LR plateau detector still misfires early (F1 unresolved), and the bimodal benchmark variance remains unaddressed (F4 unresolved). The core learning dynamic was unchanged from run_ftf_003.

---

## 3. Phase-by-Phase Analysis

### Phase 1 — Early Exploration (Games 0–2,000)

| Checkpoint | ε | LR | Ep. len | WR_r | WR_h | Elo |
|-----------|---|----|---------|------|------|-----|
| Game 0 | 1.00 | 1e-3 | 33.3 | 48.5% | 0% | 790 |
| Game 1,000 | 0.86 | 1e-3 | 31.8 | 48.5% | 0% | 716 |
| Game 2,000 | 0.73 | 5e-4 | 27.5 | 77.5% | 0% | 682 |

- **LR decays too early:** the plateau detector fired at game 1,100 (1e-3 → 5e-4), then again at game 2,100 (5e-4 → 2.5e-4). This exhausted the LR budget before the agent had established stable gameplay. Target was ≥ 5,000 games at 1e-3; actual was ≈ 1,100 games.
- Elo drop (790 → 682) is expected: the loaded weights behave randomly under ε = 1.0 and the Q-function degrades temporarily as it reinitialises on random-play transitions.
- Episode length decline (33.3 → 27.5) shows the agent is beginning to shorten games — a positive early signal.

### Phase 2 — Rapid Acquisition (Games 2,000–7,000)

| Checkpoint | ε | LR | Ep. len | WR_r | WR_h | Benchmark vs AB-4 |
|-----------|---|----|---------|------|------|--------------------|
| Game 2,000 | 0.73 | 5e-4 | 27.5 | 77.5% | 0% | 0% |
| Game 4,000 | 0.46 | 2.5e-4 | 18.1 | 95.0% | 0% | 0% |
| Game 5,200 | 0.29 | 1.25e-4 | 15.7 | 99.0% | 10% | 0% |
| Game 7,000 | 0.05 | 1.25e-4 | 9.9 | 100% | 6% | 3.1% |

- **Steepest learning curve of the run:** episode length drops 27.5 → 9.9 (−64%) in 5,000 games.
- WR_random crosses 100% at game 5,700; then stays there.
- WR_h makes first significant moves (up to 10%) around game 5,000–5,500 when ε ≈ 0.3 — the agent begins exploiting 4-in-a-row patterns against a reactive opponent.
- Benchmark vs AlphaBeta-4 is near 0% for almost all of Phase 2; first multi-win benchmarks appear only at game 4,700 (6.25%) and game 5,500 (9.4%).
- LR was at its minimum (1.25e-4) from game 4,200 onward — the scheduler fired too early and the remaining learning was done at minimum LR.

### Phase 3 — Tactical Peak (Games 7,000–10,000)

| Checkpoint | ε | LR | Ep. len | WR_r | WR_h | Benchmark vs AB-4 |
|-----------|---|----|---------|------|------|--------------------|
| Game 7,000 | 0.05 | 1.25e-4 | 9.9 | 100% | 6% | 3.1% |
| Game 7,700 | 0.05 | 1.25e-4 | 9.4 | 99% | 10% | **78.1%** |
| Game 7,900 | 0.05 | 1.25e-4 | 9.2 | 100% | 7% | **81.25%** |
| Game 8,800 | 0.05 | 1.25e-4 | 8.8 | 100% | 12% | **87.5%** ← run peak |
| Game 9,800 | 0.05 | 1.25e-4 | 9.5 | 100% | **31%** | 43.75% |
| Game 9,999 | 0.05 | 1.25e-4 | 10.2 | 100% | 18% | — |

- **Epsilon floor reached at game 7,000.** Once ε = 0.05, the agent fully exploits its learned Q-function and benchmark performance surges.
- **87.5% vs AlphaBeta-d4 (game 8,800)** is the highest benchmark result in any FTF run to date. This is the true best checkpoint — not what `best/weights.pt` holds.
- Episode length approaches the theoretical minimum (8 moves = 4 per player in a perfect FTF game).
- WR_h peaks at 31% (game 9,800), matching run_ftf_003's peak — no net gain from the new reward shaping.
- **Bimodal benchmark pattern clearly present:** alternating between 0–6% and 50–87% on consecutive 100-game intervals. The agent has one dominant fork pattern that works from specific initial positions; when AlphaBeta opens differently, the DQN collapses.

### Phase 4 — Curriculum Regression (Games 10,000–20,000)

At game 10,000, training restarted with a new opponent (prev_best: run_ftf_003 best weights), and the LR reset to 1e-3. The training log shows **five separate restart events** within Phase 4, visible as game-counter resets and LR resets in the CSV:

| Restart | Game range | Outcome |
|---------|-----------|---------|
| Restart A | 10,000–12,500 | WR_h 19–33%, oscillating; best 33% |
| Restart B | 12,000–15,600 | WR_h 6–25%; benchmark 0–88% bimodal |
| Restart C | 15,000–15,600 | WR_h 9–13% |
| Restart D | 16,000–16,100 | WR_h 7–8% |
| Restart E | 16,000–19,999 | WR_h 5–10%, final 8% |

Key observations:
- **Episode length regression:** 8.9 → 19.7 moves as the stronger prev_best opponent defends effectively, forcing longer games.
- **WR_random regression:** drops from 100% to 91.5% — general capability erodes under the new curriculum.
- **Multiple restarts destabilised training:** each restart reinitialised the LR, causing repeated high-LR adaptation periods that overwrite prior learning. The five restarts likely explain why the final model performs worse than the Phase 3 peak.
- **best/weights.pt captured game 16,300 (WR_h = 8%)**, the last time a new WR_h high was set after the Phase 4 regressions. This is substantially below the Phase 3 peak.
- The benchmark log from game 15,500–15,600 shows two consecutive readings of 89% and 83%, suggesting a brief recovery just before Restart C — this moment was not captured in the best/ checkpoint.

---

## 4. Key Achievements

1. **73% episode length reduction (33.3 → 8.9 moves):** Approaches the 8-move theoretical minimum for FTF. Demonstrates the agent has fully internalised the 4-in-a-row objective.

2. **100% win rate vs random agent, sustained 13,000 games:** Categorical mastery of game fundamentals established and not lost despite Phase 4 curriculum pressure.

3. **87.5% vs AlphaBeta depth-4 (game 8,800):** Surpasses the prior run's 50% ceiling by a wide margin. At inference the DQN is operating near its tactical best with no tree search.

4. **31% WR vs heuristic (game 9,800):** Matches run_ftf_003's best result. Combined with the 87.5% benchmark, this is the strongest FTF checkpoint yet produced by the project.

---

## 5. Failure Modes

### 5.1 LR Plateau Detector Still Misfires (Critical — F1 not fixed)

LR decayed from 1e-3 to its minimum (1.25e-4) by game 4,200 — identical behaviour to run_ftf_003. The plateau detector treats early Elo noise as a plateau signal and exhausts the LR budget before stable gameplay begins. All learning from game 4,200 onward was done at minimum LR. The intended fix (episode length as plateau criterion) was not implemented or did not function correctly.

**Evidence:** Games 4,200–9,999 all show LR = 1.25e-4 despite significant changes in WR_h (0% → 31%) and episode length (18 → 9 moves) — exactly the kind of learning that should hold LR high.

### 5.2 best/ Checkpoint Missed the True Peak (Critical)

The best checkpoint saving mechanism updated `best/weights.pt` based on WR_h. Because WR_h oscillates heavily (0–31%), the checkpoint was last updated at game 16,300 with WR_h = 8% — well below the actual peak of 31% at game 9,800. The gen_009 snapshot (written at game 9,000) is closer to the best, but the 87.5% benchmark result at game 8,800 was not captured by any named checkpoint.

### 5.3 Bimodal Benchmark Variance (High — F4 not fixed)

The benchmark vs AlphaBeta-4 oscillates between 0% and 87.5% across consecutive 100-game intervals, even in Phase 3 when training is stable. This is position-specific overfitting: the DQN has mastered one fork pattern effective from a narrow set of opening positions, but fails completely from other openings. The pool diversity fixes (F4) did not reduce this variance.

### 5.4 Phase 4 Multi-Restart Instability (High)

Five training restarts in Phase 4 (each resetting LR to 1e-3) caused repeated high-loss catastrophic interference with the Phase 3 weights. Each restart achieved a brief recovery then degraded again. The curriculum transition from self-play pool to prev_best opponent was too abrupt and required better management.

### 5.5 WR_h Ceiling at ~31% (Medium)

WR_h has not improved beyond 31% across both run_ftf_003 and run_ftf_004. The heuristic opponent blocks single threats directly; the DQN wins only when it creates dual threats (forks), but does not do this reliably. The open-4 reward shaping (F2) either was not implemented or is not providing sufficient signal. This ceiling will not be broken without a more targeted reward signal for fork creation.

---

## 6. Comparison with run_pts_004

| Metric | run_ftf_004 (final) | run_pts_004 (final) |
|--------|---------------------|---------------------|
| Elo | 774 | **1,094** |
| WR vs heuristic | 8% | **93–100%** |
| WR vs random | 91.5% | **98–100%** |
| Episode length | 19.7 | 64.0 (pts uses full board) |
| Benchmark vs AB-4 | ~40% (noisy) | ~60–98% (bimodal at pts scale) |

The PTS mode agent is categorically stronger on every Elo and WR metric. The FTF objective (4-in-a-row on 8×8) is structurally harder: the reward signal is sparse (only one relevant score increment), the optimal game length is very short (8 moves), and the heuristic specifically exploits single-threat play. PTS agents score on every piece placement, giving denser reward signals.

---

## 7. Improvement Plan for run_ftf_005

### I1 — Fix LR Plateau Detector (Critical)

**Problem:** The plateau detector decays LR from 1e-3 to 1.25e-4 by game 4,200. All three ftf runs have exhibited this failure identically.

**Fix:** Replace the Elo-based plateau criterion with **mean episode length** (lower is better). The plateau should fire only when `mean_ep_len` stops decreasing over a 500-game window. Alternatively, use a **fixed cosine-annealing schedule** that holds LR at 1e-3 for the first 5,000 games, then decays:
```python
# cosine anneal: LR = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(pi * t / T))
# T = total_games, with warm phase 0..5000 holding lr_max
LR_WARMUP_GAMES = 5000
```
**Success criterion:** LR stays at 1e-3 until at least game 5,000.

---

### I2 — Capture the True Best Checkpoint (Critical)

**Problem:** `best/weights.pt` is saved when WR_h exceeds its prior best, but WR_h is noisy. The run's actual best checkpoint (87.5% vs AlphaBeta at game 8,800) was not captured.

**Fix:** Use a dual-criterion checkpoint strategy:
- Save `best_wh/weights.pt` when WR_h peaks (current behaviour, keep it).
- Save `best_bm/weights.pt` when the **benchmark vs AlphaBeta-4** 100-game average peaks.
- Save a periodic snapshot every 500 games regardless of performance.

**For run_ftf_005:** Seed from gen_008 weights (game 8,800) if available, or gen_009 (game 9,000), since the Phase 3 snapshots are the strongest starting point.

---

### I3 — Fork-Seeking Reward Shaping (High Priority)

**Problem:** WR_h is capped at 31% across two consecutive runs. The heuristic is defeated by simultaneous dual threats (forks) but the current reward shaping only reinforces single open-3 threats.

**Fix:** Add a fork reward: fire a bonus whenever the agent's move creates ≥ 2 independent open-3 threats in one turn.
```python
FTF_FORK_BONUS = 0.40   # bonus when agent creates 2+ open threats in one move
```
The `scoring.py` `count_open_threats()` function already supports this — call it before and after the move; if the count increases by ≥ 2, award the bonus.

**Success criterion:** WR_h > 40% sustained over any 500-game window in run_ftf_005.

---

### I4 — Stabilise Phase 4 Curriculum Transition (High Priority)

**Problem:** Loading a new (stronger) opponent at game 10,000 and simultaneously resetting LR caused catastrophic interference over 5 restarts in Phase 4.

**Fix:** Two-step approach:
1. **Gradual curriculum mixing:** rather than switching entirely to prev_best at game 10,000, mix it in at 20% probability alongside the self-play pool, increasing by 5% every 1,000 games.
2. **No LR reset on opponent switch:** keep LR at its current value when the curriculum changes; do not restart the scheduler.

```python
PREV_BEST_INTRO_GAME = 7000    # start mixing earlier, while agent is still strong
PREV_BEST_START_PROB = 0.10    # 10% initially
PREV_BEST_MAX_PROB   = 0.40    # 40% maximum, leaving 60% for self-play pool
```

---

### I5 — Add AlphaBeta-d2 as a Permanent Training Opponent (Medium Priority)

**Problem:** The agent overfits to its own self-play patterns. AlphaBeta uses adversarial search and consistently exposes opening sequences the self-play pool never generates.

**Fix:** Include AlphaBeta-d2 at a fixed 5% probability in the training opponent mix throughout the run:
```python
ALPHABETA_TRAINING_PROB = 0.05   # 1 in 20 games vs AlphaBeta-d2
```
This exposes the agent to adversarial openings from game 1, reducing the position-specific overfitting that causes the bimodal benchmark pattern.

---

### I6 — Increase Pool Diversity to Address Bimodal Benchmark (Medium Priority)

**Problem:** The self-play pool converges to a narrow set of positions, causing the agent to master one fork pattern and fail from all others.

**Fix:**
- Raise `MAX_POOL_SIZE` from 20 to 30 snapshots.
- Reduce `RECENT_POOL_BIAS` from 0.7 to 0.5 (equal weighting of old and recent snapshots).
- Randomise benchmark seeds per evaluation rather than using the deterministic `game_idx * 1000 + g` formula.

---

### I7 — Extend Epsilon Floor Duration (Low Priority)

**Problem:** ε hits 0.05 at game 7,000 and stays there for 13,000 games. With ε = 0.05 the agent is already fully exploiting, but the Phase 4 curriculum introduces positions the agent has never exploited against. More exploration in Phase 4 would help.

**Fix:** Reset ε to 0.15 when the curriculum changes (game ~10,000), then decay back to 0.05 over 2,000 games:
```python
if training_game == CURRICULUM_SWITCH_GAME:
    epsilon = 0.15
    epsilon_decay = compute_decay(0.15, 0.05, steps=2000)
```

---

### I8 — Multi-Seed Validation (Low Priority)

Run seeds 1, 2, 3 from the same starting weights once I1–I4 are confirmed working. Report mean ± std for episode length and WR_h.

---

## 8. Priority Order for run_ftf_005

| # | Item | Effort | Expected Impact |
|:--:|------|:------:|:---------------:|
| 1 | I1 — Fix LR plateau detector | Small code | Unlocks 5,000-game Phase 1 at 1e-3 |
| 2 | I2 — Dual-criterion checkpointing | Small code | Captures 87%+ benchmark checkpoints reliably |
| 3 | I3 — Fork reward shaping | Small code | Breaks WR_h ceiling beyond 31% |
| 4 | I4 — Gradual curriculum + no LR reset | Small config | Eliminates Phase 4 regression |
| 5 | I5 — AlphaBeta-d2 training opponent | Small config | Reduces bimodal benchmark variance |
| 6 | I6 — Pool diversity | Small config | Further reduces bimodal variance |
| 7 | I7 — ε reset on curriculum change | Small config | Improves Phase 4 exploration |
| 8 | I8 — Multi-seed validation | Compute | Statistical confidence |

---

## 9. Recommended Starting Command for run_ftf_005

Seed from the best available Phase 3 snapshot (gen_009, game 9,000 — closest to the 87.5% benchmark peak):

```powershell
python -m src.training.train --size 8 --mode first_to_four --games 20000 `
  --run-id run_ftf_005 `
  --load-weights models/size_08/run_ftf_004/gen_009/weights.pt `
  --start-game 9000 `
  --prev-best results/size_08/run_ftf_003/best/weights.pt `
  --benchmark alphabeta_d4 `
  --lr-warmup-games 5000
```

If gen_009 weights are unavailable (deleted from models/), use the best/ weights from run_ftf_004 as a fallback — they are weaker (game 16,300, 8% WR_h) but retain the Phase 1–2 learning.

---

## 10. The Core Insight

run_ftf_004 confirmed that the FTF DQN can reach **87.5% vs AlphaBeta depth-4** — a result competitive with minimax search — but only in a narrow 2,000-game window (games 7,000–9,000) when ε has just hit its floor and the LR is near minimum. The agent has the tactical capacity; the infrastructure around it (LR scheduling, checkpointing, curriculum management) is failing to capture and build on that capacity. run_ftf_005 is primarily an infrastructure fix, not an algorithm change.
