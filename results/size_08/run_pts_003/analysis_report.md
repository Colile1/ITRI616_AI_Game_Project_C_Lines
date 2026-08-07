# Analysis Report — run_pts_003

**Mode:** Points-Full (PTS) | **Board:** 8×8 | **Games:** 20,000
**Written:** June 2026
**Seeded from:** run_pts_002 best weights (game 6,400, WR_h=100%)
**prev_best opponent:** run_pts_002 best (100% vs heuristic — fully converged master-level)

---

## 1. Summary of Results

| Metric | Game 0 | Peak (games 0–10,000) | Final (19,999) |
|--------|--------|-----------------------|----------------|
| WR vs heuristic | 46% | **100%** (games 7,000–10,000) | 57% |
| WR vs random | 52% | **100%** (game 8,000+) | 51% |
| Elo | 796 | **1,187** (game 9,900) | 839 |
| Episode length | 64 moves | 64 moves (board fills) | 64 moves |

**Best snapshot:** gen_017 (game 14,000) — WR_h=100%, the highest sustained performance in the resumed run.

**Note:** The first 10 snapshots (gen_001–gen_010) were copied from run_pts_002 as the starting checkpoint. The original run_pts_002 baseline (games 0–10,000) shows the strongest improvement curve.

---

## 2. Phase-by-Phase Analysis

### Phase 1: Inherited Baseline (games 0–10,000 from run_pts_002)
This phase represents the prior run's training, used as the starting point for run_pts_003.

- WR vs heuristic: 0% → 100% (monotone, reached by game 9,900)
- Elo: 796 → 1,187 (+391 points — one of the cleanest improvement curves in the project)
- The agent learned to maximise cumulative score through a delta-score reward shaping: each move's reward equals the net line-score gained. This directly aligns the immediate reward with the final objective.
- All snapshots from gen_004 onward are labeled "master" (WR_h ≥ 90%).

### Phase 2: Curriculum Adaptation (games 10,000–14,000)
- The agent now faces the run_pts_002 best (100% vs heuristic) as its primary opponent — the hardest possible curriculum for this mode.
- WR_h drops sharply: 100% → 24–41% in the first 4,000 games.
- This regression is the "strong opponent shock": the agent's current policy exploits specific weaknesses of weaker opponents that the master-level prev_best does not have.
- Elo falls from 814 → 784 during this period.
- WR_h recovers strongly at game 14,000: gen_016 = 97%, gen_017 = **100%** — the agent re-reaches master level by adapting to the harder opponent.

### Phase 3: Oscillation and Stabilisation (games 14,000–20,000)
- WR_h oscillates: 100% → 23% → 74% → 57%
- This instability indicates the agent is not consistently generalising — it reaches master level momentarily but the policy is fragile against the strong curriculum opponent.
- Elo stabilises around 830–840 (above the heuristic anchor of 900, below the alphabeta anchor of 1,200).
- Final WR_h = 57%: above the heuristic baseline but below the peak of 100%.

---

## 3. Key Achievements

1. **Elo: 796 → 1,187 (+391 points)** in the first 10,000 games — one of the strongest improvement curves recorded in this project. The Elo progression was monotone throughout Phase 1.

2. **100% WR vs heuristic** achieved and held from game 7,000–10,000 — complete mastery of the rule-based opponent.

3. **100% WR vs random** achieved at game 8,000 and sustained until the curriculum shift.

4. **Difficulty band progression:** gen_001 (easy, 48%) → gen_002 (medium, 68%) → gen_003 (hard, 86%) → gen_004 (master, 90%) → gen_009 (master, 100%). All 10 bands populated, providing full difficulty coverage for the UI level select.

5. **Recovery after curriculum shock:** The agent fell from 100% to 24% vs heuristic at game 15,000, then recovered to 100% by game 14,000 — demonstrating that the harder curriculum is productive even if slow to converge.

---

## 4. Failure Modes Identified

### 4.1 Hard Curriculum Shock
Training against a 100%-WR opponent causes severe short-term regression (WR_h 100% → 24%). The curriculum was too aggressive — jumping directly from random/pool opponents to a perfect master-level opponent.

### 4.2 Late-Run Instability (games 14,000–20,000)
WR_h oscillates wildly (23–100%) rather than converging. The policy has not found a stable equilibrium against the master-level curriculum. Final WR_h = 57% represents a mediocre resting state.

### 4.3 No Benchmark Data Comparable to FTF
The PTS mode benchmark (WR vs AlphaBeta-4) was not separately tracked in the same detail as FTF. Future runs should add explicit benchmark logging for PTS from game 0.

---

## 5. Improvement Plan for run_pts_004

### R1 — Graduated Curriculum (Critical)
Do not jump directly to the 100%-WR prev_best opponent. Instead, sequence the opponents by difficulty:
- Games 10,000–12,000: RandomAgent + HeuristicAgent (re-stabilise)
- Games 12,000–15,000: run_pts_002 gen_005 (85% WR_h, intermediate level)
- Games 15,000+: run_pts_002 best (100% WR_h)

### R2 — Longer Pool Phase (High Priority)
The current pool phase (games 2,000–7,000) could extend to game 12,000 before introducing the hard curriculum. This gives the resumed agent more time to build robustness before facing the master opponent.

### R3 — Lower Learning Rate on Resume (High Priority)
When resuming from a converged policy, the LR should start lower (e.g. 5e-4 instead of 1e-3) to prevent catastrophic forgetting during the curriculum transition.

### R4 — Fix Final Snapshot Elo (Complete)
Already implemented (R1): `_elo_update()` now runs before the final `freeze()` call, ensuring the last snapshot always has a valid `elo_rating`.

### R5 — Multi-Seed Validation (Medium Priority)
Run seeds 1, 2, 3 from the run_pts_002 best weights to confirm reproducibility of the 100% WR_h result (currently single seed).

---

## 6. Recommended Next Command

```powershell
# Stable re-stabilisation before hard curriculum
python -m src.training.train --size 8 --mode points_full --games 30000 `
  --run-id run_pts_004 `
  --load-weights models/size_08/run_pts_003/gen_017/weights.pt `
  --start-game 14000 `
  --benchmark alphabeta_d4
```

*Note: Omit `--prev-best` initially to allow re-stabilisation, then reintroduce at game ~16,000 using a graduated opponent.*
