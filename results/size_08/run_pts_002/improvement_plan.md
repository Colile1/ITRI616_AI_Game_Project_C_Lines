# Improvement Plan — Lessons from `run_pts_002`

**Written:** 2026-06-01
**Based on:** `results/size_08/run_pts_002/analysis_report.md`
**Cross-reference:** `docs/deliverables/in_md_format/13_ML_Training_Improvement_Plan.md`
**Status:** No code modified yet.

`run_pts_002` is the first successful run — the agent learned correctly, Elo rose monotonically, and the final model scored 100%/100% against random and heuristic. The items below are refinements to an already-working system, not emergency fixes. They are ordered by the return-on-effort estimate from the improvement plan doc.

---

## What is Already Confirmed Working (do not change)

- Negamax sign (`−γ`), Double DQN, Huber loss — leave untouched
- Plateau-based LR decay — worked exactly as intended (4 decays)
- Performance-based difficulty band assignment — all 10 labels correct
- Elo tracking — clean monotone curve, suitable for report graphs
- Best-model checkpoint — preserved the best policy throughout
- Anchor pool (10% random/heuristic) — curriculum never collapsed

---

## R1 — Fix Missing Elo for Final Snapshot (Quick — 1 line)

**Problem:** `gen_010` in the registry has `elo_rating: null`. The final `freeze()` call in `train.py` doesn't receive an Elo value because it runs before the final `_elo_update()`.

**Fix:** Move the final `_elo_update()` call to before the final `freeze()` call and pass the updated Elo in `final_stats`.

**Impact:** Complete registry — all 10 snapshots have Elo.

---

## R2 — Reduce Benchmark Variance (High priority for robustness)

**Problem:** Game 8,300 showed 0% win rate (32/32 losses), surrounded by 97–100% checks. The agent has at least one consistently exploitable pattern against specific alpha-beta opening sequences. This would allow a human who finds that sequence to beat the agent.

**Fix options:**
- **Option A** — Increase games per benchmark check from 32 to 64. Halves sampling noise; a single adversarial seed cannot dominate.
- **Option B** — Randomise the starting player for benchmark games (currently alternates deterministically by game_idx). A stronger seed diversification reduces the chance of hitting the same adversarial seed repeatedly.
- **Option C** — Add alpha-beta depth-2 as a permanent benchmark alongside depth-4, giving a second, more-stable signal.

Recommend: Option A + B together. Change `BENCHMARK_GAMES_PER_CHECK = 64` in config.py and add a random offset to the benchmark seed.

---

## R3 — n-step Returns (Medium — P2.2 from improvement plan doc)

**Problem:** TD is still 1-step. End-of-game outcomes require many bootstrap steps to reach early-game Q-values. With 64 moves per game and γ=0.99, the terminal reward discounted to move 1 is `0.99^63 ≈ 0.53` — less than 3-step n-step would provide.

**Fix:** Implement 3-step returns in `self_play.py`. Accumulate `r_t + γ·r_{t+1} + γ²·r_{t+2}` as the target for step t, bootstrapping from `Q(s_{t+3})`. With negamax, the sign alternates each step:
```
n-step target = r_t − γ·r_{t+1} + γ²·r_{t+2} − γ³·max_a Q(s_{t+3}, a)
```
This also makes the synthetic-loss-transition hack in `train.py` (for FTF mode) unnecessary — remove it once n-step is confirmed working.

**Confirm:** Elo should cross 900 faster (within ~1,500 games rather than ~3,100), and benchmark crossover should occur earlier.

---

## R4 — Prioritized Experience Replay (Medium — P2.1 from doc)

**Problem:** The replay buffer samples by source weight (human/demo/alphabeta get higher weight) but not by how much a transition would teach the network. The agent wastes gradient steps on transitions it can already predict correctly.

**Fix:** Store per-transition TD error priority `p = (|δ| + ε)^α` alongside the source weight. Multiply the two weights when sampling. Apply importance-sampling correction `w = (N·P(i))^{-β}` to the per-sample loss term (anneal β from 0.4 to 1.0 over training). Use a sum-tree for efficient O(log N) sampling.

**Confirm:** The Elo curve should rise faster per-game (fewer games needed to reach 900, 1000, 1100).

---

## R5 — Multi-seed Validation (Medium — for credible evidence)

**Problem:** `run_pts_002` is one run with one random seed. The result is strong but run-to-run variance has not been quantified. For the project report, one run is good; three runs is definitive.

**Fix:** Add `--seed N` flag to `train.py` (seed `torch`, `numpy`, `random`). Run `run_pts_003`, `run_pts_004`, `run_pts_005` with seeds 1, 2, 3. Report mean ± std of final Elo and WR across three seeds.

**Confirm:** If all three runs reach Elo > 1,100 by game 7,000, the result is reproducible and the confidence bands in the report are tight.

---

## R6 — Calibrate Band Thresholds to Elo (Low — label quality)

**Problem:** gen_005 has Elo=1,006 (above heuristic anchor 900) but is labeled "hard" because WR_heuristic=85% < 90% threshold. Elo and the WR-based band disagree.

**Fix options:**
- Lower the master threshold from ≥90% to ≥85% WR vs heuristic.
- OR: use Elo > 950 (midpoint between heuristic anchor 900 and alphabeta anchor 1200) as the master criterion.

Either change would label gen_005 and gen_006 (Elo 1050) as master, which better reflects their actual skill. Recommend Elo > 950 as the master gate since Elo is now available in every snapshot.

---

## R7 — Extend to 9×9 and Larger Boards

**Problem:** `run_pts_002` only covers 8×8. The project brief expects agents for N ∈ {8, 9, 10, 11, 12}. The same pipeline now works reliably for 8×8 — the architecture is board-size parameterised so the same code runs for larger boards without changes.

**Plan:**
```
python -m src.training.train --games 10000 --size 9  --mode points_full --benchmark alphabeta_d4
python -m src.training.train --games 10000 --size 10 --mode points_full --benchmark alphabeta_d4
```
For sizes 11 and 12, the action space grows (121, 144 cells) so training may need more games. Consider `--games 15000` for size 11–12.

**Expected Elo curve:** Same qualitative shape as 8×8 but shifted right (more games to convergence) because the action space is larger.

---

## R8 — FTF Mode (run_ftf_003+) — Verify Negamax Fix Transferred

**Problem:** `run_ftf_002` (pre-negamax) scored 0% vs heuristic for 9/10 snapshots. `run_ftf_003` started but was aborted early (only 1 eval row in its log). A complete ftf run with the negamax fix has not yet been validated.

**Fix:** Start `run_ftf_004` with the same command used for `run_pts_002` but `--mode first_to_four`.

**Expected outcome:** WR vs heuristic should leave 0% within 1,500–2,000 games (doc 13 acceptance gate for P0.4). If it doesn't, the FTF reward shaping (open-3 threat delta only, no survival) needs further tuning.

---

## Priority Order for Next Runs

| # | Item | Effort | Impact |
|:---:|------|:---:|:---:|
| 1 | R1 — Fix final Elo null | tiny | Complete registry |
| 2 | R7 — Train 9×9 agent | medium (run) | Required for submission |
| 3 | R8 — FTF run_ftf_004 | medium (run) | Validate FTF mode fix |
| 4 | R5 — Multi-seed validation | medium (3 runs) | Credible evidence for report |
| 5 | R2 — Benchmark variance fix | small code | Robustness |
| 6 | R6 — Band threshold calibration | tiny | Label accuracy |
| 7 | R3 — n-step returns | medium code | Faster convergence |
| 8 | R4 — PER | large code | Sample efficiency |

---

## Key Number for the Project Report

The Elo column in `training_log.csv` directly answers "does performance improve with experience?":

```
Game     0:   795   →   Final: 1,190   (+394 Elo, +49%)
```

This is a monotone, quantitative, directly comparable skill curve across all 100 evaluation checkpoints. It is the cleanest single graph for the 30% evaluation criterion in the ITRI 616 brief.
