# Improvement Plan — run_pts_005

**Written:** June 2026
**Based on:** `results/size_08/run_pts_004/analysis_report.md`
**Starting point:** `models/size_08/run_pts_004/gen_012/weights.pt` (game 24,000, ELO 1,097.4)

---

## What is Already Working (do not change)

- Lower initial LR (2e-4) when fine-tuning from a converged policy — confirmed effective in run_pts_004 Segment 1.
- Plateau-based LR decay — drove ELO from 813 → 1,097 cleanly in Segment 2.
- 10% anchor opponent mix (heuristic/random) — re-stabilised the agent in the Phase 5 instability.
- `--load-weights` + `--start-game` continuation pattern — works, but needs ELO carry-forward (see R1).
- BATCH_SIZE=128, GRADIENT_STEPS_PER_GAME=4 — no issues observed.

---

## R1 — Carry Forward ELO on Restart (Critical — code fix)

**Problem:** When `--load-weights` is used with `--start-game`, the Elo tracker resets to `800.0` regardless of the loaded model's actual Elo. In run_pts_004, this caused the log to show ELO 1,015 → 813 (false regression), and wasted ~200 games re-warming the LR.

**Fix:** Add `--elo-start FLOAT` CLI parameter to `train.py`. When provided, the training loop initialises `agent_elo` to this value instead of 800.0. Also attempt to auto-read ELO from the metadata.json sibling of `--load-weights` if `--elo-start` is not given.

**Status:** Implemented before run_pts_005.

---

## R2 — Raise LR Floor (High Priority — config change)

**Problem:** At LR=2.5e-5 (the fourth decay step), the agent entered a 1,500-game instability period (games 21,300–22,800, ELO drop 1,058→1,020). The LR was too small to correct the policy against its growing pool of strong opponents, causing oscillation.

**Fix:** Set `min_lr = lr_start * 0.25` instead of `lr_start * 0.125`. This limits the LR to two halving steps (e.g. 2e-4 → 1e-4 → 5e-5) rather than three. The fourth halving step (5e-5 → 2.5e-5) is where instability occurred.

**For run_pts_005 specifically:** Use `--lr-start 2.5e-5`. Since gen_012 ended at LR=2.5e-5, starting there avoids a sharp LR step. The min_lr floor will be `2.5e-5 * 0.25 = 6.25e-6` — but the plateau mechanism will only decay if WR_h stalls, so in practice LR will remain at 2.5e-5 if the agent keeps improving.

---

## R3 — Double Benchmark Games per Check (High Priority — config change)

**Problem:** 64 games per benchmark check produces structural 0%/100% swings in the AlphaBeta-d4 log that do not reflect real policy changes. The benchmark graph is effectively unusable for tracking progress.

**Fix:** Change `BENCHMARK_GAMES_PER_CHECK` from 64 to 128 in `config.py`. This halves sampling variance (σ ≈ 4.4% at 50% WR instead of 6.2%) and eliminates the most extreme seed-driven spikes.

**Status:** Implemented before run_pts_005.

---

## R4 — Elo-Based Best-Model Gate (Medium — code improvement)

**Problem:** `best/weights.pt` saves the model the first time WR_h reaches 100% and never updates again (since WR_h can't exceed 100%). In run_pts_004, the best checkpoint is game 17,100 — a weaker model than gen_012 (game 24,000, ELO 1,097).

**Fix:** Change the best-model condition from `wr_h > best_wr_heuristic` to `wr_h > best_wr_heuristic OR (wr_h == best_wr_heuristic AND agent_elo > best_elo)`. Once WR_h saturates at 100%, Elo becomes the tiebreaker.

**Status:** Implemented before run_pts_005.

---

## R5 — Use gen_012 as prev_best Opponent (High Priority — run config)

**Problem:** run_pts_004 did not use `--prev-best`. The training curriculum was entirely pool-based (self-play vs accumulated snapshots). No fixed strong opponent was present to maintain consistent pressure.

**Fix:** Pass `--prev-best models/size_08/run_pts_004/gen_012/weights.pt` for run_pts_005. This introduces a frozen copy of the best current policy as the base curriculum opponent (replacing the 10% random anchor fraction for the non-anchor games). The agent must constantly beat its own peak to make progress.

---

## Priority Order

| # | Item | Type | Impact |
|:---:|------|:---:|:---:|
| 1 | R1 — ELO carry-forward | code | Correct Elo tracking |
| 2 | R3 — 128 benchmark games | config | Usable benchmark graph |
| 3 | R4 — Elo tiebreaker for best checkpoint | code | Best weights = truly best |
| 4 | R2 — LR floor raise | run config | Prevent Phase 5 recurrence |
| 5 | R5 — prev_best opponent | run config | Stronger curriculum |

---

## run_pts_005 Launch Command

```powershell
python -m src.training.train `
  --size 8 `
  --mode points_full `
  --games 34000 `
  --run-id run_pts_005 `
  --load-weights "models/size_08/run_pts_004/gen_012/weights.pt" `
  --start-game 24000 `
  --lr-start 2.5e-5 `
  --elo-start 1097.4 `
  --prev-best "models/size_08/run_pts_004/gen_012/weights.pt" `
  --benchmark alphabeta_d4
```

**What this does:**
- Continues from gen_012 at game 24,000 with the correct LR (2.5e-5) and ELO (1,097.4) carried forward.
- Uses gen_012 as the static curriculum opponent (`--prev-best`) — the agent must beat its own peak.
- Runs 10,000 more games (to game 34,000), targeting ELO > 1,150.
- Benchmark checks every 100 games with 128 games each (after config change R3).

**Expected outcome:**
- ELO: 1,097 → 1,150+ (if the curriculum pressure from prev_best is productive)
- WR_h: sustain 98–100% throughout
- vs AlphaBeta-d4: 85–100% with a cleaner, less noisy signal

**Key success gate:**
- ELO > 1,100 by game 26,000 (2,000 games in): confirms the LR carry-forward is working.
- If ELO drops below 980 at any point: stop and investigate curriculum pressure from prev_best.
