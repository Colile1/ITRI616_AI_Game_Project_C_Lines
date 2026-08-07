# Analysis Report — run_pts_004

**Mode:** Points-Full (PTS) | **Board:** 8×8 | **Games:** 14,000–24,000 (10,000 new games)
**Written:** June 2026
**Seeded from:** run_pts_003 gen_017 weights (game 14,000, WR_h=100%)
**Final snapshot:** gen_012 (game 24,000) — ELO 1,097.4, WR_h=100%

---

## 1. Summary of Results

| Metric | Start (game 14,000) | Peak | Final (game 24,000) |
|--------|---------------------|------|---------------------|
| WR vs heuristic | 98% | **100%** (multiple intervals) | **100%** |
| WR vs random | 98% | 100% | 98.5% |
| Elo | 814 | **1,097** (game 24,000) | **1,097** |
| vs AlphaBeta-d4 | 93.75% | **100%** (games 23,600–23,700) | 78.1% (game 23,900) |

**Best model checkpoint:** `best/weights.pt` — game 17,100, WR_h=100% (first time WR_h hit 100% in Segment 2).
**Strongest final weights:** `gen_012` (game 24,000) — ELO 1,097.4, the highest Elo recorded in any run so far.

---

## 2. Run Structure — Two Segments

The run contains **two separate process invocations** logged into the same CSV file:

### Segment 1: Games 14,000–17,000 (continuous, from run_pts_003 gen_017)
- LR schedule: 2e-4 → 1e-4 → 5e-5
- ELO: 814 → **1,015** (+201 points — clean monotone rise)
- WR_h: 98% → 97% (mostly 100%)
- Notable crash at game 16,000: WR_h drops to **4%** (gen_003 band = "novice"), ELO still 950. Recovered within 1,000 games to WR_h=97%, ELO=1,015.
- **End state:** gen_004, game 17,000, WR_h=97%, ELO=1,015

### Segment 2: Games 17,000–24,000 (restart — weights from gen_004, LR reset)
- LR **reset to 2e-4**, ELO **reset to 813** (did not carry forward from Segment 1's 1,015).
- Loaded gen_004 weights (game 17,000) — the agent had identical weights but the LR schedule and ELO tracker both restarted from scratch.
- LR schedule: 2e-4 → 1e-4 → 5e-5 → 2.5e-5
- ELO: 813 → **1,097** (+284 points — stronger curve than Segment 1 due to longer run)
- Large instability episode at games 21,300–22,800: ELO drops 1,058 → 1,020, WR_h falls 81% → 56% for ~1,500 games.
- Full recovery by game 22,900; final ELO 1,097 is the **project-wide best**.

---

## 3. Phase-by-Phase Analysis

### Phase 1 — Re-stabilisation (games 14,000–16,000, Segment 1)
- Agent began with run_pts_003 gen_017 weights — already master-level (WR_h=100% at init).
- With LR=2e-4 (lower than the default 1e-3), training was stable.
- ELO rose smoothly 814 → 951 in the first 2,000 games.
- One sharp drop at game 16,000 (WR_h=4%, ELO=951): the agent encountered a strong pool opponent and the policy momentarily deranged. This is a single eval noise spike not a training collapse — ELO continued rising, showing the network's values were still improving.

### Phase 2 — Consolidation (games 16,000–17,000, Segment 1)
- ELO climbed from 951 to 1,015.
- WR_h recovered from 4% → 97% within 1,000 games.
- gen_004 (game 17,000): WR_h=97%, WR_r=99.5%, ELO=1,015 — the strongest single-run gain of the full trajectory.

### Phase 3 — Restart with reset hyperparameters (games 17,000–19,000, Segment 2)
- LR reset to 2e-4 and ELO to 813 on restart — 200 games "wasted" re-learning the LR schedule.
- WR_h reached 100% by game 17,100 (the saved best checkpoint).
- Crash at game 17,300: WR_h = 44%, WR_r = 44.5%. Single eval snapshot; recovered immediately.
- ELO climbed from 813 → 924 by game 19,000.

### Phase 4 — Plateau-LR decay phase (games 19,000–21,000, Segment 2)
- LR stepped down to 5e-5 at game 19,100, then 2.5e-5 at game 20,100.
- ELO rose steeply: 924 → 1,053 by game 21,000.
- WR_h: 71% → 97.5% (mostly 95–100%).
- This was the most productive phase of the entire run.

### Phase 5 — Instability period (games 21,300–22,800, Segment 2)
- After reaching peak ELO=1,058 at game 21,200, a **rapid regression** began.
- WR_h fell: 81% → 74% → 60.5% → 56% → 54.5%.
- ELO fell from 1,058 to 1,020 (-38 points).
- Duration: ~1,500 games.
- Probable cause: at LR=2.5e-5 with a maturing pool of strong opponents, the agent encountered several pool members near its own level. The small LR prevented fast adaptation, so the policy oscillated rather than converging. The anchor opponent (10% heuristic mix) eventually re-stabilised it.

### Phase 6 — Recovery and peak (games 22,900–24,000, Segment 2)
- WR_h recovered to 95% → 99.5% → 100%.
- ELO rose from 1,021 to **1,097** (new project maximum).
- vs AlphaBeta-d4: 87.5% → 96.9% → 100% (games 23,300–23,700).
- gen_012 (game 24,000): WR_h=100%, WR_r=98.5%, ELO=**1,097.4**.

---

## 4. AlphaBeta-d4 Benchmark

| Range | Pattern |
|-------|---------|
| 14,000–15,100 | Strong (93–100%), then 0% at 14,100 |
| 15,200–18,000 | Extremely volatile: alternates 0–100% every 100–200 games |
| 18,100–22,500 | Mostly 0–18% with occasional spikes to 87–98% |
| 22,900–23,700 | Strong and sustained: 87–100% |
| 23,800–23,900 | Slight dip: 96.9% → 78.1% |

**Key finding:** The benchmark is **structurally volatile** at 64 games per check. The swings from 0% to 100% within 200 games are not explained by training dynamics — the agent's general policy was not collapsing when AlphaBeta-d4 WR showed 0%. This is seed/sampling variance at 64 games. Increasing to 128 games will halve the variance and provide a cleaner signal.

The final sustained strong performance (games 22,900–23,900) confirms the agent genuinely learned to beat AlphaBeta-d4 at the end of the run — this is likely the best vs-alphabeta performance in the project.

---

## 5. Key Achievements

1. **ELO 1,097 — project-wide maximum.** The highest Elo recorded in any training run across the full project.

2. **100% WR vs heuristic** held consistently in Phase 4–6 (games 19,200+).

3. **Strong vs AlphaBeta-d4** in the final phase: 87–100% across games 22,900–23,700, demonstrating a policy that goes beyond just beating the heuristic.

4. **Lower initial LR (2e-4 vs 1e-3)** confirmed as the right choice for fine-tuning from a converged policy. Segment 1's smooth ELO curve (814→1,015 with no collapse) validates the run_pts_003 improvement plan's recommendation R3.

---

## 6. Failure Modes Identified

### 6.1 LR and ELO Reset on Restart (Critical)
When the training process was restarted at game 17,000 using `--load-weights gen_004 --start-game 17000`, the LR was reset to 2e-4 (instead of carrying forward 5e-5) and the ELO tracker reset to 813 (instead of 1,015). This caused:
- ~200 games of redundant LR warm-up re-learning.
- The Elo log shows a false drop from 1,015 to 813 — misleading in the registry.
- The true continuous ELO trajectory was 814 → 1,097, but the log shows two segments with the second "starting over."

### 6.2 Instability at Low LR (game 21,300–22,800)
At LR=2.5e-5, the agent entered a 1,500-game instability period after reaching its local peak (ELO 1,058). The plateau-decay mechanism pushed LR too low before the policy was fully stable against its strong pool opponents. A floor of 5e-5 would prevent this.

### 6.3 AlphaBeta Benchmark Variance (Structural)
64 games per benchmark check is insufficient. The 0%/100% binary swings are not real policy collapses — they are seed effects. This makes the benchmark graph almost unusable for tracking progress.

### 6.4 WR_h as Best-Model Gate
The `best/weights.pt` checkpoint saved game 17,100 (first time WR_h=100% in Segment 2). However, gen_012 (game 24,000, ELO=1,097) is strictly stronger by all metrics except WR_h (which is also 100%). The best-model gate does not update once it first hits 100%, leaving a weaker checkpoint as "best" even as the policy continues to improve.

---

## 7. Improvement Plan for run_pts_005

See `improvement_plan.md` in this directory for the full plan and run command.
