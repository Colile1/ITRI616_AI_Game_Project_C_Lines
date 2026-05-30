# Run `run_ftf_002` — Post-Run Analysis Report

**Date:** 2026-05-30
**Board size:** 8×8 | **Mode:** `first_to_four` | **Total training games:** 10,000
**Benchmark opponent:** Alpha-Beta depth-4 (`alphabeta_d4`)
**Network:** ResNet v1 — 4 residual blocks, 64 channels, 10-channel state encoding
**Run duration:** ~7 hours (2026-05-30 02:56 → 09:23 UTC)
**Throughput:** ~1,400 games/hour (steady)

---

## 1. What the Numbers Say

### 1.1 Snapshot Performance Summary

| Gen | Games | Label | Band | WR vs Random | WR vs Heuristic |
|-----|-------|-------|------|:---:|:---:|
| gen_001 | 1,000 | Apprentice | novice | 56% | 0% |
| gen_002 | 2,000 | Beginner | novice | 61% | 0% |
| gen_003 | 3,000 | Improver | easy | 50% | 0% |
| gen_004 | 4,000 | Cadet | easy | 38% | 0% |
| gen_005 | 5,000 | Tactician | medium | 73% | 0% |
| gen_006 | 6,000 | Adept | medium | 38% | 0% |
| gen_007 | 7,000 | Tactician | medium | 44% | 0% |
| gen_008 | 8,000 | Veteran | hard | 38% | 0% |
| gen_009 | 9,000 | Strategist | hard | 81% | 0% |
| gen_010 | 10,000 | Champion | master | **95%** | **6%** |

Unlike `run_pts_001` where the final model was the worst, **gen_010 is the best model here** — the run improved to the very end. However, the win rate vs heuristic is 0% for nine out of ten snapshots, and the final value of 6% is barely above noise.

### 1.2 Benchmark vs Alpha-Beta depth-4 — Complete Picture

**The agent never won a single benchmark game across the entire 10,000-game run.** `benchmark_result = 0.0` for all 101 checks without exception.

Additional signals from the benchmark log:
- `benchmark_score_diff` is almost invariably **−1.0125** — the same value, almost every check, for the full run. The alpha-beta wins by the same margin every single time.
- `benchmark_episode_length` ranges from **8 to 18 moves**. The alpha-beta finds a 4-in-a-row in under 18 moves. By contrast, the agent's training games average 30–38 moves in the same phase. The DQN is losing very fast to a fixed tactical pattern.

### 1.3 Training Curve — Three Distinct Phases

**Phase A — Flat wandering (games 0–4,900):**
```
WR vs random:    41–63%, mostly flat, no clear upward trend
WR vs heuristic: 0–4%, effectively zero throughout
Episode length:  30–39 moves, slowly lengthening (longer games = no early wins)
Loss:            Drops from 0.014 at game 300 to <0.001 by game 2,000
```
The agent is learning something (loss falls) but it isn't translating to wins against the heuristic. Win rate vs random is barely above the 50% chance line.

**Phase B — Deep trough (games 3,400–5,400):**
```
WR vs random:    14%–38%  ← worst stretch of the run
Episode length:  34–39 moves ← longest games of the run
Loss:            <0.0005, essentially converged
```
The agent is actively getting *worse* vs random during this period. Episode lengths are at their maximum, indicating the agent is failing to close out games. This is the mid-training collapse, comparable to `run_pts_001`'s late-training collapse — except here it happens in the middle.

**Phase C — Late breakthrough (games 8,200–9,999):**
```
WR vs random:    83%→91%→99%→99%→98%→99%→95%→99%→92%→96%
WR vs heuristic: 0→0→2%→6%→10%→2%→4%→6%→10%→2%→8%
Episode length:  29→25→22→17→20→16→18→16→21→20 moves ← major drop
```
Something clicked after game 8,200. The episode length drop from ~35 moves to 15–22 is the clearest signal of what happened: **the agent learned to complete 4-in-a-rows quickly**. It stopped optimising for line score accumulation and started finishing games. This is the mode-correct behaviour for `first_to_four`.

### 1.4 Loss Behaviour

Loss is extremely low throughout — two orders of magnitude lower than `run_pts_001`. By game 5,000 it is consistently below 0.0002, and by late training it sits at 0.000013–0.000060. A loss this low says the network's Q-value predictions are internally consistent, but says nothing about whether those Q-values are strategically correct. The network converged to a stable (but initially wrong) belief about the game.

---

## 2. What Actually Happened — Root Cause Analysis

### Problem 1: Reward Shaping Is Misaligned with the First-to-Four Objective

**This is the primary root cause of the entire run.**

`env.py` applies delta-score shaping in BOTH modes:
```python
# Delta-score shaping — applied to BOTH modes.
return (p1_now - p1_prev - (p2_now - p2_prev)) * STEP_REWARD_SCALE
```

`compute_scores` rewards lines of length 3, 4, 5, 6, 7, 8 according to `SCORE_FOR_LENGTH`. In `points_full` mode this is correct — the objective IS cumulative line score. In `first_to_four` mode the objective is completely different: **be the first to complete exactly one line of length ≥ 4, regardless of score**. Building three 3-in-a-rows earns positive shaped reward but contributes nothing to winning. Completing a 4-in-a-row earns the same shaped reward as completing a 3-in-a-row plus 0.75 more — but it also ends the game.

The agent spent roughly the first 8,000 games optimising the wrong objective (score accumulation) because that is what the shaped reward tells it to do. The eventual breakthrough at game 8,200 was not a design success — it was the agent accidentally discovering 4-in-a-row completion through accumulated self-play experience despite the misleading shaping signal.

This also explains:
- Why episode length went UP during the trough (agent was building 3s instead of completing 4s)
- Why heuristic win rate was 0% for so long (the heuristic explicitly blocks 4-threats; an agent building 3s is harmless to it)
- Why the benchmark score_diff is identically −1.0125 every check (alpha-beta plays the same fork-building sequence every time, and the DQN — focused on score accumulation — never learns to block it)

### Problem 2: Benchmark Never Won — Alpha-Beta Plays a Forced Line

The constant benchmark episode length (8–18 moves) and constant score differential (−1.0125) together indicate the alpha-beta is executing essentially the same tactical sequence every match. In `first_to_four` mode, depth-4 alpha-beta can find and execute fork threats reliably. The DQN never learns to disrupt this because:
1. The reward shaping doesn't signal 4-threat blocking as important.
2. The pool of self-play opponents also doesn't create fork threats until very late training.

Winning even once against alpha-beta depth-4 in ftf mode would require the DQN to recognise and block fork patterns — a threat-defence skill that was simply never developed during this run.

### Problem 3: Mid-Training Collapse (Games 3,400–5,400)

Win rate vs random dropped as low as 7% at game 4,900 — worse than a random agent — while loss was near zero. This is a self-play collapse: the snapshot pool contained agents that had learned a particular pattern of 3-building moves. Self-play against these opponents produced transitions that reinforced that pattern. When random agents started placing pieces in positions that disrupted the pattern, the DQN failed completely. Loss was low because the network's internal Q-predictions were consistent — it had converged to a degenerate local policy that happened to lose to random play.

The LR decay at game 4,000 (1e-3 → 5e-4) coincides with the onset of this collapse. The lower LR slows adaptation to new experience, extending how long the degenerate policy persists before correction.

### Problem 4: Difficulty Band Labels Are Wrong (Same Bug as run_pts_001)

The band is still assigned by ordinal position:
- gen_004 (38% vs random) is labeled "easy / Cadet" — it is genuinely novice-level.
- gen_008 (38% vs random) is labeled "hard / Veteran" — same actual performance as gen_004 but different label.
- gen_010 (95% vs random, 6% vs heuristic) is labeled "master / Champion" — the label happens to be correct this time, but only by coincidence.

The root bug in the registry-writing code is unchanged from `run_pts_001`.

### Problem 5: Benchmark Is a Single Game (Same Bug as run_pts_001)

The benchmark_log confirms the same single-game-per-check problem. In `first_to_four` mode this is even more severe than in `points_full`: ftf games are shorter and more tactically decisive. A single game result (0 or 1) tells you nothing about the DQN's true win rate against the benchmark.

### Problem 6: The Late Surge Is Fragile

The strong late performance (95–99% vs random, 15–22 move episodes) happened over just the final ~1,500 games under very low LR (2.5e-4). This is the same phase that caused catastrophic forgetting in `run_pts_001`. The DQN appears stable here only because the ftf mode's reward structure finally aligned with what the agent had learned. There is no guarantee this stability would persist with more training, and the policy has not been verified to be robust across diverse starting conditions.

---

## 3. What Went Right

1. **The final model is genuinely strong.** gen_010 at 95% vs random with 15–22 move wins is the best result in any run so far. Unlike `run_pts_001`, the training improved to the very end.

2. **The late breakthrough is real.** The episode length drop from 35 to 15–22 moves is a structural change in behaviour — the agent is not just getting lucky against random opponents, it has learned to build 4-in-a-rows efficiently.

3. **Training was stable and fast.** ~1,400 games/hour, 7-hour total wall clock, no crashes or illegal-move spikes. The infrastructure is solid.

4. **The run revealed the reward-shaping bug clearly.** The 8,000-game delay before the agent learned ftf-correct behaviour is direct evidence that delta-score shaping in ftf mode teaches the wrong objective. This is a high-value finding.

---

## 4. Comparison with run_pts_001

| Aspect | run_pts_001 (points_full) | run_ftf_002 (first_to_four) |
|--------|--------------------------|------------------------------|
| Final model | WORST in run (15%/10%) | BEST in run (95%/6%) |
| Best snapshot | gen_004 at game 4,000 | gen_010 at game 10,000 |
| WR vs heuristic (peak) | 80–100% in games 2800–7300 | 6–10% only in final 500 games |
| Benchmark wins | 42% peak (42/101 checks) | 0% (0/101 checks) |
| Primary failure | Late-training catastrophic forgetting | Wrong reward shaping for the mode |
| Episode length | Always 64 (full board, by design) | 30–39 → 15–22 (mode-correct late) |
| Loss level | 0.001–0.04 | 0.000013–0.014 |
| Duration | ~16 hours | ~7 hours |

The two runs failed in opposite directions: `pts_001` was strong in the middle and collapsed at the end; `ftf_002` was weak in the middle and surged at the end. The root cause in both cases is a mismatch between what the reward signal teaches and what the game mode actually requires.

---

## 5. Summary Table

| What | Finding |
|------|---------|
| **Best model produced** | gen_010 (game 10,000): 95% vs random, 6% vs heuristic |
| **Weakest snapshot** | gen_004 and gen_008 (both 38% vs random, 0% vs heuristic) |
| **Benchmark result** | 0 wins in 101 checks — agent never beat alpha-beta depth-4 |
| **Primary failure** | Delta-score reward shaping teaches score accumulation, not first-to-four |
| **Secondary failure** | Single-game benchmark, wrong difficulty band labels (same bugs as pts_001) |
| **Key observation** | Episode length drop (35→15 moves) in late training is mode-correct behaviour |
| **Mid-training collapse** | Games 3,400–5,400: WR vs random falls to 7%, caused by self-play pool collapse + LR decay |
| **Heuristic gap** | WR vs heuristic essentially 0% for 8,500 games; peaks at 10% briefly |
| **What to keep** | gen_010 weights, late-training episode-length pattern as a diagnostic metric |
| **What must change** | Reward shaping for ftf mode, benchmark game count, band assignment |
