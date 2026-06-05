# Improvement Plan: Lessons from `run_ftf_003`

**Written:** 2026-06-02
**Based on:** `results/size_08/run_ftf_003/analysis_report.md`
**Cross-reference:** `results/size_08/run_ftf_002/improvement_plan.md`, `docs/deliverables/in_md_format/13_ML_Training_Improvement_Plan.md`

---

## What is Already Working (keep)

- Negamax TD target, Double DQN, Huber loss — all confirmed working
- Threat-delta reward shaping (open-3) — producing fast 4-in-a-row completions
- Early-loss synthetic terminal transition — confirmed: episode length drops to 8–10 moves
- Plateau-based LR decay — conceptually correct; the misfiring is a metric problem (see F1)
- Performance-based band assignment — correctly labels all snapshots novice (honest)

---

## F1 — Replace Elo with Episode Length as the Primary FTF Metric

**Problem:** The Elo tracker pits the agent against two anchors — Random (800) and Heuristic (900). In FTF mode, near-100% wins against Random cancel with near-90% losses against Heuristic, leaving Elo flat at ~790 for the entire run. This makes Elo useless as a skill signal and causes the plateau-based LR scheduler to decay LR to minimum by game 4,200.

**Fix:** For FTF mode, use **mean episode length** as the primary metric driving both the plateau detector and the best-model checkpoint:
- A decreasing mean episode length = the agent is winning faster = improving
- Gate the plateau detector on `mean_ep_len` (lower is better) instead of Elo
- Best-model checkpoint: save when `mean_ep_len` hits a new minimum (not when WR_heuristic peaks)
- Elo for FTF should use a FTF-appropriate heuristic anchor, not the pts-mode heuristic

**Confirm:** LR should hold at 1e-3 for at least 5,000 games. Plateau decay should fire when episode length stops decreasing, not when Elo stops rising.

---

## F2 — Add Open-4 Threat Counting to Reward Shaping (High Priority)

**Problem:** The reward shaping uses open-3 threats (3-in-a-row with a free end). But the direct precursor to a 4-in-a-row win is an **open-4** (also called a "forced win" — a line where placing the 4th piece wins immediately). Rewarding open-4 creation would teach the agent specifically about the critical last step before winning.

**Fix:** Add a second threat term:
```python
open3_delta * FTF_THREAT_SCALE_3 + open4_delta * FTF_THREAT_SCALE_4
```
where `FTF_THREAT_SCALE_4 >> FTF_THREAT_SCALE_3` to emphasise the winning threat. Also add a blocking bonus: large negative reward when the opponent creates an open-4 (i.e. the agent failed to block).

`scoring.py` already has `count_open_threats(board, player, run_length)` — passing `run_length=3` gives open-3s; passing `run_length=3` but checking the specific "one move from 4" definition is what's needed.

**Confirm:** WR_heuristic should start climbing within 1,500 games of the next ftf run (doc 13 acceptance gate).

---

## F3 — Fix the Difficulty Ladder (Medium Priority)

**Problem:** All 10 snapshots are labeled "novice" because WR_heuristic never reaches 25%. This is correct reporting but creates a flat, unusable difficulty system for players.

**Fix options:**
- **Option A:** Use episode length to define bands for FTF. Suggested thresholds: `mean_ep_len > 25 = novice`, `15–25 = easy`, `10–15 = medium`, `< 10 = hard/master`. gen_007–gen_010 (episode length ~9–10) would be labeled hard/master under this scheme.
- **Option B:** Use WR_random as the band criterion instead of WR_heuristic for FTF mode. gen_007–gen_010 at 100% WR_random would be "master."
- **Option C:** Use mode-appropriate Elo anchors (Random=800, FTF-calibrated anchor=850) that reflect FTF difficulty levels.

Recommend Option A as the most meaningful for player experience.

---

## F4 — Address the Bimodal Benchmark Policy (Medium Priority)

**Problem:** The benchmark oscillates between 0% and 87.5%. The agent has one strong fork pattern that works from specific positions but no general strategy. Seeds that start in other configurations produce complete losses.

**Root cause:** The self-play pool at game 7,000+ likely converged to a narrow set of positions. The agent's Q-function is overfit to the few positions it encounters most often in self-play.

**Fix options:**
- **Option A:** Increase pool diversity by keeping more pool snapshots (raise `MAX_POOL_SIZE` from 20 to 30) and reducing recency bias (`RECENT_POOL_BIAS` from 0.7 to 0.5).
- **Option B:** Add a FTF-specific alpha-beta opponent (depth-2) as a permanent 5% anchor alongside random/heuristic. Games against it expose the agent to adversarial opening sequences consistently.
- **Option C:** Diversify benchmark seeds by not using the `game_idx * 1000 + g` seed formula — use truly random seeds so a single bad seed doesn't dominate.

Recommend all three.

---

## F5 — n-step Returns for FTF Credit Assignment (Medium Priority)

**Problem:** FTF games ending in 8–10 moves still require the loss signal from move 8 to propagate back to move 1. With 1-step TD and γ=0.99, the terminal reward discounted back 8 steps is `0.99^8 ≈ 0.92` — reasonably close, but the current synthetic-loss-transition workaround could be replaced by proper 3-step returns.

**Fix:** Implement n=3 returns in `self_play.py`. With 3-step returns, a loss at move 8 propagates to move 5 directly, reducing the effective credit-assignment depth by 3×. This also eliminates the need for the synthetic terminal transition hack added to `train.py`.

**Confirm:** WR_heuristic should start climbing earlier without the synthetic transition, and the bimodal benchmark variance should reduce.

---

## F6 — Tune the Early-Loss Penalty Threshold (Low Priority)

**Problem:** `FTF_EARLY_LOSS_TURNS = 8` was calibrated for early training when games against random opponents lasted 30+ moves. Now that the self-play games themselves last 8–10 moves, almost every loss is "early" and receives the −1.5 penalty. This may be discouraging the agent from playing at all in positions where it cannot win quickly.

**Fix:** Make the threshold relative to the current mean episode length, e.g. `threshold = max(8, int(recent_mean_ep_len * 1.2))`. As games shorten, the "early loss" threshold should adjust accordingly.

---

## F7 — Multi-seed Validation (Low Priority)

**Problem:** Like run_pts_002, this is a single-seed run. The bimodal benchmark suggests the policy is sensitive to initialisation.

**Fix:** Run run_ftf_004 with seeds 1, 2, 3 once the above fixes (F1–F4) are applied, and report mean ± std for episode length and WR_heuristic.

---

## Priority Order for Next FTF Run (run_ftf_004)

| # | Item | Effort | Impact |
|:---:|------|:---:|:---:|
| 1 | F1 — Episode length as primary FTF metric | small code | Fixes LR scheduler, fixes best-model checkpoint |
| 2 | F2 — Open-4 threat shaping | small code | Teaches the critical last step to winning |
| 3 | F3 — FTF-appropriate difficulty bands | small code | Fixes player-facing UI |
| 4 | F4 — Diversify pool + benchmark seeds | small config | Reduces bimodal variance |
| 5 | F5 — n-step returns | medium code | Cleaner credit assignment |
| 6 | F6 — Adaptive early-loss threshold | small config | Prevents penalty misfiring |
| 7 | F7 — Multi-seed validation | runs | Statistical confidence |

---

## The Core Insight

run_ftf_003 proved that the agent **can** learn to play FTF correctly (episode length 8–10, fast wins). The remaining gap — WR_heuristic only 13–31% — is not a fundamental failure of the algorithm; it is a **metric and curriculum problem**: the wrong metric (Elo) caused the LR to decay too early, and the heuristic opponent's blocking ability requires a more targeted reward signal (open-4 specifically) to overcome.

The episode length curve (33 → 8 moves) is the clearest learning signal the project has produced for FTF mode. Build on it.
