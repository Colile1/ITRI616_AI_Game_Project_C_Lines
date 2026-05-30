# Improvement Plan — Lessons from `run_ftf_002`

**Written:** 2026-05-30
**Based on:** `results/size_08/run_ftf_002/analysis_report.md`
**Cross-reference:** `results/size_08/run_pts_001/improvement_plan.md`
**Applies to:** Next `first_to_four` training run and all future ftf runs

Items marked **(SHARED)** were already identified in `run_pts_001` and remain unresolved. Items marked **(NEW)** are specific to what `run_ftf_002` revealed.

---

## F1 — Fix Reward Shaping for First-to-Four Mode (Critical — NEW)

**Problem:** `env.py` applies delta-score shaping in both modes. In `points_full` mode this is correct. In `first_to_four` mode it is wrong: it rewards building lines of any length (score accumulation), but the game objective is to be first to complete a line of length ≥ 4. The agent spent ~8,000 games optimising the wrong objective. This is the single biggest factor in the 0% heuristic win rate for most of the run and the 0/101 benchmark record.

**Fix options to consider (not yet chosen):**

- **Option A — Threat-based shaping:** Replace delta-score shaping in ftf mode with a reward that explicitly values creating open-4 threats and penalises allowing opponent open-4 threats. E.g. `+0.3` for each new open-4 the agent creates, `−0.3` for each new open-4 the opponent gains.
- **Option B — Mode-gated shaping:** In ftf mode, zero out the step reward entirely and rely only on `WIN_REWARD / LOSS_REWARD` at the terminal. This removes the wrong signal. The terminal reward discounting problem (see pts_001 plan I7) becomes more severe but the signal is at least correct.
- **Option C — Completion bonus only:** Give a small positive reward only when the agent completes a line of length ≥ 4 (i.e., the move that triggers `check_terminal_mode1`). All other intermediate rewards are zero. This preserves dense reward where it matters.

Option C is the most direct fix. Option A is richer but harder to tune. Option B is the simplest to implement and rules out the wrong signal as a cause before adding complexity.

**Metric to verify the fix:** WR vs heuristic should start climbing within the first 2,000 games of the next ftf run. If it's still 0% at game 3,000, the shaping is still wrong.

---

## F2 — Add Episode-Length as a Primary Diagnostic (High Priority — NEW)

**Problem:** The episode length drop (35→15 moves) in late training was the clearest signal that the agent had learned mode-correct behaviour. But this was only visible in hindsight from the training log. There was no runtime alert or early-stopping condition based on it.

**Fix:** Track a moving average of episode length (over the last 200 games). In ftf mode, a sustained drop below a threshold (e.g. mean_ep_len < 25 on 8×8) is evidence of genuine policy improvement — the agent is winning quickly. Use this as a secondary performance metric alongside WR. Consider logging it in the benchmark check as well, not just in the training eval.

---

## F3 — Multi-Game Benchmark (Critical — SHARED with pts_001 I2)

**Problem:** 101 benchmark checks, all single-game, all resulted in 0.0. A single ftf game is even less informative than a single pts game — it can end in 8 moves if one side finds a fork. The benchmark log is useless as currently recorded.

**Fix:** Run 32–50 games per benchmark check. Record win fraction, average episode length, and standard deviation. In ftf mode, also record average moves-to-win — a decreasing trend there is as informative as the win rate itself.

---

## F4 — Fix Difficulty Band Assignment (Critical — SHARED with pts_001 I1)

**Problem:** gen_004 and gen_008 both have 38% WR vs random but are labeled "easy" and "hard" respectively. The ordinal-based assignment bug is unchanged from pts_001.

**Fix:** Band must be computed from measured win rates after evaluation, not from the generation number. See pts_001 improvement plan I1 for the specific fix description.

---

## F5 — Best-Model Checkpointing (High Priority — SHARED with pts_001 I3)

**Problem:** This run ended well, so the issue was less visible — but consider: if run_ftf_003 had been a 12,000-game run, the last 2,000 games might have produced a collapse as in pts_001. The gen_010 weights from this run are the only copy of the best policy. There is no separate "best ever" checkpoint.

**Fix:** Track best WR (vs heuristic, or combined score) continuously. Save best weights to `models/size_08/run_ftf_002/best/weights.pt` whenever a new best is set. This is the same fix as pts_001 I3 and is still unimplemented.

---

## F6 — LR Decay Coincidence with Mid-Training Collapse (Medium Priority — SHARED with pts_001 I6)

**Problem:** The mid-training trough (games 3,400–5,400) began almost exactly when LR decayed from 1e-3 to 5e-4 at game 4,000. Lower LR slows the ability to correct the degenerate self-play policy. In pts_001 the second decay triggered collapse; in ftf_002 the first decay deepened a trough.

**Fix:** Consider plateau-based LR decay — only reduce LR when WR vs heuristic has not improved for N consecutive eval intervals, rather than at fixed game-count fractions. This would have kept LR higher during the trough, allowing faster escape.

---

## F7 — Self-Play Pool Quality Gate (Medium Priority — NEW)

**Problem:** During the mid-training collapse (games 3,400–5,400), the snapshot pool contained agents that had learned degenerate 3-building policies. Self-play against these agents reinforced the degenerate behaviour. The pool has no quality filter — a snapshot is added every 1,000 games regardless of whether it represents an improvement.

**Fix:** Only add a snapshot to the self-play pool if it meets a minimum quality threshold (e.g. WR vs random ≥ 50%). A snapshot at 38% WR vs random (as gen_004 is in this run) should not be in the pool as a training opponent — it teaches wrong patterns. The pool should only contain opponents worth beating.

---

## F8 — Extend Epsilon Decay (Medium Priority — SHARED with pts_001 I9)

**Problem:** Epsilon hits its floor (0.05) at game 5,000 — exactly during the worst part of the mid-training trough. For the remaining 5,000 games the agent exploits a potentially degenerate policy with minimal exploration. In ftf mode this is especially harmful because the search space (valid threat lines) is large and the agent needs diverse experience to discover 4-in-a-row completion patterns.

**Fix:** Extend `EPS_DECAY_GAMES` to 7,000–8,000 for a 10,000-game ftf run. The agent needs to keep exploring longer before it can confidently exploit. The late breakthrough happened under very low LR AND near-floor epsilon — more exploration earlier might have triggered it sooner.

---

## F9 — More Evaluation Games (Medium Priority — SHARED with pts_001 I5)

**Problem:** WR vs heuristic fluctuates 0%→10%→0%→6%→10%→2% in the final 1,000 games. With only 50 evaluation games, at 10% true win rate the standard error is ±4.2% — which means the difference between 0% and 10% measured is within measurement noise. We cannot confirm the late improvement is real.

**Fix:** Increase to 200 games vs random and 100 vs heuristic. For ftf mode in particular, the heuristic evaluation needs more games because wins are rare and high variance.

---

## F10 — Verify Policy Stability Before Declaring the Run Complete (Low Priority — NEW)

**Problem:** The late-training surge happened over only the final ~1,500 games. The policy may not be stable — it emerged quickly and was never tested across a longer stability window. If training had continued for another 2,000 games it might have produced a collapse similar to pts_001's end.

**Fix:** After training completes, run a post-hoc evaluation of gen_010 with 500 games vs random and 200 vs heuristic (more games than during training) to confirm the performance is real and not an evaluation artefact. Only promote a model to the difficulty ladder after this extended eval passes.

---

## Priority Order for Next ftf Run

| Priority | Item | Reason |
|:---:|------|---------|
| 1 | F1 — Fix ftf reward shaping | The single biggest cause of failure in this run |
| 2 | F3 — Multi-game benchmark | Benchmark log was completely uninformative |
| 3 | F4 — Fix band assignment | Same bug as pts_001, still unfixed |
| 4 | F5 — Best-model checkpoint | Protects good weights if stability breaks |
| 5 | F2 — Episode length diagnostic | Essential monitoring for ftf mode |
| 6 | F7 — Pool quality gate | Prevents degenerate self-play collapse |
| 7 | F6 — Plateau-based LR decay | Mid-trough LR drop made collapse worse |
| 8 | F8 — Extend epsilon decay | More exploration would help in mid-training |
| 9 | F9 — More eval games | Measurement confidence |
| 10 | F10 — Post-hoc stability eval | Due diligence on final model |

---

## Open Question for Next Run

Should the next ftf run use the gen_010 weights from this run as a starting point (warm-start / transfer learning), or start fresh with corrected reward shaping? The gen_010 policy has already learned fast 4-in-a-row completion. Warm-starting might preserve that while letting corrected shaping clean up the remaining heuristic gap. Starting fresh ensures the shaping fix gets a clean test. Both are valid — the choice depends on whether we want to build on this run's result or establish a clean experimental baseline.
