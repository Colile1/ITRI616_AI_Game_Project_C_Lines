# Run `run_ftf_003` — Post-Run Analysis Report

**Date:** 2026-06-02
**Board size:** 8×8 | **Mode:** `first_to_four` | **Total training games:** 10,000
**Benchmark opponent:** Alpha-Beta depth-4 (`alphabeta_d4`, 32 games per check)
**Network:** ResNet v1 — 4 residual blocks, 64 channels, 10-channel state encoding
**Run duration:** ~51 hours (2026-06-01 ~05:45 → 2026-06-02 ~04:23 UTC)
**Throughput:** ~190 games/hour average (slowed by benchmark overhead)

**First FTF run with:** negamax TD target, Double DQN, Huber loss, threat-delta reward shaping (no survival bonus), early-loss synthetic transitions, plateau LR decay, Elo tracking.

---

## 1. Headline Numbers

| Metric | run_ftf_002 (pre-fixes) | run_ftf_003 (this run) | run_pts_002 (points_full) |
|--------|:---:|:---:|:---:|
| WR vs random (final) | 95% | **100%** | 100% |
| WR vs heuristic (final) | 6% | **13–31%** | 100% |
| Elo start → end | — | **790 → 792 (+2)** | 796 → 1,190 (+394) |
| Benchmark peak | 0% (0/101) | **87.5%** | 100% |
| Episode length (final) | 15–22 moves | **8–10 moves** | 64 (full board) |
| All snapshots labeled | varied (wrong) | **All novice** (correct) | Easy → Master |

FTF mode improved significantly over run_ftf_002 but remains far weaker against the heuristic opponent compared to the points_full run. The agent dominates random play and wins games in 8–10 total moves, but the heuristic's explicit 4-threat-blocking keeps WR_heuristic under 31%.

---

## 2. Snapshot Performance

| Gen | Games | Label | Band | WR vs Random | WR vs Heuristic | Elo |
|-----|-------|-------|------|:---:|:---:|:---:|
| gen_001 | 1,000 | Apprentice | novice | 48.5% | 0% | 716 |
| gen_002 | 2,000 | Beginner | novice | 77.5% | 0% | 682 |
| gen_003 | 3,000 | Apprentice | novice | 88.0% | 1% | 682 |
| gen_004 | 4,000 | Beginner | novice | 95.0% | 0% | 687 |
| gen_005 | 5,000 | Apprentice | novice | 96.5% | 5% | 703 |
| gen_006 | 6,000 | Beginner | novice | 99.0% | 6% | 723 |
| gen_007 | 7,000 | Apprentice | novice | **100%** | 6% | 744 |
| gen_008 | 8,000 | Beginner | novice | **100%** | 17% | 767 |
| gen_009 | 9,000 | Apprentice | novice | **100%** | 17% | 783 |
| gen_010 | 10,000 | Beginner | novice | 99.5% | 13% | — |

Every snapshot is labeled **novice** because WR vs heuristic never reached the 25% threshold for "easy." This is correct: the performance-based band assignment works as designed — the agent genuinely never beat the heuristic reliably. However, it creates a flat difficulty ladder with no variation for players in the UI.

---

## 3. Training Curve Analysis

### Most Important Signal — Episode Length

Episode length is the clearest evidence of learning in FTF mode. A shorter episode means the agent is completing 4-in-a-rows faster:

```
Game     0:  33.3 moves  (random play, slow)
Game  1,000: 31.8 moves
Game  2,000: 27.5 moves  ← agent starting to win faster
Game  3,000: 23.9 moves
Game  4,000: 18.1 moves  ← major drop, agent races to 4
Game  5,000: 17.1 moves
Game  6,000: 13.1 moves
Game  7,000:  9.9 moves  ← wins in under 10 total moves
Game  8,000:  9.1 moves
Game  9,000:  8.9 moves
Game  9,999: 10.2 moves
```

By game 7,000 the agent is building a 4-in-a-row in approximately 5 of its own turns. Against random opponents this is a near-perfect win rate. This is the correct, mode-aligned behaviour for first_to_four.

### Win Rate vs Random — Clean Rise

```
Game  0:   48.5%
Game  500:  51.0%
Game 1,500: 61.0%
Game 2,000: 77.5%
Game 2,500: 84.0%
Game 3,000: 88.0%
Game 4,000: 95.0%
Game 5,000: 96.5%
Game 7,000: 100%   ← first 100%, stays there
Final:      100%
```

### Win Rate vs Heuristic — Slow, Partial Climb

```
Games 0–4,100: 0–1%   (essentially zero for first 4,100 games)
Game 4,200:    4%
Game 5,200:   10%     ← first double-digit
Game 5,500:   10%
Game 5,800:   14%
Game 7,600:   20%
Game 9,200:   24%
Game 9,700:   26%
Game 9,800:   31%     ← peak
Final (9999): 18%
```

The heuristic win rate is climbing slowly but has not reached a reliable level. The 31% peak at game 9,800 is real learning — from 0% to 31% represents genuine progress — but it is far from the 100% achieved in points_full mode.

### Elo — Flat, Not Monotone

Unlike run_pts_002, Elo is not a useful skill signal here:

```
Start:    790.2
Game 500: 750.9  ← declining
Game 2500: 678.9  ← trough
Game 5000: 703.3
Game 7000: 744.0
Game 9000: 782.7
Final:     792.4  ← barely above start
```

Net change: +2 Elo over 10,000 games. The issue is structural: the two Elo anchors (Random=800, Heuristic=900) give contradictory signals. The agent wins nearly all random games (pushing Elo up) but loses most heuristic games (pulling it down). The nearly equal magnitude of these effects means Elo barely moves — it is not a useful diagnostic metric for FTF mode.

### LR Decay — Triggered Too Early

The plateau-based LR decay fired 4 times but much earlier than in run_pts_002:

| Decay | Game | LR | Trigger |
|-------|------|----|---------|
| 1 | ~1,000 | 1e-3 → 5e-4 | Elo declining (plateau) |
| 2 | ~2,100 | 5e-4 → 2.5e-4 | Elo still not rising |
| 3 | ~4,200 | 2.5e-4 → 1.25e-4 | Elo barely moving |
| 4 | — | (no further decay) | Elo started rising |

LR reached its minimum (1.25e-4) at game 4,200 — only 42% of training. The remaining 58% ran with a very low LR. Because the plateau detector uses Elo and Elo is broken for FTF mode, the LR decayed too aggressively and too early.

---

## 4. Benchmark vs Alpha-Beta Depth-4

Unlike run_pts_002 (which was consistently 90–100% after game 5,700), the FTF benchmark is highly volatile:

| Game range | Win rate | Character |
|---|---|---|
| 0 – 4,400 | **0%** (all 44 checks) | Complete shutout |
| 4,500 – 6,800 | 0–9.4% | First wins, rare |
| 6,300 | **43.75%** | Sudden spike |
| 6,400 | 0% | Immediate regression |
| 7,200, 7,400-7,600 | ~40% | Sustained plateau |
| 7,700 | **78.1%** | Major breakthrough |
| 7,900 | **81.25%** | Peak-ish |
| 8,800 | **87.5%** | Highest single check |
| 9,300 | 3.1% | Severe regression |
| 9,800–9,900 | 43-44% | Moderate late |

Pattern: the agent has a policy that works well against some opening patterns (win rate 78–87.5%) but is completely beaten by others (0%). The specific random seed of each benchmark check determines almost entirely whether the agent wins or loses. This is the "bimodal policy" problem: the agent has learned one strong opening line but has not generalised across all starting positions.

---

## 5. Root Cause Analysis

### Finding 1: The Agent Learned FTF-Correct Behaviour

Episode length dropping from 33 to 8–10 moves is clear, unambiguous evidence of mode-correct learning. The agent is building 4-in-a-rows in its first 4–5 turns. This is the right strategy. The negamax fix, threat-delta shaping, and early-loss penalty together produced this result — in run_ftf_002 the agent never learned to win quickly.

### Finding 2: The Heuristic Gap Is a Mode Structure Problem

The HeuristicAgent's defining behaviour is to block the opponent's 4-in-a-row threats. In FTF mode, the agent's strategy (race to 4-in-a-row) is directly countered by the heuristic's strategy (block every 4-threat). This is an inherent tension: the optimal FTF policy and the heuristic's blocking policy are adversarial by design. Reaching 31% WR against an opponent whose entire purpose is to stop you is actually meaningful progress — but it is structurally limited by the mode mismatch.

By contrast, in points_full mode the heuristic is trying to score points, and the DQN can outmanoeuvre it in long-term scoring strategies. The goals are less directly antagonistic.

### Finding 3: Elo Is Wrong Metric for FTF

The Elo computation pits the agent against Random (anchor=800) and Heuristic (anchor=900). In FTF:
- Agent dominates Random → would gain hundreds of Elo points
- Agent consistently loses to Heuristic → loses similar Elo

Net result: flat Elo ~790, which tells us nothing about the agent's actual improvement. The correct metric for FTF is **episode length** (did it learn to win faster?) or **win rate vs random** (did it learn to win at all?). Elo should be computed against FTF-appropriate anchors.

### Finding 4: LR Plateau Detector Misfired

Because Elo is the plateau detector's signal and Elo was declining early (random-play warmup losses to heuristic anchor), the LR decayed to its minimum by game 4,200. The remaining 60% of training ran at 1.25e-4. This is not catastrophic — the episode length still dropped to 8–10 — but a higher LR during the heuristic-learning phase (games 5,000–10,000) would likely have produced faster convergence against the heuristic.

### Finding 5: Bimodal Benchmark Policy

The 0% vs 40–87.5% benchmark swing shows the agent has a strong strategy for specific board configurations but no general strategy. The agent likely learned to execute one fork pattern (building toward two simultaneous 4-threats) that works against alpha-beta when the seed creates a favourable opening. In unfavourable seeds, the agent has no fallback.

---

## 6. What Went Right

1. **Episode length target achieved.** 8–10 moves per game by game 7,000 — the agent builds 4-in-a-rows in its first 4–5 turns. This is the single most important behavioral signal for FTF mode.

2. **100% vs random.** WR_random reaches 100% from game 7,000 onward. Perfect calibration against the random baseline.

3. **Benchmark improvement over run_ftf_002.** Previously 0/101 across all checks. This run peaked at 87.5% and hit 40%+ on multiple checks in the late run. Real progress.

4. **No catastrophic forgetting.** Later snapshots consistently better than earlier ones on episode length and WR_random.

5. **Band assignment correct.** All snapshots labeled "novice" because WR_heuristic < 25%. The labels are honest.

---

## 7. Summary Table

| What | Finding |
|------|---------|
| **Best snapshot for play** | gen_008–gen_010 (episode length 9 moves, WR_random 100%) |
| **Most meaningful metric** | Episode length (33 → 8 moves) |
| **Heuristic WR (peak)** | 31% (game 9,800) — climbing but far below pts_002 |
| **Benchmark peak** | 87.5% (game 8,800) |
| **Benchmark character** | Volatile 0–87.5%, bimodal policy |
| **Elo trajectory** | Flat 790→792 — broken metric for FTF mode |
| **LR decay** | Too early (min reached at game 4,200) due to Elo plateau misfiring |
| **Root cause of heuristic gap** | Mode-adversarial structure + bimodal policy |
| **Key win** | Episode length drop confirms mode-correct behaviour |
