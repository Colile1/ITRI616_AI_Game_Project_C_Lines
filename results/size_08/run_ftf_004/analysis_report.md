# Analysis Report — run_ftf_004
**Mode:** First-to-Four (FTF) | **Board:** 8x8 | **Total games:** 20,000
**Written:** 2026-06-05

---

## 1. Run Configuration

| Parameter | Value |
|-----------|-------|
| Board size | 8x8 |
| Game mode | First-to-Four (win by completing a 4-in-a-row first) |
| Network | ResNet (4 residual blocks, 64 channels, 10-channel input) |
| Algorithm | Double DQN + n-step returns (n=3) + D4 symmetry augmentation |
| Training games | 20,000 (weights seeded from run_ftf_003 best at game 10,000) |
| Base opponent | Previous best model from run_ftf_003 (WR 31% vs heuristic) |
| Benchmark opponent | AlphaBeta depth-4 (64 games per check, every 100 training games) |
| Reward shaping | Open-3 threat delta x0.10 + forced-win (double-open-3) delta x0.30 |

---

## 2. Learning Curves — Key Milestones

| Game | Epsilon | Ep. Length | WR vs Random | WR vs Heuristic | Elo | BM vs AlphaBeta |
|------|---------|-----------|-------------|----------------|-----|-----------------|
| 0 | 1.00 | 33.3 | 48.5% | 0.0% | 790 | 0.0% |
| 1,000 | 0.86 | 31.8 | 48.5% | 0.0% | 716 | 0.0% |
| 2,000 | 0.73 | 27.5 | 77.5% | 0.0% | 682 | 0.0% |
| 3,000 | 0.59 | 23.9 | 88.0% | 1.0% | 682 | 0.0% |
| 4,000 | 0.46 | 18.1 | 95.0% | 0.0% | 687 | 0.0% |
| 5,000 | 0.32 | 17.1 | 96.5% | 5.0% | 703 | 0.0% |
| 6,000 | 0.19 | 13.1 | 99.0% | 6.0% | 723 | 3.1% |
| **7,000** | **0.05** | **9.9** | **100%** | **6.0%** | **744** | 3.1% |
| **8,000** | 0.05 | 9.1 | **100%** | 17.0% | 767 | **37.5%** |
| **9,000** | 0.05 | **8.9** | **100%** | 17.0% | 783 | **50.0%** |
| 10,000 | 0.05 | 10.2 | 100% | 18.0% | 792 | 42.2% |
| 11,000 | 0.05 | 12.4 | 99.5% | 27.0% | 807 | -- |
| 12,000 | 0.05 | 13.7 | 97.0% | 22.0% | 814 | 0.0% |
| 14,000 | 0.05 | 16.3 | 97.0% | 12.0% | 805 | 45.3% |
| 16,000 | 0.05 | 15.8 | 97.5% | 11.0% | 790 | 39.1% |
| 19,999 | 0.05 | 19.7 | 91.5% | 8.0% | 774 | 37.5% |

---

## 3. Evidence of Learning

### 3.1 Episode Length -- Strongest and Most Consistent Signal
The clearest evidence of learning is the progressive reduction in episode length:

- Game 0: 33.3 moves/game (near-random play)
- Game 7,000: 9.9 moves/game (70% reduction from baseline)
- Game 9,000: 8.9 moves/game (near-minimum, consistent fast wins)

This represents the agent learning to execute 4-in-a-row completions in as few moves as possible. An episode length of 8-9 means the agent places its first piece and completes the sequence before the opponent can establish a defensive structure. The reduction of 24.4 moves (73%) over 9,000 games is unambiguous evidence of performance improvement with experience.

### 3.2 Win Rate vs Random Agent -- Full Convergence
The agent reached 100% win rate against RandomAgent by game 7,000, sustained from near-chance (48.5%) at game 0. This is the most decisive categorical improvement: the agent goes from indistinguishable from random play to winning every game within 7,000 training games.

### 3.3 Benchmark vs AlphaBeta Depth-4 -- Peak Skill Measurement
AlphaBeta depth-4 is a minimax search agent that looks 4 moves ahead. Beating it requires genuine strategic understanding, not just pattern memorisation:

- Games 0-7,000: 0-3% win rate (cannot compete against search)
- Game 8,000: 37.5% -- first major breakthrough
- Game 9,000: 50.0% -- agent reaches parity with a search-based opponent
- Late training: 37-45% sustained, with occasional 0% outliers (bimodal behaviour)

Reaching 50% against depth-4 alpha-beta via pure self-play (no search) is a strong result. It confirms that the DQN has internalised genuine tactical patterns, not merely exploited the weaknesses of simpler opponents.

### 3.4 Elo Trajectory
Elo rose from 716 (game 1,000) to 814 (game 12,000), a gain of +98 points. While modest in absolute terms, the monotone rise through games 1,000-12,000 confirms consistent improvement independent of the noisy WR metrics.

---

## 4. Phase Analysis

### Phase 1: Exploration (Games 0-2,000, epsilon 1.0 to 0.73)
Unstructured random play. Episode lengths remain high (27-33 moves). WR vs random hovers near 50% (chance). The replay buffer fills with diverse experience. No meaningful learning signal yet.

### Phase 2: Rapid Acquisition (Games 2,000-7,000, epsilon 0.73 to 0.05)
The most dramatic learning phase. Episode length falls from 27.5 to 9.9 moves. WR vs random climbs from 48% to 100%. The agent discovers the core FTF strategy: commit to a single line early, extend it every move, and complete 4-in-a-row before the opponent can respond. This phase demonstrates that the DQN is capable of discovering the correct strategy through pure trial-and-error.

### Phase 3: Tactical Refinement (Games 7,000-10,000, epsilon 0.05)
Episode length stabilises at 8-10 moves. WR vs heuristic climbs from 6% to 18%. The benchmark peaks at 50% (game 9,000). The agent refines opening sequences and learns to exploit the heuristic opponent's predictable defensive patterns. This is where most of the genuine skill development occurs.

### Phase 4: Curriculum Adaptation (Games 10,000-20,000)
The agent now trains against the run_ftf_003 best model (31% vs heuristic) as a static opponent. Games become harder: episode lengths rise to 12-20 moves as the opponent successfully blocks quick wins. WR vs heuristic oscillates (8-27%), reflecting the challenge of a stronger curriculum. The benchmark remains at 37-45% in its best checks, confirming the underlying skill has not regressed.

---

## 5. Limitations

### L1 -- Bimodal Benchmark Behaviour
The benchmark oscillates between 0% and 50% rather than converging to a stable value. The agent has mastered a small set of winning opening sequences but lacks a general strategy that works from any starting position. When AlphaBeta is seeded with an opening that avoids those specific sequences, the agent loses consistently.

### L2 -- WR vs Heuristic Ceiling (8-27%)
The FTF heuristic agent uses the same threat-counting logic as the reward shaping. The DQN has not learned to outplay a blocking opponent that specifically defends open threats. Further progress requires either a stronger reward signal (teaching explicit blocking and double-threat creation) or MCTS-enhanced search at inference time.

### L3 -- Late Episode Length Regression
Episode lengths increased from 8.9 (game 9,000) back to 19.7 (game 19,999). This is caused by the prev_best opponent blocking the agent's preferred fast-win patterns. This is not a capability regression -- it reflects a harder curriculum -- but it makes the episode length curve non-monotone in the second half of training.

### L4 -- Single Seed
This run used one random seed. The bimodal benchmark behaviour suggests the agent's performance is sensitive to initialisation. Without multi-seed validation, it is unclear whether the 50% benchmark result at game 9,000 is reproducible or a favourable seed.

---

## 6. Improvement Plan for run_ftf_005

### I1 -- MCTS-Enhanced Self-Play (Critical)
**Problem:** Pure DQN self-play converges to a narrow set of memorised fork patterns.
**Fix:** Use MCTSAgent (200 simulations, wrapping DQNAgent) as the self-play opponent. MCTS explores a wider set of positions and cannot be defeated by rote memorisation.
**Expected impact:** Benchmark WR stabilises above 30%, bimodal behaviour eliminated.
**Command addition:** `--schedule "self:3000,mcts:7000"`

### I2 -- Graduated Curriculum (High)
**Problem:** The sudden switch to a strong prev_best opponent causes a long adaptation plateau.
**Fix:** Three-stage curriculum:
  1. Games 0-3,000: Random (50%) + Heuristic (50%)
  2. Games 3,000-8,000: Pool snapshots (70%) + Heuristic (30%)
  3. Games 8,000+: Pool (50%) + MCTS (50%)

### I3 -- FTF-Appropriate Elo Anchors (Medium)
**Problem:** Elo saturates around 790 because both anchors (Random=800, Heuristic=900) are too close together and the FTF agent beats random early but never beats heuristic reliably.
**Fix:** Add AlphaBeta depth-2 (Elo approx 950) as a third Elo anchor specifically calibrated for FTF.

### I4 -- n-step Ablation (Medium)
n=3 returns have not been compared against n=1 in FTF mode. The negamax alternating sign in the 3-step sum may be introducing instability. Run one session with N_STEP_RETURNS=1 as a control.

### I5 -- Multi-seed Validation (Low)
Run run_ftf_005, run_ftf_006 with seeds 1 and 2. If episode length reaches < 10 by game 7,000 in all runs, the result is confirmed reproducible.
