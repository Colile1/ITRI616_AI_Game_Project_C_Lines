# C_lines: Designing and Training a Self-Play Reinforcement Learning Agent
## ITRI 616 Mini-Project Technical Report

**Student:** *----* *S----* | **Module:** ITRI 616 — Artificial Intelligence 1
**Code repository:** [Submit GitHub link here]

---

## 1. Introduction and Purpose

This project digitises and extends a class of Southern African line-formation games — traditional stone and seed games played across the region where players occupy a grid by forming lines of pieces. The digital variant, named **C_lines**, was purpose-built for this project to serve two goals: (1) to preserve and make accessible a cultural gaming tradition through a free, open-source digital implementation, and (2) to provide a well-defined learning environment for an AI agent whose measurable progress can be compared across training iterations.

C_lines is played on an 8×8 grid. Players alternate placing one piece per turn anywhere on the board. Unbroken lines of three or more pieces in any of the four directions (horizontal, vertical, and both diagonals) score points according to their length — three pieces score 0.25 points, four score 1.0, five score 2.0, and so on up to a maximum of 5.0 for eight in a row. The game ends when the board is completely filled. The player with the higher cumulative line score wins. A three-round mutual-removal tiebreak resolves draws, ensuring a decisive result in every game.

The game was chosen because it has no existing AI opponent, no published strategy, and no solvable compact form — which makes it a genuine challenge for machine learning rather than a textbook exercise.

---

## 2. TEP Framework — Task, Experience, Performance

### 2.1 Task (T)

**Task category:** Sequential decision-making — specifically a finite-horizon Markov Decision Process (MDP). Not classification (no labelled examples of the "correct" move). Not prediction (no continuous variable to forecast). The agent must learn a *policy*: a mapping from board states to actions that maximises expected long-term return.

**Formal MDP definition:**

| Component | Specification |
|---|---|
| State S | All reachable 8×8 board configurations, encoded as a 10-channel float32 tensor. Channels: own pieces, opponent pieces, empty cells, open-3 threats (both players), open-4 threats (both players), immediate-win cells, immediate-lose cells, turn-progress fraction. |
| Action A | 64 cell indices (flat: `i = row × 8 + col`). Illegal cells masked to −∞ before selection. |
| Transition δ | Deterministic: `apply_placement(board, action)`, followed by turn advance and terminal check. |
| Reward R | Per step: `(Δown_score − Δopp_score) × 0.05` — delta-score shaping aligned with the winning objective. Terminal: `+1` win, `−1` loss, `0` draw. |
| Discount γ | 0.99 (games are short — ≤ 64 moves — so future reward should be valued near-equally). |

**Relevance of the task design:** The delta-score shaping is potential-based, meaning it does not change the optimal policy — it only makes the sparse terminal reward dense enough for the agent to credit-assign within a 64-move episode. Without it, the agent would receive no signal until the very last move of every game, making learning impractically slow.

### 2.2 Experience (E)

The agent learns exclusively from **self-play simulated episodes** — there is no human-generated game data.

**Training curriculum (three phases):**

1. **Warm-up (games 0–2,000):** Agent plays against a `RandomAgent` (70%) and `HeuristicAgent` (30%). Bootstraps the replay buffer with diverse, non-degenerate positions.
2. **Self-play (games 2,000–10,000):** Agent plays 45% against pool snapshots (frozen past checkpoints), 45% against a clone of itself, and 10% against permanent anchor opponents (Random/Heuristic). The anchor games prevent the curriculum from collapsing if the pool fills with weak clones.
3. **Snapshot pooling:** Every 1,000 games, the current agent is frozen and added to a pool (max 20 snapshots). This provides a self-organising difficulty ladder — as training progresses, the pool contains progressively stronger opponents.

Each game generates 64 transitions `(s_t, a_t, r_t, s_{t+1}, done, legal_mask_{t+1})` stored in a 100,000-capacity replay buffer. The network trains from 128-sample minibatches, 4 gradient steps per game, with D4 symmetry augmentation (8-fold) applied at sample time.

### 2.3 Performance (P)

Three quantitative metrics are tracked every 100 games:

| Metric | Definition | Why it matters |
|---|---|---|
| Win rate vs Random (W_r) | Fraction of 200 games won against `RandomAgent` | The baseline floor — any learning should exceed 50% |
| Win rate vs Heuristic (W_h) | Fraction of 100 games won against `HeuristicAgent` | Non-learning tactical ceiling — the agent should surpass it |
| Elo rating | Updated against fixed anchors (Random=800, Heuristic=900) after each eval | Smooth, monotone skill estimate comparable across all checkpoints |

Additionally, a fixed benchmark (32 games against Alpha-Beta depth-4 search) runs every 100 games to measure strength against principled tree search.

**Formal hypotheses:**
- **H1:** W_r trends upward and ends ≥ 0.75
- **H2:** W_h(final) ≥ 0.90
- **H3:** Elo gain ≥ 200 points over the full run
- **H4:** Agent wins >50% of benchmark games against alpha-beta depth-4 by end of training

---

## 3. Algorithm Design and Implementation

### 3.1 Algorithm: Deep Q-Network (DQN)

DQN was selected from a survey of six candidates (tabular Q-learning, REINFORCE, PPO, AlphaZero-style, neuroevolution, supervised-from-self-play). The choice rests on three arguments:
1. C_lines is a discrete-action, perfect-information MDP — the textbook DQN domain.
2. The win-rate curve is the natural "improves with experience" evidence the brief requires.
3. The target-network state dict is simultaneously the snapshot artefact for the difficulty ladder.

### 3.2 Network Architecture

A residual convolutional Q-network (ResNet-v1):

```
Input  : (B, 10, 8, 8)  — 10-channel board encoding

Conv stem:   Conv2d(10→64, 3×3, pad=1) → BatchNorm → ReLU
             × 4 residual blocks [Conv→BN→ReLU→Conv→BN + skip → ReLU]
Q-head:      Conv2d(64→1, 1×1) → Flatten → Linear(64→64)

Output : (B, 64)  — Q-value per cell
```

Illegal cells are set to −10⁹ before argmax, guaranteeing legal play at all times.

### 3.3 Key Training Corrections

Three corrections over vanilla DQN proved critical:

**1. Negamax TD target (decisive change):** In a two-player game, the next state `s'` is encoded from the *opponent's* perspective. Vanilla DQN adds the opponent's Q-value: `target = r + γ·max_a Q(s',a)` — this teaches the agent that "a position great for my opponent is great for me." The correct formula negates it:
```
target = r − γ · Q_target(s', argmax_a Q_online(s', a))
```

**2. Double DQN:** Online network selects the action; target network evaluates it. Eliminates value overestimation and prevents late-training collapse.

**3. Huber loss:** Smooth-L1 loss replaces MSE, making training robust to large TD error spikes.

Additionally: plateau-based LR decay (halves when Elo plateaus for 10 intervals), epsilon decay from 1.0 to 0.05 over 7,000 games, and performance-based snapshot labelling (difficulty band from measured win rate, not generation count).

---

## 4. Results

**Training command:** `python -m src.training.train --games 10000 --size 8 --mode points_full --benchmark alphabeta_d4`
**Run ID:** `run_pts_002` | **Duration:** ~29 hours (CPU)

### 4.1 Snapshot Performance

| Snapshot | Games | Difficulty | WR vs Random | WR vs Heuristic | Elo |
|----------|-------|-----------|:---:|:---:|:---:|
| gen_001 | 1,000 | Easy | 61% | 48% | 803 |
| gen_003 | 3,000 | Hard | 89.5% | 86% | 894 |
| gen_005 | 5,000 | Hard | 70% | 85% | 1,006 |
| gen_007 | 7,000 | **Master** | **100%** | **100%** | 1,092 |
| gen_010 | 10,000 | **Master** | **100%** | **100%** | 1,190 |

### 4.2 Elo — Primary Evidence of Improvement

The Elo rating rose monotonically from **796 at game 0** to **1,190 at game 9,999** — an increase of 394 points, without a single reversal across all 100 evaluation checkpoints. This is the primary "performance improves with experience" graph.

Key milestones: crossed the heuristic anchor (900) at game 3,100 · crossed 1,000 at game 5,000 · reached 1,092 at game 7,000 (both win rates at 100%) · ended at 1,190.

### 4.3 Benchmark vs Alpha-Beta Depth-4

| Phase | Win Rate |
|---|---|
| Game 0 | 18.8% |
| Game 2,300 | **50%** — first crossover |
| Game 5,700 | **100%** — first perfect check |
| Games 6,400–6,700 | **100%, 100%, 100%, 100%** |
| End-of-run majority | 90–100% |

The agent beat a 4-ply tree-search engine in a majority of checks from game 2,300 onward — something that had never occurred in any prior training run.

### 4.4 Hypothesis Results

| | Threshold | Result | Met? |
|---|---|---|:---:|
| H1 WR_random ≥ 0.75 | 0.75 | **1.00** | ✅ |
| H2 WR_heuristic ≥ 0.90 | 0.90 | **1.00** | ✅ |
| H3 Elo gain ≥ 200 | +200 | **+394** | ✅ |
| H4 Beat α-β d4 >50% | >50% of checks | **Yes, from game 2,300** | ✅ |

---

## 5. Critical Analysis and Reflection

**Was the problem well-posed?** Yes, by Mitchell's three criteria. The task is fully specified (pure functions, no hidden state), the experience is reproducible (fixed seeds give identical curves), and the performance is single-valued and directly answers "did it improve?"

**What drove the improvement?** Three previous training runs failed completely — the final model was the worst in each run. The failure was a sign error in the TD target: adding `+γ·max_q` instead of `−γ·max_q` for the opponent's next state. Fixing this single character was the decisive change. It confirmed that the infrastructure (replay buffer, curriculum, snapshots, Elo, benchmark) was correct all along — only the learning rule was wrong.

**Honest limitations:**

1. *Residual variance.* At game 8,300 the agent lost all 32 benchmark games, surrounded by 100% checks. The policy has at least one exploitable weakness.
2. *Single seed.* The result is one run. Confirming reproducibility across three seeds would strengthen the claim.
3. *CPU constraint.* 10,000 games at ~500 games/hour is a practical ceiling. GPU training at 100,000+ games would likely close the residual variance.

**Reflection:** The project demonstrates that problem framing matters as much as algorithm choice. Three implementations of the same algorithm failed; one implementation with a corrected theoretical foundation succeeded completely. The key insight is that two-player self-play requires negating the bootstrap term — a fact specific to zero-sum alternating-turn games and not obvious from single-player DQN literature.

---

## 6. References

- Mitchell, T. M. (1997). *Machine Learning*. McGraw-Hill, Ch. 1. — TEP framework.
- Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533. — DQN.
- Van Hasselt, H., Guez, A., Silver, D. (2016). Deep reinforcement learning with Double Q-learning. *AAAI*, 30(1). — Double DQN.
- Sutton, R. S., Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. — replay buffer, TD learning.
- Silver, D. et al. (2016). Mastering Go with deep neural networks and tree search. *Nature*, 529, 484–489. — self-play residual network.
