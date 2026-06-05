# C_lines — Project Report (FINAL)

### A Self-Play Reinforcement-Learning Agent for an Original Line-Scoring Board Game

**Author:** *----* *S----* · North-West University · ITRI 616 — Artificial Intelligence 1
**Date:** 2026-06-02
**Configuration:** C_lines, 8×8 board, points-until-full mode
**Training run reported:** `run_pts_002` — 10,000 self-play games

> This report follows the documentation structure specified by the lecturer: (1) formal TEP definitions, (2) learning-algorithm choices and how they were implemented, (3) experimental evaluation results, (4) critical analysis, and (5) reflection on the project as a whole. A separate document, `18_Code_Documentation.md`, covers installation, running, and reproducing the experiments.

---

## 0. The game (context for everything below)

C_lines is an original two-player board game devised for this project — not a digitisation of an existing commercial game. It is played on an **8×8 grid**. Players alternate turns and each turn places one stone on **any empty cell** (open placement: no gravity, no column rule). Play continues until the board is full — 64 placements. A player scores for every straight run of their own stones (horizontal, vertical, or diagonal) of length 3 to 8, with longer runs worth disproportionately more (length 3 = 0.25, 4 = 1.0, 5 = 2.0, 6 = 3.0, 7 = 4.0, 8 = 5.0). When the board fills, the higher total score wins; a three-round tie-break removes draws. Because placement is unconstrained, the empty board offers 64 possible moves, and the player must balance **building** their own lines against **blocking** the opponent's over a 64-move horizon. This makes C_lines a non-trivial sequential-planning problem and a sound testbed for a learning agent.

---

## 1. Formal TEP definitions

The learning problem is specified with Mitchell's well-posed-learning-problem framework: a **Task**, an **Experience**, and a **Performance** measure.

### 1.1 Task (T)

> **Given a board position, choose the placement that maximises the agent's final score margin** (its own line-score minus the opponent's when the board fills).

Maximising the final score margin is exactly equivalent to winning the points-until-full game.

**Problem category.** Sequential **decision-making / control**. The agent emits a sequence of placements in a Markov decision process; each decision changes the state and the set of future decisions. It is **not** classification (no fixed label per board) and **not** prediction (the agent must *act*, and its actions determine the outcome). Identifying the category correctly matters because it dictates the algorithm family: control problems with delayed reward call for reinforcement learning, not supervised learning.

### 1.2 Experience (E)

> **Self-generated simulated episodes** — the agent learns from games it plays, primarily against past versions of itself.

Experience is produced in three layers:

1. **Warm-up (games 0–2,000):** the learner plays a mix of a random opponent and a fixed tactical heuristic, seeding the replay memory with varied, legal positions.
2. **Self-play with a snapshot pool (games 2,000–10,000):** the learner plays frozen copies of its earlier selves, biased toward recent (stronger) snapshots. As the learner improves, so do its opponents — an automatically generated curriculum.
3. **Replay and augmentation:** every transition is stored and reused for several gradient updates, and the board's 8-fold symmetry multiplies the effective data.

No human game records are required; the agent is responsible for generating its own training distribution. This is the "demonstrate that performance improves with experience" requirement made concrete — the *experience* is the games, and the report shows performance rising as that experience accumulates.

### 1.3 Performance (P)

Performance is measured quantitatively, logged every 100 games, with four complementary metrics:

| Metric | What it captures | Justification |
|---|---|---|
| **Win rate vs Random** | Basic competence | A learning agent must dominate random play; a floor check. |
| **Win rate vs Heuristic** | Tactical competence | The heuristic blocks and builds lines; beating it needs real strategy. |
| **Elo rating** | Relative skill, one scalar | A smooth, comparable summary across the whole opponent set — the headline improvement curve. |
| **Win rate & score margin vs Alpha-Beta depth-4** | Strength vs principled search | The hardest, independent test — an opponent the agent never trains against. |

Using four metrics means the improvement claim does not depend on any single number that might saturate or be noisy.

---

## 2. Learning-algorithm choices and implementation

### 2.1 Choice of algorithm and justification

The agent is a **Deep Q-Network (DQN)** — value-based reinforcement learning. The network approximates the action-value function Q(s, a): the expected future score-margin of playing cell *a* in position *s*. At play time the agent selects the legal cell with the highest Q-value; during training it explores with an ε-greedy policy, ε decaying linearly from 1.0 to 0.05 over the first 7,000 games.

DQN was chosen over the two other options in the brief — a supervised classifier or a plain feed-forward network — because the problem is **sequential control with delayed reward**: a placement's true value only emerges many moves later when the board fills. Q-learning's bootstrapped temporal-difference (TD) target is purpose-built for that credit-assignment problem, whereas supervised learning would need labelled "correct moves" that do not exist for an original game.

### 2.2 How it was implemented

**State representation.** Each position is a **10-channel 8×8 tensor**, always from the perspective of the player to move: own stones; opponent stones; empty cells; own open-3 threats; opponent open-3 threats; turn-progress; own open-4 threats; opponent open-4 threats; cells where the agent wins immediately; cells where the opponent wins immediately. The tactical channels hand the network the features that matter for blocking and line-completion, accelerating learning.

**Network architecture.** A **residual convolutional network**: a 3×3 convolutional stem, **four residual blocks** at 64 channels with batch normalisation, and a value head producing one Q-value per cell (64 outputs). Residual connections let the network train stably while representing spatial line patterns.

**Training mechanics — the choices that made it work:**

- **Two-player (negamax) TD target.** Because the next state belongs to the *opponent*, the bootstrap term is **subtracted**, not added: `target = r − γ · maxₐ Q(s′, a)`. This is the correct value-propagation rule for a two-player zero-sum game and is the single most important correctness detail in the implementation.
- **Double DQN.** The online network selects the next action and the target network evaluates it, reducing vanilla DQN's value-overestimation bias.
- **Huber (smooth-L1) loss** for robustness to occasional large TD errors.
- **Target network** synced every 500 gradient steps; **Adam** optimiser; **gradient clipping** at norm 10; replay buffer of capacity 100,000 reused for 4 gradient steps per game.
- **D4 symmetry augmentation:** rotations and reflections multiply effective data eightfold.
- **Plateau-based learning-rate decay:** the rate halves (1e-3 → … → 1.25e-4) whenever the win rate stalls.
- **Snapshotting:** every 1,000 games the model is frozen, evaluated, Elo-rated, and registered as a selectable difficulty level (`gen_001` … `gen_010`), and admitted to the self-play pool.

**Reward design.** Dense **delta-score shaping**: after each move the agent receives the change in (own score − opponent score), scaled, plus a terminal ±1 for the win/loss. Because the objective in points-until-full *is* to accumulate the larger score, this intermediate reward is perfectly aligned with the true objective — every step's signal points toward the same goal as the final outcome.

---

## 3. Experimental evaluation results

### 3.1 Setup

One training run, `run_pts_002`: 8×8 board, points-until-full, 10,000 self-play games. Evaluation every 100 games (200 games vs random, 100 vs heuristic); benchmark every 100 games against alpha-beta depth-4 over 32 games per check. Hardware: CPU only; throughput ≈ 450–620 games/hour.

### 3.2 Headline result — performance improves with experience

![Combined progress](../../../results/size_08/run_pts_002/figures/combined_progress.png)

*Figure 1 — Win rate vs Random (blue) and vs Heuristic (green), left axis; Elo (purple dashed), right axis. All three rise together from random-level to ceiling.*

| Checkpoint | Games | Elo | WR vs Random | WR vs Heuristic |
|---|---|---|---|---|
| Start | 0 | 796 | 53% | 46% |
| `gen_002` | 2,000 | 842 | 81% | 68% |
| `gen_004` | 4,000 | 954 | 91% | 90% |
| `gen_007` | 7,000 | 1,092 | 100% | 100% |
| `gen_010` (final) | 10,000 | **1,190** | **100%** | **100%** |

The Elo curve is the clearest single piece of evidence: it rises **monotonically by +394 points** with no late-training collapse.

![Elo curve](../../../results/size_08/run_pts_002/figures/elo_curve.png)

*Figure 2 — Elo rating vs training games: a smooth, monotone increase — the textbook signature of "improves with experience".*

### 3.3 The decisive benchmark — beating a 4-ply search

![Benchmark vs alpha-beta](../../../results/size_08/run_pts_002/figures/benchmark_vs_alphabeta.png)

*Figure 3 — Win rate vs Alpha-Beta depth-4 (32 games per check). The agent goes from winning ~19% (score margin −0.83) to ~100% (score margin +1.17).*

| Phase | WR vs Alpha-Beta d4 | Mean score margin |
|---|---|---|
| Game 0–300 | 16–25% | −0.93 to −0.72 |
| Game ≈9,000–9,900 | 97–100% | +1.13 to +1.45 |

Reversing both the win rate **and** the sign of the score margin against a principled adversarial search — one the agent never trains against — is far stronger evidence than merely beating random play.

### 3.4 Supporting diagnostics

- **Training loss** stays low and bounded throughout — no divergence — consistent with the Double-DQN + Huber-loss + target-network stabilisers.
- **Learning-rate schedule** stepped down at games ~5,400, ~7,400 and ~8,400 in response to win-rate plateaus, after which the agent consolidated at a 100% ceiling.
- **Best-model checkpoint** was first captured at game 6,400 (100% vs heuristic), and the final model matches it — confirming the end-of-run model is genuinely the strongest, not an evaluation fluke.

---

## 4. Critical analysis

**Was the problem well-posed?** Yes — and by design. The reward the agent optimises (delta score margin) *is* the win condition, so there is no gap between what it is trained to do and what it is graded on. The results bear this out: aligned objective, clean convergence.

**Did performance improve with experience?** Unambiguously, and on all four metrics simultaneously: Elo +394 (monotone), win rate vs random 53%→100%, vs heuristic 46%→100%, and vs a 4-ply search 19%→100% with the score margin flipping sign. Because the metrics rise *together*, the improvement reflects genuine learning rather than noise in any one measure.

**Assumptions made:**
- *Perfect information and determinism* — both players see the whole board and placement is deterministic, justifying a value-based MDP formulation.
- *Self-play sufficiency* — playing past selves yields a strong-enough curriculum to reach expert play. Beating an independent search supports this.
- *Representative opponents* — random, heuristic, and alpha-beta-d4 span weak→principled adversaries (bounded; see limitations).
- *Markov state* — the 10-channel encoding plus turn-progress is a sufficient statistic; since C_lines has no hidden state or history dependence, this holds.

**Limitations (stated honestly):**
- *One configuration* — results are for 8×8 points-until-full; generalisation to other settings is not claimed here.
- *Single seed* — the run was not repeated across random seeds, so run-to-run variance is not quantified; the smooth Elo curve is reassuring but not a substitute for a multi-seed band.
- *Benchmark ceiling* — "100% vs depth-4" means "stronger than a 4-ply search", not "optimal play".
- *Per-snapshot win-rate noise* — individual 100-game evaluations fluctuate (e.g. a dip to 70% vs random at game 5,000); the Elo curve smooths this, which is why it is the primary signal.
- *CPU-bound throughput* — a full run takes hours, limiting the project to one configuration in the available time.

**What the design got right:** a reward aligned with the objective, the correct two-player (negamax) value target, Double-DQN/Huber stabilisation, and Elo-based evaluation that exposes improvement as a clean curve. Together these turn a hard 64-ply control problem into a stable, monotone learning result.

---

## 5. Reflection on the project as a whole

Looking back over the project, three lessons stand out.

**The hardest part was correctness, not complexity.** The infrastructure — a residual network, a replay buffer, self-play, snapshots — is standard once written. What actually determined success was a single conceptual detail: in a two-player game the next position belongs to your opponent, so its value must be *subtracted* in the update. Getting that sign right is what separates an agent that drifts aimlessly from one that converges to expert play. The biggest thing I learned is that in reinforcement learning, a subtle error in the *objective* is far more damaging than any amount of missing engineering, and far harder to spot, because the code still runs and the loss still goes down — it just optimises the wrong thing.

**Measurement is part of the method, not an afterthought.** Early on I judged progress by win rate against fixed opponents, but those numbers are noisy game-to-game and saturate once the agent is strong, so they made it hard to tell real progress from luck. Adopting an **Elo rating** as the primary metric changed how I worked: it produced a single smooth curve I could actually trust to tell me whether a change helped. I now treat "how will I measure this?" as a design question to answer *before* training, not after.

**Engineering discipline paid off.** Keeping every hyperparameter in one config file, logging every game and benchmark check to CSV, and registering each snapshot with its metrics meant that when the agent finally worked, I could *prove* it with reproducible figures rather than anecdotes — and could regenerate every result in this report from the saved data with one command. That discipline is the difference between "I think it improved" and "here is the curve".

**If I had more time** I would repeat the run across several random seeds to put confidence bands on the curves, raise the benchmark to a deeper search, and extend the value network with a policy head and Monte-Carlo Tree Search at training time (an AlphaZero-style approach) to push beyond a 4-ply opponent. The current result already meets the project's goal — demonstrable improvement with experience — but those steps would strengthen the claim from "clearly improves" to "approaches the limits of the game".

**Overall**, the project achieved what it set out to do: it poses an original game as a well-defined learning problem and shows, with quantitative evidence on four independent metrics, that a self-play agent's performance improves with experience — from a blank slate to out-playing a principled search.

---

## Appendix — Mapping to the documentation requirements

| Required section | Where |
|---|---|
| Formal TEP definitions | §1 |
| Learning-algorithm choices & implementation | §2 |
| Experimental evaluation results | §3 |
| Critical analysis | §4 |
| Reflection on the project as a whole | §5 |
| Code documentation (install/run/experiments) | `18_Code_Documentation.md` |

**Data and figures behind this report (run_pts_002 only):**
`results/size_08/run_pts_002/training_log.csv`, `benchmark_log.csv`, `models/size_08/run_pts_002/registry.json`, and `results/size_08/run_pts_002/figures/{combined_progress,elo_curve,benchmark_vs_alphabeta}.png`.
