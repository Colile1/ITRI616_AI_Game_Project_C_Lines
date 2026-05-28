# Machine Learning Methods for C_lines — Research Report

**Author:** Colile Sibanda
**Module:** ITRI 616
**Date:** 2026-05-28
**Status:** Recommendation finalised — Deep Q-Network with self-play

---

## 1. Purpose and scope

This report surveys candidate machine-learning approaches for the C_lines game agent, evaluates each against the project's task structure and constraints, and recommends a single algorithm with explicit motivation. The intended outcome is to lock in an algorithmic choice before the implementation phase begins, so that the network architecture, environment interface, and training loop can be designed around a single coherent approach rather than retrofitted across competing paradigms.

The ITRI 616 brief restricts the choice to one of three families — supervised learning, a simple neural network, or reinforcement learning — so each candidate below is positioned relative to those families.

## 2. The learning problem at a glance

C_lines is a two-player, perfect-information, zero-sum, sequential, deterministic game with a finite but large state space. Players alternate placing a single piece on any empty cell on an N×N grid (N in 8..12). The terminal condition is either the first 4-in-a-row (Mode 1) or a full board (Mode 2), with a custom three-round piece-removal tie-break to enforce no-draw outcomes whenever possible. Scoring is non-linear in line length (3-in-row = 0.25, 4 = 1, 5 = 2, 6 = 3, 7 = 4, 8 = 5) and counts each maximal segment once per direction.

The branching factor starts at N×N (up to 144 on the 12×12 board) and shrinks linearly with move count. Game length is bounded above by N×N. There is no published expert-play corpus, so the agent cannot be bootstrapped from human games — experience must come from self-play, simulated episodes, or hand-coded opponents.

Rough state-space estimate: each of N×N cells is one of {empty, P1, P2}, so an upper bound is 3^(N×N). For N=12 that is 3^144 ≈ 10^68. Even with symmetry quotienting, the reachable state space is vastly too large for any tabular method.

The problem category is **sequential decision-making (a Markov Decision Process)** with delayed reward and a discrete, masked action space.

## 3. Candidate algorithms

For each candidate, the same five-axis assessment applies: theoretical fit to the task, sample efficiency, implementation complexity, hardware feasibility on a student laptop, and alignment with the ITRI 616 grading criteria (especially "performance improves with experience").

### 3.1 Tabular Q-learning

Tabular Q-learning maintains a table Q(s, a) and updates entries with the Bellman update Q(s, a) ← Q(s, a) + α[r + γ max_a' Q(s', a') − Q(s, a)]. It is the textbook reinforcement-learning method and the simplest to implement.

**Verdict — rejected.** With 3^144 states for the largest board, even a hashed sparse table would not see the same state twice during any feasible training run, so the table cannot generalise. Tabular methods only work when the agent revisits states often enough for the Q-values to converge, which is impossible here. Tabular Q-learning is a useful pedagogical baseline but not a feasible solution for C_lines.

### 3.2 Deep Q-Network (DQN)

DQN replaces the Q-table with a neural network Q(s, a; θ) and adds three stabilising tricks: (i) an experience replay buffer that samples i.i.d. minibatches to break correlation between consecutive transitions, (ii) a periodically-synced target network whose weights θ⁻ are used in the Bellman target to prevent the moving-target problem, and (iii) an epsilon-greedy exploration policy that anneals from near-random to near-greedy over training. Action masking is added by setting Q-values of illegal actions to −∞ before the argmax — essential for C_lines because most actions are illegal at any given state.

DQN's natural fit for board games comes from three properties: (1) the discrete, finite action space maps cleanly to a fixed-size output layer; (2) convolutional layers exploit translational regularities in the board (a 3-in-row threat at the top-left looks the same as one at the bottom-right); (3) the off-policy nature of Q-learning lets the agent learn from snapshots of itself, which is exactly what the project's versioning system produces anyway.

**Verdict — strong candidate.** Implementation complexity is moderate — every component (network, replay, target, epsilon, masking) is a textbook block, with no exotic dependencies beyond PyTorch. CPU training of a 12x12 board within 4–8 hours is realistic at the proposed hyperparameters. The Mnih et al. (2015) Nature paper provides a citable, well-understood reference.

### 3.3 Policy Gradient methods (REINFORCE, A2C, PPO)

Policy-gradient methods directly parameterise the policy π(a|s; θ) and update θ by gradient ascent on expected return. REINFORCE is the textbook entry point; A2C adds a learned value baseline to reduce variance; PPO clips the policy update to keep it close to the old policy and is the modern workhorse in continuous-control and games (used in OpenAI Five and many AlphaStar derivatives).

For C_lines, a policy-gradient approach would output a probability over the N×N action grid, masked to legal moves. The main appeal is theoretical: policy gradients can express any policy, including stochastic ones, and they handle large action spaces more gracefully than Q-learning when the optimal policy is highly non-greedy.

**Verdict — viable but heavier than necessary.** REINFORCE alone is too high-variance for episodes that can be 144 moves long with reward only at the end; A2C and PPO need an additional value head and an advantage estimator, doubling the network and the training-loop complexity. The Mode-2 reward (cumulative score) is already dense once delta-shaped, so the variance problem that policy gradients solve well is partly already addressed by the reward design. PPO would be the right choice if the problem were continuous-control, partially-observable, or had a vastly larger action space; for a small discrete action space with full observability, DQN matches or beats PPO at lower implementation cost.

### 3.4 AlphaZero-style (MCTS + policy/value network)

AlphaZero combines Monte Carlo Tree Search with a neural network that outputs both a policy prior and a state-value estimate. The network is trained on self-play games where the search-improved policy is the target for the policy head and the eventual game outcome is the target for the value head. AlphaZero superseded handcrafted-evaluation game engines in Go, chess, and shogi and is the state-of-the-art recipe for two-player perfect-information board games.

**Verdict — most powerful, but disproportionate to the project.** AlphaZero on C_lines would almost certainly produce the strongest agent. The cost is implementation complexity (a correct MCTS with PUCT, virtual loss, dirichlet exploration noise, and proper handling of legal-move masks at the root and in expansion), debugging time, and the need to interleave search and training in a way that is not amenable to the simple `train --games N` interface assumed by the skeleton. For a single-semester ITRI 616 mini-project where the grading emphasis is on TEP rigour, learning-curve evidence, and critical analysis rather than absolute playing strength, AlphaZero is over-engineered. It is the right choice for a follow-on Masters project.

### 3.5 Neuroevolution (NEAT, evolution strategies)

Neuroevolution evolves a population of networks by selection, crossover, and mutation, scoring each by its fitness in self-play. NEAT (NeuroEvolution of Augmenting Topologies, Stanley & Miikkulainen 2002) additionally evolves the network topology itself, growing connections and nodes as needed.

**Verdict — elegant but slow.** Neuroevolution is appealing pedagogically — it makes "improvement with experience" mean "each generation is fitter than the last" — but each fitness evaluation requires playing many games, and a population of 50 networks playing 100 games each per generation is 5000 games per generation, far slower than DQN's gradient-based updates. It also produces a less natural "snapshot pool" for the versioning system (do you save the generation champion? the whole population?). For C_lines, gradient-based methods will likely converge faster and with cleaner snapshots.

### 3.6 Supervised learning from self-play games

A two-step approach: first run self-play between random or weak heuristic agents to generate a dataset of (state, best-action) or (state, outcome) pairs; then train a neural network in the standard supervised manner to predict the label. This is closer in spirit to imitation learning than reinforcement learning.

**Verdict — rejected as the primary algorithm.** The "labels" produced by random or heuristic self-play are not actually optimal moves — they are the moves that random or heuristic agents chose — so a network trained on them will at best match the labelling policy, not exceed it. The "improvement with experience" criterion is also harder to demonstrate: a supervised network plateaus at the labelling policy's strength, while a reinforcement-learning agent's win rate visibly climbs over training. Supervised learning is, however, a useful **pre-training** step that could warm-start the DQN with a heuristic policy.

### 3.7 Simple neural network with TD-Gammon style training

TD-Gammon (Tesauro, 1992-95) trained a single neural network as a state-value evaluator using temporal-difference (TD(λ)) learning, then chose moves by one-ply look-ahead over the network's evaluations. It is historically important — it reached world-class backgammon by self-play alone — and conceptually simpler than DQN because it has a single value head and no replay buffer.

**Verdict — close runner-up.** A TD-style value network with one-ply (or shallow-MCTS) look-ahead would be a clean fit for C_lines. The main reasons it loses to DQN are: (i) action masking is more awkward on a value network (you have to enumerate next states for each legal action and evaluate each one, which costs N²-fold inference per move on the larger boards) versus DQN's single network call returning all Q-values; (ii) replay-buffer-based DQN is more sample-efficient than online TD on small data; (iii) the DQN literature is more standardised and citable for a mini-project report.

## 4. Comparative summary

The table below is the head-to-head, scored 1 (worst) to 5 (best) on five axes that matter to this project.

| Algorithm | Theoretical fit | Sample efficiency | Implementation complexity (lower is better → higher score) | CPU feasibility | TEP & report fit | Total |
|-----------|----------------|-------------------|----|----|-----|-------|
| Tabular Q-learning | 1 | 1 | 5 | 5 | 2 | 14 |
| **DQN with self-play** | **5** | **4** | **4** | **4** | **5** | **22** |
| PPO / A2C | 4 | 3 | 2 | 3 | 4 | 16 |
| AlphaZero-style | 5 | 5 | 1 | 2 | 4 | 17 |
| Neuroevolution | 3 | 1 | 3 | 2 | 3 | 12 |
| Supervised from self-play | 2 | 4 | 5 | 5 | 2 | 18 |
| Simple NN + TD-Gammon style | 4 | 3 | 4 | 4 | 4 | 19 |

Scoring rationale lives in the discussion in section 3; the column "Implementation complexity" is inverted so that easier-to-build counts higher (a project deliverable, not a research deliverable).

## 5. Recommendation

**Deep Q-Network with self-play and a snapshot pool.**

The motivation chains four reasons:

1. **Fit.** C_lines is a discrete-action, perfect-information, two-player MDP with a small enough action space to enumerate as network outputs and a board structure that convolutional layers exploit. DQN is the textbook algorithm for exactly this setting.

2. **Improvement-with-experience clarity.** The win-rate-against-random curve over training is the canonical "performance improves with experience" demonstration. The course explicitly grades on this. DQN produces this curve naturally; tabular methods cannot, and AlphaZero's stronger curve costs disproportionately more engineering.

3. **Snapshot economy.** The project requires the agent to be frozen at intervals into immutable "levels" with metadata. DQN's natural unit — the target-network state dict — is precisely the right thing to checkpoint. The replay buffer can also be self-played against the snapshots, so the pool is doing double duty as opponents and as a diversity injector for the training data.

4. **Course constraint.** The brief explicitly allows "supervised learning, a simple neural network, or reinforcement learning". DQN is reinforcement learning, with a simple neural network at its core. It satisfies the constraint without forcing a more exotic algorithm (AlphaZero) past the brief's framing.

The runner-up is the TD-Gammon-style value network; if the DQN approach fails to converge after reasonable tuning, that is the documented fallback (see `04_implementation_plan.md`, risk register).

## 6. What this means for the build

The choice of DQN locks in specific design decisions that the rest of the documentation depends on:

* **Network output size = ACTION_SPACE_SIZE = N × N** (one Q-value per cell, for each board-size agent family).
* **State encoding** is multi-channel `(C, N, N)`; channels are described in `04_implementation_plan.md`.
* **Legal-move masking** sets illegal Q-values to −∞ before the argmax in both training and inference.
* **Replay buffer** stores tuples (state, action_idx, reward, next_state, done, legal_mask_next).
* **Target network** is synced every `TARGET_SYNC_STEPS` gradient updates.
* **Self-play curriculum** is warm-up vs `RandomAgent` for `WARMUP_GAMES` episodes, then a mixed schedule of current-agent vs snapshot pool.
* **Snapshot cadence** every `SNAPSHOT_INTERVAL` games; weights frozen, metadata recorded, registered in `models/registry.json`.

## 7. References (full report citations)

* Mitchell, T. M. (1997). *Machine Learning*. McGraw-Hill — TEP framework, Chapter 1.
* Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533 — canonical DQN reference.
* Tesauro, G. (1995). Temporal difference learning and TD-Gammon. *Communications of the ACM*, 38(3), 58–68 — TD-style value-network self-play.
* Silver, D., Hubert, T., Schrittwieser, J., et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. *Science*, 362(6419), 1140–1144 — AlphaZero reference.
* Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
* Stanley, K. O., Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies. *Evolutionary Computation*, 10(2), 99–127 — NEAT reference.
* Sutton, R. S., Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press — foundational MDP / Q-learning treatment.

---

*End of ML methods research report.*
