# Similar-Game AI Research — How Other Open-Placement Line Games Are Solved

**Author:** *----* *S----*
**Module:** ITRI 616
**Date:** 2026-05-28
**Status:** Research complete; informs `09_algorithm_upgrade_plan.md`

---

## 1. Why this research

The 8×8 DQN-only agent in `report.md` reaches 86% win rate against `RandomAgent`, but practical play against a thinking opponent is weak — it misses obvious blocks, plays passively on the wider boards, and shows little long-term planning. This matches what the public literature reports for value-only DQN on board games: it learns a competent reactive policy but lacks the look-ahead that line-forming games reward.

This report surveys the AI agents built for games that are structurally close to C_lines — Gomoku, Connect Four, Hex, Othello, Go — extracts the techniques that consistently produce strong play, and maps each technique onto what C_lines specifically needs. The output is an evidence base for the upgrade plan in `09_algorithm_upgrade_plan.md`.

## 2. The five reference games (closest to farthest)

### 2.1 Gomoku / Renju / Five-in-a-Row (closest analogue)

**Why it matters for C_lines.** Gomoku is open-placement on a square grid (typically 15×15 or 19×19) and the win condition is forming a line of length 5 in any of four directions. C_lines is the same template with a smaller board, variable line lengths, and a graded scoring schedule. Strategies, agent architectures, and engineering trade-offs map almost 1-to-1.

**Strong public agents.**

* **Yixin** (Renju Engine) — commercial; uses **threat-space search** (an alpha-beta variant that only explores "threat-creating" moves) layered with a learned position evaluator. Search depth 20+ moves in the threatening branch; orders of magnitude shallower elsewhere.
* **AlphaZero-Gomoku** (Junxiao Song's open-source PyTorch implementation, GitHub) — a textbook AlphaZero adaptation. Residual-block policy/value network + PUCT MCTS + self-play. Documented to beat strong heuristic baselines after a few hundred thousand self-play games on a single GPU.
* **Gomocup** tournament agents — annual competition; recent winners typically combine MCTS with a hand-engineered evaluation function or a small NN evaluator, plus threat-space pruning.

**Techniques that matter.**

* **Threat-space search** — restrict the tree to moves that create or block a forced threat (open-3, open-4); reduces branching factor from 200+ to ~10 in the critical branches.
* **MCTS / PUCT** — the dominant choice over the last decade; AlphaZero recipe is well-documented.
* **Symmetry augmentation** — 8-fold dihedral symmetry on a square board multiplies training data per game.
* **Domain-specific input channels** — open-3 threats, open-4 threats, closed-4 threats explicitly encoded.

### 2.2 Connect Four

**Why it matters for C_lines.** Connect Four is the closest "small-action-space line game" and is **solved** (Allis 1988; Tromp's enumeration). This means we have a perfect-information ground truth to compare against and a long history of engine engineering tricks.

**Strong public agents.**

* **Pascal Pons's Connect4 Solver** (open-source, C++) — alpha-beta with bitboards; ~100 million positions per second; solves 7×6 in milliseconds.
* **Connect4-AlphaZero** (multiple GitHub clones) — full AlphaZero recipe applied to Connect Four; reaches near-perfect play after ~50k self-play games on a GPU.
* **Connect4 DQN agents** — many published baselines. Reach 80–90% vs random but plateau there, exactly mirroring C_lines's current state.

**Techniques that matter for C_lines.**

* **Bitboards** — represent the position as two 64-bit integers (one per player); enables fast move enumeration and pattern matching. For 12×12 we need 144 bits, so two 128-bit integers (numpy `uint64[2]` or Python `int`).
* **Alpha-beta with iterative deepening + transposition tables** — classical, deterministic, no NN needed; sets a baseline that learning agents must clear.
* **Pre-computed pattern tables** — every length-4 window over the board enumerated once; per-move lookup is O(1) per window.

### 2.3 Hex

**Why it matters for C_lines.** Hex has a similar branching factor (up to 121 on 11×11 boards), no draws by design, and is a perfect-information two-player game. It is well-studied for MCTS performance.

**Strong public agents.**

* **MoHex** (University of Alberta) — MCTS with **RAVE** (Rapid Action Value Estimation), virtual connections (a domain-specific pruning), and a learned evaluator. Multiple international Hex-tournament wins.
* **DeepZenHex / AlphaZero-Hex** clones — AlphaZero recipe applied to Hex.

**Techniques that matter for C_lines.**

* **RAVE** — uses the average outcome of all simulations that played a given move (regardless of when) to seed early UCT estimates. Big sample-efficiency win when full MCTS is expensive.
* **Virtual-connections-style domain pruning** — for C_lines this maps to "ignore moves that neither create nor block a 3-threat in the early-game tree", a tractable analogue.

### 2.4 Othello / Reversi

**Why it matters for C_lines.** Othello is a flat board game with piece-placement (not gravity), like C_lines. Its long history of strong agents pre-AlphaZero gives a good window onto what value-function learning alone can achieve.

**Strong public agents.**

* **Logistello** (Michael Buro, 1990s) — alpha-beta + a **logistic-regression evaluator** trained on tens of millions of positions. Beat the human world champion 6-0 in 1997. Pre-deep-learning, but demonstrates how far a strong tactical search + a competent learned eval can go.
* **Edax** — modern alpha-beta engine with bitboards; super-human.
* **NTest** — neural-net evaluator + alpha-beta; super-human.

**Techniques that matter for C_lines.**

* **Hand-engineered features (mobility, parity, corner-control)** combined with a small learned evaluator outperform many neural-net-only agents at low compute. For C_lines: mobility = legal-move count, threat-count, longest-line-so-far.
* **Iterative deepening alpha-beta + good move-ordering** is competitive on small boards without any NN.

### 2.5 Go (farthest, but most influential)

**Why it matters for C_lines.** Go is the modern proving ground of game AI. The recipes invented for Go set the template for every modern board-game agent.

**Strong public agents.**

* **AlphaGo → AlphaGo Zero → AlphaZero → MuZero** (DeepMind) — the canonical reference.
* **KataGo** (open-source, David Wu) — AlphaZero-style with significant improvements: score-aware value head, auxiliary policy targets, intermediate value targets, batch-renorm. Strongest open-source Go engine.
* **Leela Zero** — distributed self-play reproduction of AlphaGo Zero.

**Techniques that matter for C_lines.**

* **Residual blocks** — train deeper networks (10-40 blocks) more stably than a plain CNN stack.
* **Policy + value head** instead of Q-values; MCTS uses the policy head as a search prior.
* **N-step / Monte Carlo returns** — credit assignment over longer horizons.
* **Auxiliary heads** — predicting ownership / final-score in addition to win/loss accelerates learning. For C_lines: predict final-score-difference and/or per-cell "owner at end of game".
* **Batch normalisation** inside residual blocks.

## 3. Techniques that recur across all five games

Treating the five game-agents as a population and ranking techniques by how universally they appear:

| Technique | Appears in | What it buys |
|-----------|-----------|--------------|
| **MCTS / PUCT** | Gomoku, Connect4, Hex, Go (modern) | Look-ahead at inference; biggest single strength multiplier over value-only methods |
| **Symmetry / data augmentation** | Gomoku, Hex, Go | 4-8x effective training data per game on a symmetric board |
| **Domain-specific input channels (threats)** | Gomoku, Connect4, Hex | Inductive bias; the network spends capacity on strategy, not pattern detection |
| **Residual blocks + BatchNorm** | Modern Gomoku, Hex, Go | Deeper, stabler networks; better feature reuse |
| **Policy + value heads (AlphaZero)** | Modern Gomoku, Hex, Go | Policy head guides MCTS; value head bootstraps before terminal reward |
| **Alpha-beta with bitboards (classical)** | Connect4, Othello | Very strong without any learning; baseline you must beat |
| **Prioritized experience replay** | DQN-based Connect4 agents | Sample-efficiency on transitions with high TD error |
| **Threat-space search (domain pruning)** | Gomoku | Massive search-tree reduction in the critical sub-tree |
| **Population-based self-play** | Go (AlphaStar/AlphaZero variants) | Diversity + curriculum |
| **N-step or Monte Carlo returns** | All modern systems | Faster credit assignment on long episodes |

The first three rows are the unambiguous wins — every strong agent in every game uses them, and they are independent of the underlying algorithm choice (DQN, AlphaZero, or alpha-beta).

## 4. What is currently missing in C_lines

Cross-referencing the existing build against the table above:

| Technique | C_lines today? | Gap |
|-----------|---------------|-----|
| MCTS / PUCT | ❌ | Pure greedy / epsilon-greedy on DQN Q-values. No look-ahead. |
| Symmetry augmentation | ❌ | Every game produces only the data it actually generated. |
| Domain-specific input channels | Partial | Channels 3/4 encode open-3 threats; open-4 and closed-4 not encoded. |
| Residual blocks | ❌ | Plain 3-layer CNN. Adequate but limited. |
| Policy + value heads | ❌ | Q-value head only. |
| Alpha-beta classical baseline | ❌ | No bitboard / minimax baseline to anchor against. |
| Prioritized replay | ❌ | Uniform sampling. |
| Threat-space search | ❌ | None. |
| Population-based self-play | Partial | Snapshot pool exists; no PBT-style hyperparameter diversity. |
| N-step returns | ❌ | 1-step TD targets only. |
| Symmetric evaluation (P1/P2 split) | ✓ | Already in place. |

**The five gaps that explain "still results in bad game play":**

1. **No search at inference.** Every published strong agent for line games does at least shallow search at decision time. DQN-only is reactive, not strategic.
2. **No symmetry augmentation.** The agent has effectively seen ~10 000 games of experience instead of ~80 000 (8-fold dihedral × 10 000).
3. **Plain CNN, no residual blocks.** The network is small relative to the complexity of multi-direction threat patterns.
4. **No open-4 / closed-4 / fork features in the input.** The most strategically pivotal patterns are not in the network's input vocabulary; it must learn them from scratch, which costs games.
5. **No classical-search baseline to anchor evaluation.** "Strong play" is not benchmarked against any thinking opponent — only against random and a 1-ply heuristic.

## 5. What good would look like for C_lines

Translating the cross-game patterns to a C_lines-specific recipe:

1. **MCTS at inference**, with PUCT, expanded by the existing DQN's Q-values as priors (a softmax over Q gives a policy proxy). 100-400 simulations per move is enough to be visibly stronger; 800-1600 starts approaching tournament strength.
2. **Symmetry augmentation** at sample time in the replay buffer: every transition is duplicated 8x with the dihedral group of the board, including the action index appropriately rotated/reflected.
3. **Residual-block network** (3-6 blocks of `Conv-BN-ReLU-Conv-BN + skip`) replaces the plain 3-conv stack. Same parameter budget, more depth.
4. **Threat-rich state encoding**: add channels for open-4 threats, closed-4 threats, immediate-win cells, immediate-loss cells (cells where the opponent would win next turn). These are cheap to compute (same scanner that scores lines).
5. **A classical search baseline** (alpha-beta with depth 4-6 and a hand-engineered eval) as a fixed benchmark that does not learn, separate from the random and heuristic baselines.
6. **Policy + value heads at the next step.** Optional but recommended as the natural follow-on — moves the project firmly into AlphaZero territory.
7. **Prioritized experience replay** for sample efficiency, especially when human games are added (Section 6 of `10_human_in_loop_training_plan.md`).

## 6. Specific public implementations worth studying or borrowing structure from

These are open-source code bases that map closely to C_lines and are good references for the rewrite:

| Project | Game | Stack | What to take from it |
|---------|------|-------|----------------------|
| `junxiaosong/AlphaZero_Gomoku` (GitHub) | Gomoku | PyTorch | Policy/value head, PUCT loop, dihedral augmentation. ~600 lines of clear code. |
| `connect4-alpha-zero` (multiple forks) | Connect Four | PyTorch / TensorFlow | Snapshot self-play loop, evaluator design |
| `KataGo` (lightvector/KataGo) | Go | C++ / Python | Residual block design, auxiliary heads, score-aware value head |
| `OpenSpiel` (DeepMind) | Many | Python / C++ | Game-agnostic MCTS implementation; useful as a reference for the PUCT formula |
| `cpp-connect4` (Pascal Pons) | Connect Four | C++ | Bitboard + alpha-beta + transposition tables; reference for a classical baseline |

Each is permissively licensed. The recommended pattern is to read for structure, not copy code wholesale — direct copying ties the project to architectural assumptions that may not fit C_lines's graded scoring.

## 7. What the literature also says about DQN-vs-AlphaZero on small games

Empirical findings across the surveyed papers and reproductions:

* On small board games (Connect Four, 8×8 Othello, 7×7 Go), AlphaZero-style agents typically reach **70-90% win rate against strong alpha-beta** after a similar number of self-play games as a tuned DQN reaches **30-50%** vs the same opponent. AlphaZero wins because MCTS at inference compensates for any residual value-function error.
* DQN's strength gap closes substantially if MCTS is bolted on at inference only, even without retraining: the same network, played through 200 PUCT simulations per move, plays measurably stronger than the greedy policy.
* Symmetry augmentation alone, with no other change, has been documented to cut training time to a given strength by 30-60% on Gomoku and Go.

This evidence underwrites the upgrade plan in `09_algorithm_upgrade_plan.md`: the highest-impact, lowest-risk single change is to bolt MCTS onto the existing DQN at inference. Symmetry augmentation is the highest-impact, lowest-risk single change to **training**. Both can ship before any architectural rewrite.

## 8. Recommended path for C_lines

Ordered from highest expected impact per hour of engineering, lowest risk:

1. **Symmetry augmentation in the replay buffer** — 1-2 hours of code, no architectural change, 30-60% faster convergence to the same strength.
2. **MCTS-at-inference using the existing DQN as a policy/value proxy** — 1-2 days of code, no retraining required, immediate strength gain. Apex of value-for-effort.
3. **Add open-4 / closed-4 / immediate-win / immediate-loss channels to the state encoder** — half a day; uses code already in the line scanner.
4. **Replace the plain conv stack with 3-5 residual blocks**, retrain — 1 day to code, retrain wall-clock.
5. **Add a fixed classical alpha-beta benchmark agent** (depth 4 with a hand-engineered eval) for benchmark logging — half a day; informs the human-in-loop plan.
6. **(Later)** Add a policy head alongside the Q head, train with MCTS-improved policy targets — this is the AlphaZero-light step. 2-3 days of code, retrain wall-clock.
7. **(Optional, ambitious)** Full AlphaZero with policy-targets-from-MCTS. Significant rewrite. Reserve for a follow-on project.

Stages 1-4 are the deliverable scope for `09_algorithm_upgrade_plan.md`. Stages 5 ties directly into `10_human_in_loop_training_plan.md`. Stages 6-7 are documented as future work in `11_unified_upgrade_plan.md`.

## 9. References

* Allis, L. V. (1988). *A Knowledge-Based Approach of Connect Four — Master's thesis*, Vrije Universiteit Amsterdam. — Connect Four is a first-player win.
* Buro, M. (1995). Logistello — *Statistical feature combination for the evaluation of game positions*. AAAI Workshop. — Othello evaluator.
* Wu, D. J. (2019). Accelerating Self-Play Learning in Go. *arXiv:1902.10565*. — KataGo improvements.
* Silver, D., Hubert, T., Schrittwieser, J., et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. *Science*, 362(6419), 1140-1144. — AlphaZero.
* Schrittwieser, J., Antonoglou, I., Hubert, T., et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model. *Nature*, 588, 604–609. — MuZero.
* Browne, C. B., Powley, E., Whitehouse, D., et al. (2012). A survey of Monte Carlo tree search methods. *IEEE Transactions on Computational Intelligence and AI in Games*, 4(1), 1-43. — Canonical MCTS survey.
* Gelly, S., Silver, D. (2011). Monte-Carlo tree search and rapid action value estimation in computer Go. *Artificial Intelligence*, 175(11), 1856-1875. — RAVE.
* Schaul, T., Quan, J., Antonoglou, I., Silver, D. (2016). Prioritized Experience Replay. *ICLR*. — PER.
* Hessel, M., Modayil, J., et al. (2018). Rainbow: Combining Improvements in Deep Reinforcement Learning. *AAAI*. — DQN improvements.
* Tromp, J. C-4 enumeration page (online). — Connect Four exact solution.
* Junxiao Song (2018). *AlphaZero-Gomoku* (open-source repo). — Reference implementation for Gomoku.
* Pons, P. *Connect 4 Solver* (online article series). — Reference for bitboard alpha-beta.

---

*End of similar-games research report.*
