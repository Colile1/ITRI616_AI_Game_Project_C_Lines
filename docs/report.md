# ITRI 616 Mini-Project Report — C_lines AI Learning Agent

**Student:** Colile Sibanda
**Game:** C_lines (original variant — open-placement four-in-a-row on a flat NxN board, N in {8..12}, two modes, custom no-draw tie-break)
**Algorithm:** Deep Q-Network (DQN) with self-play, snapshot pool, and per-step delta-score reward shaping
**Module:** ITRI 616 — Artificial Intelligence 1
**Status:** **Pre-training draft.** All numbers marked `[TBD]` are placeholders to be filled in after the full training run completes.

---

## 1. Introduction and Motivation

The ITRI 616 mini-project requires a well-posed learning problem for an intelligent agent on a Southern-African-themed, not-yet-digitised game, and asks for evidence that performance improves with experience. C_lines is an original variant designed for this project: players place a single piece per turn anywhere on an N×N grid (N selectable from 8 to 12) and score by forming uninterrupted lines of length 3 to 8 in any of the four directions. Two modes are supported — First-to-Four (the first 4-line wins) and Points-Until-Full (highest cumulative score wins, with a graded length schedule and a custom three-round mutual-removal tie-break that minimises draws).

The agent must learn from self-play alone — no published expert games or solver exist for C_lines — which makes the project a clean fit for reinforcement learning. Section 3 motivates the choice of DQN over the alternatives surveyed in `deliverables/in_md_format/02_ml_methods_research.md`.

---

## 2. Task Definition (TEP Framework)

The full formal definitions live in `tep_definitions.md`. The summary:

**Task (T):** Sequential decision-making (a finite-horizon MDP) over a perfect-information, deterministic, two-player, zero-sum game. The agent learns a policy π : S → A that maximises expected return, where S is the set of reachable board configurations and A is the flat set of N² cell indices.

**Experience (E):** Self-play simulated episodes. The first `WARMUP_GAMES` games are against a `RandomAgent`; the remainder mix games against the current agent and uniformly-sampled snapshots from a capped past-versions pool. Transitions `(s, a, r, s', done, legal_mask')` are stored in a fixed-capacity replay buffer.

**Performance (P):** Primary — win rate W(k) against `RandomAgent`, evaluated every `EVAL_INTERVAL` games over 200 fixed-seed games with the agent playing both sides equally. Secondary — win rate vs the heuristic baseline, intra-pool Elo, mean reward per episode, mean episode length.

---

## 3. Learning Algorithm

DQN was selected after surveying tabular Q-learning, REINFORCE / PPO, AlphaZero-style MCTS+NN, neuroevolution, supervised learning from self-play, and TD-Gammon-style value networks. The motivation chains four points (full discussion in `02_ml_methods_research.md`):

1. **Fit** — C_lines is a discrete-action, perfect-information, two-player MDP; DQN is the textbook algorithm for exactly this setting.
2. **Improvement-with-experience clarity** — the win-rate-vs-random curve is the natural and visually direct demonstration of learning, which is exactly what the brief grades.
3. **Snapshot economy** — DQN's target-network state dict is the natural snapshot unit, doubling as both a versioning artefact and a self-play opponent for diversity.
4. **Course constraint** — DQN is reinforcement learning with a simple neural network at its core, sitting cleanly inside the brief's allowed algorithm families.

### Architecture

```
Input  : (B, 6, N, N)    # 6 channels described in §2 of plan.md
Conv2d( 6 ->  32, 3×3, pad=1) -> ReLU
Conv2d(32 ->  64, 3×3, pad=1) -> ReLU
Conv2d(64 ->  64, 3×3, pad=1) -> ReLU
Flatten -> Linear(64·N·N -> 256) -> ReLU -> Linear(256 -> N·N)
```

Same architecture for all five board-size agent families, parameterised by N.

### Reward Function

Mode 1: per-step `+STEP_REWARD_SCALE = +0.05` for completing your own line of length ≥ 4; 0 otherwise. Terminal `+1` win / `-1` loss / 0 draw.

Mode 2: per-step `(Δown_score − Δopp_score) × STEP_REWARD_SCALE` — a potential-based delta-score shaping that converts the sparse end-of-game reward into a dense per-move signal that sums to the terminal score difference. Terminal `+1` / `-1`.

### Training Curriculum

* **Games 0 to WARMUP_GAMES (= 1000)** — DQN vs `RandomAgent`. Bootstraps the network with non-degenerate experience.
* **Games WARMUP_GAMES to TRAINING_GAMES (= 10_000)** — with probability `SELF_PLAY_MIX_PROB = 0.5`, opponent is a uniformly-sampled snapshot from the snapshot pool (capped at `MAX_POOL_SIZE = 20`); otherwise the current agent plays itself.
* Each episode randomises which colour the DQN plays to mitigate first-mover bias on the eval metric.
* Every `SNAPSHOT_INTERVAL = 1000` games, the agent is frozen and added to the pool plus the registry.

---

## 4. Experimental Results

_(All numbers below are placeholders pending the full training run.)_

### Headline numbers — `[TBD after training]`

| Board size | Final W vs random | Final W vs heuristic | Final Elo (in-pool) | Snapshots registered |
|------------|-------------------|----------------------|---------------------|----------------------|
| 8×8        | `[TBD]`           | `[TBD]`              | `[TBD]`             | `[TBD]`              |
| 9×9        | `[TBD]`           | `[TBD]`              | `[TBD]`             | `[TBD]`              |
| 10×10      | `[TBD]`           | `[TBD]`              | `[TBD]`             | `[TBD]`              |
| 11×11      | `[TBD]`           | `[TBD]`              | `[TBD]`             | `[TBD]`              |
| 12×12      | `[TBD]`           | `[TBD]`              | `[TBD]`             | `[TBD]`              |

### Figures

The five required figures are produced by `src/evaluation/plots.py` and saved to `results/figures/`:

1. `win_rate.png` — rolling-200 win rate vs random, with the 50% baseline shown. Hypothesis H1 (improvement-with-experience) is supported if this curve trends upward and ends well above 0.5.
2. `reward_curve.png` — cumulative reward per episode.
3. `episode_length.png` — transitions per episode over training.
4. `epsilon_decay.png` — linear ε schedule from 1.0 to 0.05 over the first 5000 games.
5. `loss_curve.png` — rolling-mean MSE TD loss over gradient steps.

`[TBD: insert figures and one-paragraph discussion of each after training.]`

### Hypothesis checks

| Hypothesis | Threshold | Met? |
|-----------|-----------|------|
| H1 — Win-rate vs random improves over training and ends ≥ 0.75 | `[TBD]` | `[TBD]` |
| H2 — Final win-rate vs heuristic ≥ 0.55 | `[TBD]` | `[TBD]` |
| H3 — Latest snapshot Elo is ≥ 200 above the early snapshot Elo | `[TBD]` | `[TBD]` |

---

## 5. Critical Analysis

### What worked — `[TBD post-training]`

To be answered with reference to the actual numbers: which design choices materially helped (delta-score shaping, action masking, snapshot pool, warm-up vs random), and which were neutral or harmful.

### Limitations

Three concrete limitations:

1. **Single algorithm only.** This project commits to DQN and does not run a head-to-head comparison against PPO or AlphaZero-style on the same game. The comparative argument in `02_ml_methods_research.md` is structural rather than empirical. A follow-on study could implement two algorithms and report directly comparable curves.
2. **CPU training budget caps the agent's reachable strength.** At `TRAINING_GAMES = 10_000` per board size, the agent learns enough to outpace random and heuristic baselines but is plausibly far from the game's optimal play, especially on the 12×12 board where the action space is largest. A GPU run with `TRAINING_GAMES = 100_000` would likely produce a materially stronger agent.
3. **No formal first-mover-advantage measurement.** The evaluation randomises which side the agent plays, which absorbs first-mover advantage into the headline number, but does not quantify it. An ablation reporting win-rate split by which colour the agent played would surface whether the agent has learned the same policy from both sides.

### Problem Well-Posedness

The problem is well-posed by Mitchell's three criteria. The **task** is exactly specified (action space, transition function, reward function, terminal condition — all defined as pure functions in `engine/rules.py`). The **experience** is fully reproducible (fixed-seed runs produce identical learning curves to within floating-point noise). The **performance** is single-valued and operationally simple (win-rate against a fixed baseline opponent set, evaluated on a fixed seed list, with the agent playing both colours).

The one area where well-posedness could be sharpened: in Mode 2, the per-step delta-score reward is potential-based, but is not the unique potential function that could be used. A version that included potential terms for open-3 threats might converge faster. This is an implementation refinement, not a flaw in the problem framing.

---

## 6. Conclusion — `[TBD post-training]`

The headline question to answer after training: **did performance improve with experience?** With the win-rate-vs-random curve in hand, the answer is a one-sentence yes-or-no, followed by the exact magnitude (e.g. "yes — from `[init W]` at game 0 to `[final W]` at game 10 000, with the curve crossing the 50% baseline at game `[crossover]`"). The Critical Analysis section, the figures, and the per-snapshot metadata in `models/registry.json` form the supporting evidence.

---

## 7. References

* Mitchell, T. M. (1997). *Machine Learning*. McGraw-Hill — TEP framework, Chapter 1.
* Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
* Sutton, R. S., Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
* Tesauro, G. (1995). Temporal difference learning and TD-Gammon. *Communications of the ACM*, 38(3), 58–68.
* Silver, D., Hubert, T., Schrittwieser, J., et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. *Science*, 362(6419), 1140–1144.
* Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
* Stanley, K. O., Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies. *Evolutionary Computation*, 10(2), 99–127.
