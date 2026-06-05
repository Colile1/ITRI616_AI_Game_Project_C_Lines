# Presentation Script — ITRI 616 Mini-Project

**Student:** *----* *S----*
**Topic:** C_lines AI Learning Agent
**Target time:** 12–15 minutes
**Structure:** TEP → Algorithm → Results → Demo → Critical Analysis → Conclusion

---

> **Speaker notes format:** Normal text is what you say aloud. *Italics* are stage directions.

---

## OPENING (30 seconds)

"Good [morning/afternoon]. My project is a game-playing AI agent for a game I designed called **C_lines** — an original board game inspired by Southern African line-formation stone games.

The brief asks us to demonstrate that performance improves with experience. I'm going to show you exactly that — with numbers and a live demonstration of the agent playing in real time."

---

## SECTION 1 — THE GAME (1 minute)

"Before I get into the AI, let me quickly explain the game — because the game defines the learning problem.

C_lines is played on an 8×8 board. Players take turns placing a single piece anywhere on the board. You score points by forming unbroken lines of three or more pieces in any of the four directions — horizontal, vertical, and both diagonals. Longer lines score more. The game ends when the board is completely full. Whoever has the higher cumulative line score wins.

It's a game of territory and threat. Every move either scores you points, sets up a future line, or blocks your opponent from doing the same.

Crucially: **no version of this game exists anywhere.** There is no database of expert moves, no solver, no published strategy. The agent has to learn entirely from scratch, by playing against itself."

---

## SECTION 2 — TEP FRAMEWORK (2.5 minutes)

"Now let me formally define the learning problem using Mitchell's TEP structure from Chapter 1 of *Machine Learning*.

**Task.** The task is *sequential decision-making*. Given the current board state, choose a cell to place your piece that maximises your chance of winning. I model this as a Markov Decision Process: a state space of board configurations, an action space of 64 cells, a deterministic transition function, and a reward function. The problem category is decision-making — not classification, not prediction.

The state is encoded as a 10-channel tensor. Each channel provides a different tactical feature: where my pieces are, where the opponent's pieces are, open-threat cells, immediate-win and immediate-lose cells, and turn progress. This encoding is the bridge between the raw board and what the neural network sees.

**Experience.** The agent learns from self-play simulated episodes. There is no human data to learn from — the agent starts from zero and plays against itself. Training has three phases: a warm-up phase against random and heuristic opponents, then a main self-play phase where the agent faces frozen copies of its own earlier checkpoints, and throughout all of it, a 10% anchor rate against known baseline opponents to prevent the curriculum from getting stuck.

Every game generates a sequence of transitions — state, action, reward, next state — that go into a replay buffer. The network trains from minibatches sampled from this buffer.

**Performance.** I measure three things:
- Win rate against a random opponent — the floor
- Win rate against a heuristic opponent — a tactical non-learning agent that blocks your lines
- Elo rating — a chess-style skill score updated every 100 games against fixed anchors

The core question — *did performance improve with experience?* — is answered by whether these metrics go up over 10,000 games."

---

## SECTION 3 — THE ALGORITHM (2 minutes)

"I used a Deep Q-Network — DQN. This is a reinforcement learning algorithm that learns a Q-function: Q(state, action) estimates how good it is to take a given action in a given state.

The architecture is a residual convolutional network — the same family used by AlphaGo. Four residual blocks at 64 channels, processing the 10-channel board encoding, producing one Q-value for each of the 64 cells.

I have to be honest with you about something. My first three training runs completely failed to learn. The final model of one run was the *worst* model of that run — it got worse over time. When I analysed why, I found a sign error in the learning rule.

*[Pause for effect]*

In a two-player game, the next state is seen from the **opponent's** perspective. I was adding the opponent's Q-value to the target, which told the network 'a position great for my opponent is great for me.' Every gradient update was pointing in the wrong direction. The correct formula for two-player zero-sum games negates that bootstrap term.

Once I fixed that one sign — plus Double DQN for stability and Huber loss for robustness — everything changed immediately. That is what run_pts_002 demonstrates."

---

## SECTION 4 — RESULTS (2 minutes)

*[Open training_log.csv]*

"This is the actual training log. Let me walk you through what happened.

At game zero: win rate vs random was 52.5%, win rate vs heuristic was 46%, Elo was 796. It's barely learning to play.

*[Scroll to around row 32, game 3100]*

At game 3,100, Elo crossed 900 — the heuristic anchor. The agent is now stronger than the non-learning tactical agent.

*[Scroll to around row 72, game 7000]*

At game 7,000: win rate vs random is 100%, win rate vs heuristic is 100%, Elo is 1,092. Both baselines are completely saturated.

*[Scroll to final row]*

Final: Elo 1,190. The run ended with the best model ever produced — no forgetting, no collapse.

The Elo column is the headline. **It increased at every single evaluation checkpoint across the full 10,000 games.** That is the definition of performance improving with experience.

*[Open benchmark_log.csv]*

And against alpha-beta depth-4 search — this is a principled tree-search engine, not random. The agent started winning 18.8% of games. By game 2,300 it was 50-50. By game 5,700 it won all 32 games in a check. By the end, it was regularly winning 90–100%."

---

## SECTION 5 — LIVE DEMONSTRATION (5–7 minutes)

*[Launch the UI: `python -m src.ui.app`]*

"Let me show you the actual agent playing.

*[Select 8×8, Points-Until-Full]*

This is the game interface. The left panel is the board. The right sidebar shows scores and controls.

*[On level select screen]*

These are the trained snapshots — each one is a frozen checkpoint from training. Notice the difficulty labels: Easy, Medium, Hard, Master. These are assigned from measured win rates, not manually — gen_001 scored 61% vs random and 48% vs heuristic, so it's Easy. gen_010 scored 100%/100%, so it's Master.

*[Select gen_001 — Easy]*

Let me start with the weakest model. One thousand games of training.

*[Play 4–5 moves. Let the AI play. Point out weak moves.]*

You can see it's playing — it's legal — but it's not very strategic. It's not building long lines or blocking mine.

*[Go back, select gen_010 — Master]*

Now the final model. Ten thousand games.

*[Play. Let AI respond. Narrate the tactics.]*

Watch how it responds to my moves. *[Point at a specific move]* It's blocking a line I was building there. And it's extending its own line here. This is pattern recognition from the Q-function — it's learned that blocking and extending long lines has high value.

*[If the AI wins: 'That's the intended outcome — it should beat a casual human player.']*
*[If you're winning: 'I'm playing well here — let me try a less obvious move.']*

This is what 10,000 games of self-play learns."

---

## SECTION 6 — CRITICAL ANALYSIS (1.5 minutes)

"Three quick critical points.

**Was the problem well-posed?** Yes. All three Mitchell criteria are met. The task is fully specified as a deterministic MDP with no external dependencies. The experience is reproducible — given the same seed, you get identical curves. The performance measure is single-valued and directly answers the central question.

**What worked and why?** The negamax sign fix was the decisive change — a single minus sign turned three failed runs into one successful run. Double DQN prevented the late-training collapse seen in earlier experiments. Plateau-based learning-rate decay allowed the network to compress to a precise policy without destabilising.

**What are the honest limitations?** The agent still has isolated exploitable patterns — at game 8,300 it lost all 32 benchmark games despite being near its peak strength. This shows the policy is strong on average but not perfectly robust. I would also want to validate the result with three random seeds to confirm it wasn't a lucky initialisation — right now I have one strong run."

---

## CLOSING (30 seconds)

"To summarise.

I designed an original game, defined it as a well-posed learning problem under Mitchell's TEP framework, implemented a DQN with self-play, and trained an agent for 10,000 games.

Win rate vs random went from 52.5% to 100%. Win rate vs heuristic went from 46% to 100%. Elo rose from 796 to 1,190 — monotonically, across every checkpoint.

Performance improved with experience. The project demonstrates that clearly.

Thank you."

---

## Q&A Cheat Sheet

| If asked... | Say... |
|---|---|
| Why DQN not minimax? | "64-cell action space, 64-move horizon. Full tree is infeasible. Neural network approximates the value function instead of enumerating." |
| What is Elo? | "Chess rating system. I play against fixed anchors (Random=800, Heuristic=900) every 100 games and update with the standard formula. Comparable across all checkpoints." |
| What's the sign error you fixed? | "In a two-player game, the next state is the opponent's turn, encoded from their perspective. I was adding their Q-value to the target — which pushes my Q-values toward what's good for them, not me. Subtracting it is the negamax correction." |
| Why does it lose sometimes at the end? | "The agent has a specific opening it handles poorly. Alpha-beta finds it consistently. Not a collapse — Elo continued rising through that period. An honest limitation." |
| Other board sizes? | "Same code, parameterised by N. Submission timeline meant I validated 8×8 first. It would take one command to train the 9×9 agent." |
| What's in the 10 channels? | "My pieces, opponent pieces, empty cells, open-3 threats for each player, open-4 threats for each player, immediate-win cells, immediate-lose cells, turn progress fraction." |
| Is this AlphaZero? | "Similar residual architecture, same self-play concept. The key difference: AlphaZero adds Monte Carlo tree search to improve policy targets. This project uses value-only DQN without MCTS during training — a simpler but less powerful variant." |
