# Presentation Script — ITRI 616 (Final Version)
## Structured for BOTH rubrics simultaneously

**Total time: 12–15 minutes**

---

> Normal text = say aloud. *Italics* = stage directions. **Bold** = emphasis.

---

## OPENING — Hook and Purpose (45 seconds)

"Across Southern Africa, there is a long tradition of line-formation board games. Stone games, seed games, grid games — played for generations before any digital version existed.

My project takes one of those traditions, builds it as a digital game, and then asks: **can a machine learn to play it well — with no human teaching it, no rules written for it, just by playing against itself?**

The answer is yes. And I have the numbers to prove it."

---

## PART 1 — THE GAME: What it is and why it matters (1.5 minutes)

*[If demo running in background, point at screen briefly]*

"The game is called **C_lines**. I designed it for this project, inspired by Southern African line-formation traditions.

It is played on an 8×8 board. You and your opponent take turns placing one piece anywhere on the board. You score points by forming unbroken lines of three or more pieces in any of the four directions. Longer lines score more. The board fills completely. Whoever has the higher total line score wins.

**Why this game?** Three reasons.

First — **purpose.** Digitising a traditional game makes it accessible. Anyone with a computer can now play a variant of a centuries-old tradition. The AI opponent means you can practise alone.

Second — **challenge.** This game has no published strategy, no expert database, no solver. There is nothing for the AI to learn *from* except itself. That is a harder and more interesting problem than a game like chess where you can feed it human moves.

Third — **clarity.** This game is simple enough to explain in two sentences but complex enough that a random player loses to a thinking one. That makes it a clean test of whether the AI is actually learning or just getting lucky."

---

## PART 2 — TEP FRAMEWORK (2.5 minutes)

"Mitchell's TEP structure from *Machine Learning* Chapter 1 gives us a rigorous way to define a learning problem. Let me walk through each component.

**Task.** The task is: *given the current board, choose a cell to place your piece to maximise your chance of winning*. The problem category is **sequential decision-making** — specifically a Markov Decision Process. Not classification, not prediction. The agent learns a *policy* — a rule mapping any board state to an action.

I represent the board as a 10-channel tensor. Each channel is one tactical feature: where my pieces are, where the opponent's pieces are, which cells are empty, where immediate threats exist, where I can win in one move, where I can lose in one move, and how far through the game we are. This 10-feature encoding is what the neural network sees.

**Experience.** The agent learns from **self-play** — it generates its own training data by playing games and recording the results. No human has taught it anything. It starts at random and gradually builds understanding.

The training has three phases. First, a warm-up against simple opponents — random and heuristic — to give the buffer some non-trivial positions to learn from. Then a self-play phase where the agent plays mostly against frozen copies of earlier versions of itself. And throughout, 10% of games are always against fixed baseline opponents, so the curriculum never loses its anchor.

Every game produces 64 transitions — state, action, reward, next state — stored in a replay buffer. The network trains from random mini-batches, 4 gradient steps per game.

**Performance.** I measure three things: win rate against a random opponent, win rate against a heuristic opponent, and Elo rating. Elo is a chess-style skill score — I update it every 100 games against two fixed reference agents. The key property: **it is comparable across every checkpoint**. It does not saturate at 100% and does not floor at 0%. It gives a clean, monotone curve that directly answers 'is the agent improving?'"

---

## PART 3 — ALGORITHM (2 minutes)

"I used a **Deep Q-Network — DQN**. It is a reinforcement learning algorithm that learns a Q-function: an estimate of how valuable each possible action is from the current state.

The architecture is a **residual convolutional network** — the same family used in AlphaGo. Four residual blocks at 64 channels, processing the 10-channel board encoding, producing one Q-value for each of the 64 cells. The cell with the highest Q-value is where the agent places its piece.

I want to be honest with you about something. **My first three training runs completely failed.**

*[Pause]*

The final model of each run was the worst model of that run. Performance actually went backwards. When I analysed why, I found a **sign error in the learning rule**.

Here is the core Q-learning update:

`target = reward + γ × max Q(next_state)`

In a two-player game, the next state is seen from the **opponent's** perspective. I was *adding* their Q-value — which told my agent: 'a position that is great for my opponent is great for me.' Every gradient step was pointing in the wrong direction.

The correct formula for two-player zero-sum games **negates** that term:

`target = reward − γ × max Q(next_state)`

One minus sign. That's the difference between a model that gets worse and a model that reaches 100% win rate.

I also added Double DQN — which separates action selection from action evaluation to prevent value overestimation — and Huber loss instead of MSE, which is more robust to large errors. But the sign fix was the decisive change."

---

## PART 4 — RESULTS (2 minutes)

*[Open training_log.csv]*

"This is the actual training log. Let me show you three numbers.

**Game 0:** Win rate vs random = 52.5%. Win rate vs heuristic = 46%. Elo = 796.

*[Scroll to game 3100]*

**Game 3,100:** Elo = 901. The agent has crossed the heuristic anchor. It is now stronger than the non-learning tactical agent.

*[Scroll to game 7000]*

**Game 7,000:** Win rate vs random = **100%**. Win rate vs heuristic = **100%**. Elo = **1,092**.

*[Scroll to final row]*

**Game 9,999:** Elo = **1,190**. The final model is the best model. No collapse, no reversal.

The Elo column — 796 to 1,190 — rose at every single evaluation checkpoint across 10,000 games. That is the definition of performance improving with experience.

*[Open benchmark_log.csv briefly]*

And against alpha-beta depth-4 — a principled tree-search engine — the agent started at 18.8% and by game 5,700 was winning 100% of benchmark games. It beat tree search. That is not trivial."

---

## PART 5 — LIVE DEMO (4–6 minutes)

*[Launch: `python -m src.ui.app`]*

"Let me show you the agent playing in real time.

*[Navigate: 8×8 → Points-Until-Full → Level Select]*

These cards are the trained snapshots. Each is a frozen checkpoint from training. The difficulty label — Easy, Medium, Hard, Master — is assigned from the actual measured win rate, not manually. gen_001 scored 48% against heuristic — that is Easy. gen_010 scored 100% — that is Master.

*[Select gen_001 — Easy]*

First, the weakest model. 1,000 games of training.

*[Play 4–5 moves. Let AI respond. Point at weak play.]*

You can see it is playing legally but not tactically. It is not blocking my lines or building long sequences. It has learned some basics but not much strategy.

*[Back to level select, select gen_010 — Master]*

Now the final model. 10,000 games of self-play.

*[Play. Narrate AI moves.]*

Watch here — *[point]* — it is blocking a line I was building. And there — *[point]* — it is extending its own sequence in two directions simultaneously. A fork threat. That is not something you hand-code. That is something it learned from playing against itself thousands of times.

*[If the AI wins: 'Good — it should.']*
*[If you are winning: 'It has a known weakness with this particular opening. I discuss that in my limitations.']*"

---

## PART 6 — CRITICAL ANALYSIS (1.5 minutes)

"Three critical points to close.

**Was the problem well-posed?** Yes. Mitchell's three criteria: the task is fully specified with no hidden state, the experience is reproducible — same seed gives identical curves — and the performance metric is single-valued and directly answers the question. The problem was well-posed.

**What was the most important finding?** That algorithm choice alone is not enough. I ran the same architecture three times with a wrong learning rule and got three failed runs. The fourth run — with the correct negamax TD target — produced a monotone learning curve all the way to 100%. The theory behind the algorithm matters as much as the implementation.

**What are the honest limitations?** One: the agent has an isolated exploitable weakness — at game 8,300 it lost all 32 benchmark games despite being near peak strength. Two: this is one training run; confirming the result across multiple seeds with confidence intervals would make it statistically airtight. Three: the board is 8×8 and training was CPU-bound at ~500 games per hour. A GPU run with 100,000 games would produce a tighter, more robust policy."

---

## CLOSING (30 seconds)

"To summarise.

I designed a Southern African-inspired game, defined it as a well-posed learning problem under Mitchell's TEP framework, implemented a corrected DQN with self-play, and trained an agent for 10,000 games.

Win rate vs random: **52% → 100%**.
Win rate vs heuristic: **46% → 100%**.
Elo: **796 → 1,190 — monotonically**.

Performance improved with experience. Clearly and measurably. Thank you."

---

## Q&A PREPARATION

### Rubric Criterion 5: Response to questions and critical thinking

Prepare for these — the hardest ones get you the full 4 marks:

---

**"How is C_lines related to Southern African games?"**

"Southern Africa has a long tradition of line-formation and territory games played with stones on grids drawn in the ground or carved into rock. Morabaraba, for example, is a mill-style game. Line-scoring games — where you score points for forming rows — appear in various forms across the region. C_lines is a modernised, digitalised variant that preserves the core mechanic: placing pieces to build lines, competing for territorial control. The scoring system was designed to create meaningful decisions at every move, rather than a binary win/lose outcome, which is what makes it suitable for this kind of learning."

---

**"What is the TEP framework and why does it matter?"**

"Mitchell defines a learning problem as well-posed if you can specify what the agent must learn to do — the Task — what data it learns from — the Experience — and how you measure whether it improved — the Performance. It matters because it forces you to make your assumptions explicit. If any of the three is vague, you cannot know whether your agent is genuinely improving or just exploiting a flaw in the evaluation. In this project, the Elo metric is specifically designed so the performance measure cannot be gamed — the anchor agents never change, so a higher Elo always means a stronger agent."

---

**"Why did your first three runs fail?"**

"In standard single-player DQN, the bootstrap target is `reward + γ × max Q(next_state)`. In a two-player game the next state is encoded from the opponent's view — so `max Q(next_state)` is the opponent's best value. Adding it tells the agent 'what's great for my opponent is great for me.' Subtracting it — the negamax correction — correctly implements the zero-sum principle: my value equals the negative of my opponent's value. The three failed runs had the plus sign. The fourth had the minus sign."

---

**"You said the agent beat alpha-beta. How is that possible if alpha-beta uses lookahead?"**

"Alpha-beta depth-4 searches four moves ahead but evaluates positions using a hand-crafted heuristic — a weighted count of line lengths and open threats. The DQN's Q-function has been trained on hundreds of thousands of positions. By the end of training its positional evaluation is more accurate than the hand-crafted heuristic alpha-beta uses. So while alpha-beta looks further ahead, the DQN evaluates positions more accurately. Accuracy beats depth."

---

**"What would you do differently?"**

"Three things. First, validate across three random seeds — one run is promising but not statistically definitive. Second, implement n-step returns to propagate end-of-game outcomes faster through the value function. Third, given more compute, try AlphaZero-style training where MCTS generates improved policy targets during self-play — the infrastructure (residual network, snapshot pool, benchmark) is already in place. The architecture is literally designed to support it."

---

**"What is Elo and why use it?"**

"Elo is the rating system used in chess. I maintain two fixed reference agents: Random at 800, Heuristic at 900. Every 100 training games, the agent plays 20 games against each and I update its rating using the standard Elo formula. It has three advantages over simple win rate: it does not saturate at 100%, it is comparable across checkpoints that use different evaluation opponents, and it produces a smooth curve that is interpretable as absolute skill level. An agent at Elo 1,100 is meaningfully stronger than one at 900, regardless of what opponents were available at each checkpoint."

---

**"Why not use a stronger algorithm like AlphaZero or PPO?"**

"DQN was chosen deliberately for three reasons. First, it fits the task exactly — discrete actions, perfect information, value-based. Second, it produces the clearest 'improves with experience' curve, which is exactly what the brief grades. Third, it is within the brief's allowed algorithm families. AlphaZero would be stronger — that is a genuine limitation — but it would also require implementing MCTS-based policy improvement, which is a full additional research thread beyond the project scope."
