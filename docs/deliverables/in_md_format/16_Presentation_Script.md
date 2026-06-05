# Presentation Script — C_lines Self-Play RL Agent

**Author:** *----* *S----* · ITRI 616
**Date:** 2026-06-01
**Run presented:** `run_pts_002` — 8×8, points-until-full, 10,000 self-play games
**Length:** ~10 minutes spoken + live demo (Steps from `15_Live_Demonstration_Plan.md`) + Q&A

This is a word-for-word script you can read or paraphrase. Each section has a **[SLIDE]** cue, a spoken part, and a **[DEMO]** cue where the live demonstration slots in. Speaker notes in *(italics)* are not spoken. Timings are cumulative targets. The script is deliberately structured so each block lands one grading criterion.

---

## Slide 1 — Title (0:00–0:30)

**[SLIDE: title — "C_lines: a self-play reinforcement-learning agent for an original board game", your name, ITRI 616]**

"Good [morning/afternoon]. My project is C_lines — an original two-player board game I designed, and an AI agent that learns to play it well entirely by playing against itself. Over the next ten minutes I'll show you how I posed this as a learning problem, the algorithm I used, and — most importantly — quantitative evidence that the agent's performance improves with experience. I'll finish by showing the trained agent playing live."

*(Stay under 30 seconds. Don't read the slide; set up the three-part promise: problem, algorithm, evidence.)*

---

## Slide 2 — The game (0:30–1:45)

**[SLIDE: an 8×8 board with a few stones and a couple of highlighted lines]**

"First, the game. C_lines is played on an 8×8 grid. Two players take turns placing a stone on **any** empty cell — there's no gravity and no column rule, you can play anywhere. Play continues until the board is full, 64 stones.

You score by forming straight lines of your own stones — horizontal, vertical, or diagonal. Longer lines are worth disproportionately more: a line of three is worth a quarter-point, a four is worth one, and it scales up to five points for a line of eight. When the board fills, the higher total score wins, and a tie-break rule removes draws.

What makes it interesting for AI is the tension: every move you have to both **build** your own lines and **block** the opponent's, and because you can play anywhere, the first move alone has 64 options. It's a genuine strategic-planning problem over a 64-move horizon — and it's an original game, so there's no existing engine to copy."

*(This establishes the domain and why it's non-trivial. Keep it concrete — point at a line on the slide.)*

---

## Slide 3 — The learning problem: TEP (1:45–3:30)

**[SLIDE: three boxes — Task / Experience / Performance]**

"I framed this formally using the Task–Experience–Performance structure from Mitchell.

**Task.** Given a board position, choose the placement that maximises my final score margin — my score minus my opponent's when the board fills. In category terms this is a **sequential decision-making and control** problem. It's not classification — there's no fixed label per board — and it's not prediction — the agent has to *act*, and its actions decide the outcome.

**Experience.** The agent learns from games it plays itself. It starts with a short warm-up against a random player and a fixed tactical heuristic, then moves to **self-play**: it plays frozen copies of its own earlier versions, pulled from a growing pool. As the agent gets better, its opponents get better — it builds its own curriculum. No human game records are needed.

**Performance.** I measured improvement four ways: win rate against a random opponent, win rate against the heuristic, an **Elo rating** that summarises overall skill in one number, and — the hardest test — win rate against a 4-ply alpha-beta search. Using four metrics means my improvement claim doesn't depend on any one number that might saturate."

*(This is the 15% TEP criterion — say all three words explicitly and name the problem category out loud. Don't rush it.)*

---

## Slide 4 — The algorithm (3:30–5:30)

**[SLIDE: pipeline diagram — state encoding → residual CNN → Q-values → ε-greedy move; arrow back via replay buffer]**

"The algorithm is a **Deep Q-Network** — value-based reinforcement learning. A neural network estimates Q, the expected future score margin of each possible move, and the agent plays the move with the highest value. During training it explores randomly some of the time, and that exploration decays as it learns.

The board goes in as a **10-channel image**: my stones, the opponent's, empty cells, and seven tactical layers like 'where are my three-in-a-rows', 'where would the opponent win immediately'. The network is a **residual convolutional network** — four residual blocks — which is well suited to spotting line patterns across the board.

Three design choices made the difference between a model that learns and one that doesn't:

- Because this is a **two-player** game, the value of the next position belongs to my *opponent*, so the learning update **subtracts** it rather than adding it — the correct zero-sum, negamax target. Getting that sign right is the single most important detail in the whole implementation.
- I use **Double DQN** and a **Huber loss** to keep the value estimates stable and stop them blowing up.
- And I exploit the board's **eightfold symmetry** — every position is also seven rotations and reflections — which multiplies my training data for free.

Every thousand games I freeze a snapshot, rate it, and register it as a playable difficulty level."

*(This is the 30% algorithm criterion. The three bullets are what distinguishes a strong submission — emphasise the negamax sign point; it shows you understand *why* it works, not just *that* it works.)*

---

## Slide 5 — Results: it improves with experience (5:30–7:30)

**[SLIDE: combined_progress.png — win rates + Elo]**

"Here's the core result, one training run of 10,000 games.

The blue and green lines are win rate against the random and heuristic opponents. The purple dashed line is Elo. Watch all three: they start at random level — about 50% — and climb together to the ceiling. The Elo rises **monotonically from 796 to 1190**, nearly four hundred points, with no late collapse. That smooth, rising curve *is* the assignment's central claim — performance improving with experience — shown directly."

**[SLIDE: benchmark_vs_alphabeta.png]**

"But the strongest evidence is this one — against the 4-ply alpha-beta search the agent **never trains against**. Early in training it wins about 19% of games and loses on score, by an average margin of minus 0.83. By the end it wins essentially **100%**, and the score margin has flipped to **plus 1.17**. The agent taught itself, from self-play alone, to out-play a principled lookahead search."

**[DEMO — Step 3 from the demo plan]** "And these aren't just numbers from a log file. Let me run the final agent live right now."

*(Run `python -m scripts.demo_eval --gen 10 --games 50`. While it runs:)* "This is the final snapshot playing 50 brand-new games against each fixed opponent." *(Read the result aloud: "100% and 100%.")*

*(This is the 30% evaluation criterion. The live number is what makes it bulletproof — it kills any 'cherry-picked' doubt.)*

---

## Slide 6 — See it play (7:30–9:00)

**[DEMO — Step 4: launch `python -m src.ui.app`, go to Level Select]**

"Each of these difficulty levels is a checkpoint from that same run — the registry sets the difficulty from each snapshot's measured Elo, so a harder level is literally a later stage of learning. Let me play the top one, 'Champion'."

*(Play 3–4 moves against gen_010. Narrate one concrete moment:)* "Notice it just blocked my line there rather than extending its own — that's the offence-versus-defence trade-off it learned, nobody coded that rule."

*(Keep this short — 3–4 moves is enough. The point is to make the curve tangible.)*

---

## Slide 7 — Critical analysis (9:00–10:30)

**[SLIDE: four headers — Well-posed? / Improved? / Assumptions / Limitations]**

"Finally, a critical look.

**Was it well-posed?** Yes — and deliberately so. The reward the agent optimises *is* the score margin, which *is* the win condition, so there's no gap between what it's trained to do and what it's graded on. The results confirm that alignment.

**Did it improve?** Unambiguously, on all four metrics at once — and the fact that they rise *together* tells me it's real learning, not evaluation noise.

**Assumptions.** Perfect information, deterministic placement, and that self-play against past selves is a strong enough curriculum. Beating an independent search supports that last one.

**Limitations — and I want to be honest about these.** I report one board size and one game mode, and a single training seed, so I haven't quantified run-to-run variance. Alpha-beta depth-4 is a strong benchmark but not perfect play, so '100% against it' means 'stronger than a 4-ply search', not 'solved'. And the per-snapshot win rates are noisy game-to-game — which is exactly why I lean on the *Elo* curve as the headline, because it smooths that out."

*(This is the 25% critical-analysis criterion. Volunteering limitations *gains* marks — it shows judgement. Don't skip it to save time.)*

---

## Slide 8 — Close (10:30–11:00)

**[SLIDE: one-line summary + the Elo curve thumbnail]**

"To sum up: I posed an original board game as a reinforcement-learning control problem, solved it with a self-play Deep Q-Network, and showed with four independent metrics that the agent learns from a blank slate to beating a 4-ply search — clear, quantitative evidence that performance improves with experience. The code is fully modular and every result I showed is reproducible from the logged data with one command. Thank you — I'm happy to take questions."

---

## Q&A preparation — likely questions and crisp answers

**Q: Why DQN and not a policy-gradient method or AlphaZero-style MCTS?**
"DQN fits a discrete action space and delayed reward cleanly, and it was enough to reach 100% against a 4-ply search. An AlphaZero-style policy-plus-value network with MCTS is the natural next step and would likely push past depth-4 — the architecture is already residual, so it's a small extension."

**Q: How do you know it's not just memorising or exploiting a weak opponent?**
"Two reasons. It wins against alpha-beta, which it never trains against, so it can't have memorised that opponent. And open placement means a 64-branch tree — the state space is far too large to memorise; it has to generalise."

**Q: Why does the win rate jump around between checkpoints?**
"Each point is only 100–200 games, so it has sampling noise — you can see a dip to 70% at game 5,000. That's why the primary metric is Elo, which aggregates across the whole opponent set and comes out smooth and monotone."

**Q: What's the single most important thing that made it work?**
"The two-player value target. Because the next position is the opponent's, the bootstrap has to be subtracted, not added. With the wrong sign the agent learns the opposite of what it should; with the right sign it converges cleanly."

**Q: Is the reward shaping doing too much of the work?**
"The shaping is just the change in score margin — it's the *same* quantity as the final objective, only measured per move instead of once at the end. It speeds learning but doesn't change *what* is being optimised, so it isn't a shortcut to a different goal."

**Q: How long did it take to train, and on what hardware?**
"CPU only, about 450–620 games an hour, so a full 10,000-game run is several hours. No GPU was required."

**Q: What would you do with more time?**
"Repeat the run across several seeds to put confidence bands on the curves, raise the benchmark to deeper search, and try the AlphaZero-style extension."

---

## Delivery checklist

- [ ] Rehearsed end-to-end twice on the demo machine.
- [ ] Three figures pre-opened as fallback (see demo plan §4).
- [ ] Terminal font enlarged; notifications off.
- [ ] Know your two non-negotiable lines: *"Elo rose monotonically from 796 to 1190"* and *"from 19% to 100% against a 4-ply search it never trained on."*
- [ ] If a demo command fails, fall back to the figure within 15 seconds and keep talking.
- [ ] Leave 1–2 minutes for questions.
