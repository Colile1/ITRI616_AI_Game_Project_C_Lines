# Video Presentation Script — C_lines (recorded option, due Friday)

**Author:** Colile Sibanda · ITRI 616
**Deck:** `docs/deliverables/presentation/C_lines_presentation.pptx` (11 slides)
**Target length:** 8–10 minutes
**Use this if** you present a recorded video on Friday instead of live tomorrow.

Read this almost verbatim; it is paced for narration. Each block is one slide. *(Italics = direction, not spoken.)* Timings are per-slide targets. The script is written to score on every line of the oral rubric — Problem + TEP, Technical understanding, Results + analysis, Communication, and critical thinking — so even without a live demo the video covers all five.

---

## Recording setup (5-minute checklist)

- Screen-record the slide deck in presenter/full-screen mode.
- Optional B-roll to drop in where marked **[SHOW]**: a 15–20 s screen capture of `python -m scripts.demo_eval --gen 10 --games 50` printing "100% / 100%", and a 15 s clip of `python -m src.ui.app` Level Select with you playing a move. If recording these is awkward, the figures on the slides already carry the evidence — narrate them instead.
- Record audio in a quiet room; do one practice pass for timing.
- Keep your cursor still except when pointing at a curve.

---

## Slide 1 — Title (0:00–0:30)

"Hi, I'm Colile Sibanda, and this is my ITRI 616 project: C_lines — an original board game I designed, and an AI agent that learns to play it well entirely by playing against itself. Over the next few minutes I'll show you how I framed this as a learning problem, the algorithm I used, and quantitative evidence that the agent's performance improves with experience."

*(Calm, confident open. Don't read the subtitle aloud.)*

## Slide 2 — The game (0:30–1:45)

"First the game. C_lines is played on an 8×8 grid. Players take turns placing a stone on **any** empty cell — there's no gravity and no column rule, you can play anywhere. Play continues until the board is full, all 64 cells.

You score by forming straight lines of your own stones — horizontal, vertical, or diagonal — and longer lines are worth disproportionately more, from a quarter-point for three in a row up to five points for a line of eight. When the board fills, the higher total score wins. On the left, the yellow line shows a scoring run of four.

What makes it interesting for AI is the tension on every move: you have to build your own lines while blocking your opponent's, and from the empty board there are 64 possible moves. It's a genuine planning problem over a 64-move horizon — and because it's an original game, there's no existing engine to copy."

## Slide 3 — TEP framework (1:45–3:15)

"I framed the problem formally using the Task–Experience–Performance structure.

The **Task**: given a board, choose the placement that maximises my final score margin — my score minus my opponent's. In category terms this is a sequential **decision-making and control** problem — not classification, because there's no fixed label per board, and not prediction, because the agent has to act and its actions decide the outcome.

The **Experience**: the agent learns from games it plays itself. It starts with a short warm-up against a random player and a fixed heuristic, then moves to self-play against frozen copies of its earlier selves. As it gets better, its opponents get better — it builds its own curriculum, with no human game data.

The **Performance**: I measure improvement four ways — win rate against random, against the heuristic, an Elo rating, and win rate against a 4-ply alpha-beta search. Four metrics so the claim doesn't depend on any single number."

## Slide 4 — The algorithm (3:15–4:45)

"The algorithm is a Deep Q-Network — value-based reinforcement learning. The board goes in as a 10-channel image — my stones, the opponent's, and tactical layers like where the threats are. A residual convolutional network estimates the value of each possible move, and the agent plays the highest-value legal cell, exploring randomly early in training and less as it learns.

I chose DQN because this is sequential control with delayed reward — a move's true value only emerges when the board fills — and Q-learning's temporal-difference target is built exactly for that credit-assignment problem. Supervised learning isn't even an option here: there are no labelled 'correct moves' for an original game.

The reward is the change in score margin each move, plus a terminal plus-or-minus one. Because the goal *is* to accumulate the higher score, that reward points at the true objective on every single step."

## Slide 5 — Three implementation choices (4:45–6:00)

"Three implementation details made the difference between an agent that drifts and one that learns.

First, the **value target**. Because this is a two-player game, the next position belongs to my *opponent* — so its value has to be **subtracted**, not added. That negamax sign is the single most important line of code in the project; with the wrong sign the agent learns the opposite of what it should.

Second, **Double DQN and a Huber loss**, which keep the value estimates from blowing up — the online network picks the move, the target network scores it.

Third, **symmetry**: every board is also seven rotations and reflections, so I get eight times the training data for free.

On top of that, snapshots are saved every thousand games — and each one becomes a playable difficulty level."

## Slide 6 — Results: improvement (6:00–7:15)

"Here's the core result — one training run of 10,000 games. The blue and green lines are win rate against the random and heuristic opponents; the purple dashed line is Elo. Watch them rise together from random level — about 50% — up to the ceiling. The Elo climbs **monotonically from 796 to 1190**, nearly four hundred points, with no late collapse. That smooth rising curve is exactly what 'performance improves with experience' looks like."

**[SHOW — optional]** *(cut to the demo_eval clip)* "And these aren't just numbers from a log — here's the final agent playing 50 fresh games live, winning 100% against both fixed opponents."

## Slide 7 — The decisive benchmark (7:15–8:15)

"But the strongest evidence is this one — against a 4-ply alpha-beta search the agent **never trains against**. Early on it wins about 19% of games and loses on score, by an average margin of minus 0.83. By the end it wins essentially 100%, and the score margin has flipped to plus 1.17. Purely from self-play, the agent taught itself to out-play a principled lookahead search."

## Slide 8 — Difficulty levels (8:15–8:45)

"Each difficulty level you could play against is one of these checkpoints. The registry assigns the difficulty from each snapshot's measured Elo — so a harder level is literally a later stage of learning, from 796 at the start to 1190 at the end."

**[SHOW — optional]** *(cut to Level Select clip; play one move)* "When I play the strongest one, it blocks my line rather than extending its own — that offence-versus-defence judgement is learned, not coded."

## Slide 9 — Critical analysis (8:45–9:45)

"A critical look. Was it well-posed? Yes — by design: the reward the agent optimises *is* the win condition, so there's no gap between training and grading. Did it improve? Unambiguously — and the four metrics rise *together*, which tells me it's real learning, not noise in any one of them.

My assumptions were perfect information, deterministic play, and that self-play is a strong enough curriculum — and beating an independent search supports that last one.

And the limitations, honestly: I report one board size and one mode, a single training seed so I haven't quantified run-to-run variance, and 'beating depth-4' means stronger than a 4-ply search, not perfect play. The per-checkpoint win rates are also noisy, which is exactly why I lean on the Elo curve as the headline."

## Slide 10 — Reflection (9:45–10:30)

"Reflecting on the project: the biggest lesson was that **correctness beats complexity** — the hardest part wasn't the engineering, it was getting that two-player value target right, because a subtle error in the objective still runs and the loss still falls, it just optimises the wrong thing. Second, **measurement is part of the method** — adopting Elo as the primary metric gave me one curve I could actually trust. And third, the engineering discipline — one config file, full logging, a snapshot registry — meant that when it worked, I could prove it with reproducible figures."

## Slide 11 — Conclusion (10:30–11:00)

"To sum up: I posed an original board game as a reinforcement-learning control problem, solved it with a self-play Deep Q-Network, and showed on four independent metrics that the agent learns from a blank slate to out-playing a 4-ply search — clear, quantitative evidence that performance improves with experience. Every result is reproducible from the logged data with one command. Thank you for watching."

---

## Timing summary

| Slides | Content | Cumulative |
|---|---|---|
| 1–2 | Intro + game | 1:45 |
| 3 | TEP | 3:15 |
| 4–5 | Algorithm + key choices | 6:00 |
| 6–8 | Results + benchmark + levels | 8:45 |
| 9–10 | Critical analysis + reflection | 10:30 |
| 11 | Conclusion | 11:00 |

If you need to hit a hard 8-minute cap: trim the optional **[SHOW]** clips and tighten slides 2 and 5 — the must-keep beats are TEP (slide 3), the two results slides (6–7), and critical analysis (slide 9).

## Two lines you must deliver clearly (the whole grade leans on them)

1. "The Elo rose monotonically from 796 to 1190."
2. "From 19% to 100% against a 4-ply search it never trained on."
