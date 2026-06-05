# ITRI 616 Oral Presentation Slides Outline
## C_lines: Teaching a Machine to Play First-to-Four
Duration: 5 minutes | ~10 slides

---

SLIDE 1 - Title (20 sec)
C_lines: Teaching a Machine to Play First-to-Four
- Your name, module code, date
- Visual: screenshot of board with 4-in-a-row highlighted

---

SLIDE 2 - The Game (30 sec)
- 8x8 board, two players alternate placing pieces
- First to form 4 in a row (horizontal/vertical/diagonal) wins
- Visual: 3-frame diagram showing a 4-in-a-row completion
Say: "This is a game I designed. The goal is simple but the optimal strategy requires planning ahead."

---

SLIDE 3 - TEP Framework (40 sec)
Task:        Select cell placements to complete 4-in-a-row first
Experience:  20,000 self-play games with curriculum opponents
Performance: Episode length (lower), WR vs random (higher), WR vs AlphaBeta search (higher)
Say: "The TEP framework requires performance to be measurable. I use three independent metrics so one weak metric cannot hide a failure."

---

SLIDE 4 - The Agent: How It Learns (40 sec)
Board state (10x8x8) -> ResNet (4 residual blocks, 64 channels) -> Q-values (64 cells)
- Algorithm: Double DQN -- agent plays against itself, stores (state, action, reward) tuples
- Key design: reward the agent for creating threats, not just winning
Say: "The network estimates the value of each cell placement. Training is entirely self-play -- no human moves needed."

---

SLIDE 5 - MAIN RESULT: Performance Improves (60 sec)
[Large graph: Episode Length vs Training Games -- 33 down to 8.9]
Three numbers in large text:
  Episode length:  33.3 -> 8.9 moves  (-73%)
  WR vs random:    48%  -> 100%
  Elo:             +98 points  (monotone over 12,000 games)
Say: "At game zero the agent plays randomly -- 33 moves per game. By game 7,000 it wins every game in under 10 moves. This is unambiguous improvement."

---

SLIDE 6 - Beating a Search Agent (40 sec)
Win rate vs AlphaBeta Depth-4 (looks 4 moves ahead):
  Games 0-7,000:  0%
  Game 8,000:    37.5%  <- first breakthrough
  Game 9,000:    50.0%  <- PARITY WITH SEARCH
  Game 14,000:   45.3%
Say: "AlphaBeta cannot be bluffed -- it uses minimax search. The DQN reaching 50% means it has internalised genuine tactical patterns."

---

SLIDE 7 - Training Phases (30 sec)
Phase 1 (0-2k):    Random opponents     -> buffer fills
Phase 2 (2k-7k):   Self-play pool       -> fastest learning, ep_len 28->10
Phase 3 (7k-10k):  Tactical refinement  -> 50% vs AlphaBeta
Phase 4 (10k-20k): Harder curriculum    -> adaptation to stronger opponent
Say: "The sharpest improvement happens in Phase 2 -- 5,000 games where the agent plays its own past versions."

---

SLIDE 8 - Critical Analysis (30 sec)
What worked:                     What needs improvement:
- Episode length metric           - WR vs heuristic oscillated (8-27%)
- Symmetry augmentation (8x)      - Bimodal benchmark (0-50%)
- Curriculum ladder               - No search at inference
- Reward shaping                  - Single seed only
Say: "The main limitation is that the agent has mastered specific opening patterns but has not fully generalised. MCTS at inference is the clearest next step."

---

SLIDE 9 - Demo (30 sec)
Live: open UI, select 8x8 FTF, pick a snapshot, play one game
Say: "You can play against any of the 24 trained checkpoints. The level select ranks them by difficulty."

---

SLIDE 10 - Summary (20 sec)
- Demonstrated: performance improves with experience (TEP confirmed)
- Peak: 50% vs AlphaBeta-4, 100% vs random, episode length 8.9
- Next: MCTS self-play, GPU training, board sizes 9x9 to 12x12

---

TIMING GUIDE
Slide 1: 0:20 | Slide 2: 0:50 | Slide 3: 1:30 | Slide 4: 2:10
Slide 5: 3:10 | Slide 6: 3:50 | Slide 7: 4:20 | Slide 8: 4:40
Slides 9-10: 5:00

---

LIKELY QUESTIONS

Q: Why not use MCTS from the start?
A: MCTS requires 200 simulations per move -- 200x slower per game. On CPU that makes 10,000 games take months. The DQN achieves 50% vs AlphaBeta without search, which is the more remarkable result.

Q: How does the agent compare to a human?
A: Against a naive human it wins consistently. Against an experienced defensive player (simulated by the heuristic), it scores 8-27%. A human who knows to block threats can challenge it.

Q: Is the 50% vs AlphaBeta reproducible?
A: Single seed so far. Multi-seed validation is the next experiment. The bimodal benchmark (0-50% oscillation) suggests the result is position-dependent.

Q: What is the 10-channel input?
A: It pre-computes threat information -- open-3 threats, forced-win threats, immediate-win cells -- so the network does not need to discover these from scratch. It is domain knowledge injected as features.
