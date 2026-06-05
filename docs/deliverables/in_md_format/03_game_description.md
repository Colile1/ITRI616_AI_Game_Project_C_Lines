# C_lines — Game Description and Rules

**Author:** *----* *S----*
**Module:** ITRI 616
**Date:** 2026-05-28
**Status:** Ruleset locked; improvement suggestions advisory

---

## 1. Identity

```
Game name        : C_lines
Short description: Two-player open-placement line-forming game on a flat NxN board
                   with variable-length scoring and a custom no-draw tie-break.
Board type       : Square grid, no gravity, no obstacles
Grid size        : NxN where N is chosen from {8, 9, 10, 11, 12} at game start
Number of players: 2 (Player 1 plays light-blue glass pieces, Player 2 plays warm-amber)
Turn structure   : Strictly alternating; Player 1 moves first
Information      : Perfect (both players see the full board at all times)
Originality note : Original variant — combines open placement, square grid, tunable size,
                   and a graded line-length scoring schedule. Not Connect Four (no gravity),
                   not Gomoku (line length is graded, not fixed at 5), not Renju
                   (no forbidden moves), not Five-in-a-Row.
```

## 2. Components

A square grid of N×N empty cells, where N is chosen at the start of the game from the set {8, 9, 10, 11, 12}. Two players: Player 1 and Player 2, with distinct piece colours (light-blue frosted glass for P1, warm amber for P2). An unlimited supply of pieces — neither player is capped on how many pieces they may place. A turn indicator and a running score-board. In Mode 2, a tie-break counter (0 to 3).

## 3. Game modes

The player chooses one of two modes at the start of each game from the main menu:

### Mode 1 — First-to-Four

The first player to form an uninterrupted line of exactly 4 of their own pieces, in any of the four directions (horizontal, vertical, NE-SW diagonal, NW-SE diagonal), wins immediately. If the board fills before either player makes a 4-line, the game is a draw. This mode is the closest analogue to traditional 4-in-a-row games and exists as the "quick" mode.

### Mode 2 — Points-Until-Full

Both players continue placing pieces until the board is completely full. Each line of length L (where L is the maximal uninterrupted run of one player's pieces in a single direction) scores points according to the schedule below. The player with the higher cumulative score wins. If scores are equal, the tie-break protocol runs (Section 6).

#### Mode 2 scoring schedule

| Line length L | Points |
|---------------|--------|
| 3             | 0.25   |
| 4             | 1.00   |
| 5             | 2.00   |
| 6             | 3.00   |
| 7             | 4.00   |
| 8             | 5.00   |

Lines of length 2 or shorter score nothing. Lines longer than 8 are only possible on boards of size 9 or larger, but the schedule caps at 8 — extra pieces on a single line beyond length 8 do not increase the score (a 10-in-a-row scores the same 5 points as an 8-in-a-row would).

#### Why this schedule

The schedule is super-linear up to 5, then linear from 5 onwards. This rewards reaching the iconic 4-line milestone (the doubling from 0.25 → 1.00 at length 4 is the largest single jump), encourages building beyond it for compounding gains, but flattens off so the strategy doesn't degenerate into "always extend the longest line" on the large boards.

## 4. Setup

The board is empty. Player 1 is selected to move first (UI may offer a toggle or coin-flip, but the default is "Player 1 first"). A turn counter starts at 1.

## 5. On Your Turn

### 5.1 Placement (the only regular-play action)

On your turn, you must place exactly one of your pieces on any empty cell of the board. The choice is unconstrained — no opening restrictions, no centre-only first move, no banned cells, no pie/swap rule in the default ruleset (see Section 8 for an optional swap rule). Placement is mandatory: you may not pass.

```
Action name : place
Precondition: target cell (r, c) is empty AND 0 <= r < N AND 0 <= c < N
Effect      : board[r][c] := current_player; advance turn counter; switch active player
```

### 5.2 Removal (only during Mode 2 tie-break — see Section 6)

```
Action name : remove
Precondition: tie-break is active AND target cell (r, c) contains an opponent piece
Effect      : board[r][c] := empty; rescore both players
```

## 6. Terminal conditions and tie-break

### 6.1 Mode 1 terminal

The game ends as soon as the cell just placed completes (or is part of) a line of exactly 4 of the placing player's pieces. The placing player wins. If the board fills with no 4-line for either player, the game is a draw (rare on N>=8).

### 6.2 Mode 2 terminal — the tie-break protocol

When the board is full, both players' scores are computed using the line-counting algorithm in Section 7. Three outcomes are possible:

1. **One player has a strictly higher score** — that player wins; game ends.
2. **Scores are equal** — enter the tie-break sub-game (described below).
3. **Tie-break exhausted** — game is a draw.

The tie-break sub-game:

```
Initialise tiebreak_round := 1.

WHILE tiebreak_round <= 3 AND scores are equal:
    Each player removes exactly one of the opponent's pieces.
        - Player 1 chooses a Player 2 piece first.
        - Player 2 chooses a Player 1 piece second.
        - In single-player mode, the AI uses its policy network for both removals,
          masked to opponent-only legal cells.
    Recompute both scores using the line-counting algorithm.
    IF scores now differ:
        Higher-scoring player wins; exit tie-break.
    ELSE:
        tiebreak_round := tiebreak_round + 1.

IF scores are still equal after round 3:
    Game ends in a draw.
```

The tie-break does not change which mode is active — it is a coda to Mode 2 only. Mode 1 does not invoke it.

### 6.3 Why three rounds

Three rounds gives both players up to six total opportunities to break the tie (each player removes one piece per round, three rounds). Each removal both removes points from the opponent and may break the opponent's lines, so the score swing per round is non-trivial. Capping at three prevents pathological grind-down endings while keeping draws genuinely rare.

## 7. Line counting — the scoring algorithm

This section is the single source of truth for how scores are computed. The implementation in `src/engine/rules.py` must follow this exactly.

### 7.1 Directions

There are four directions, defined by their (dr, dc) increment vectors:

```
Horizontal :  ( 0, +1)
Vertical   : (+1,  0)
Diagonal-1 : (+1, +1)   (NW-to-SE)
Diagonal-2 : (+1, -1)   (NE-to-SW)
```

Lines are direction-aware: a horizontal run of 5 of Player 1's pieces is a different line from a vertical run of 5 of Player 1's pieces, even if they share a cell.

### 7.2 Maximal-run rule (the uniqueness rule)

For each direction independently, scan every maximal uninterrupted run of one player's pieces and score it by its **maximal length**, once.

* A run of length 5 (e.g. `OOOOO`) scores 2 points (as a 5-in-row). It does **not** score 1+1 as "two overlapping 4-in-row segments". It does **not** double-count.
* A run of length 6 scores 3 points (as a 6-in-row), not 2 (as a 5) plus 1 (as a 4), not 3×0.25 (as three 3s).
* A run that is interrupted by an empty cell or an opponent piece is split into two runs at that point, and each side is scored separately by its own length.

Formally, in each direction:

```
For each starting cell (r, c) such that the cell behind it (r-dr, c-dc) is NOT the
same player's piece (i.e. (r,c) is the START of a maximal run):
    Walk forward in direction (dr, dc) counting same-player pieces until you hit
    an empty cell, an opponent piece, or the board edge.
    Let L be that count.
    If L >= 3, add points_for_length(L) to that player's score.
```

This sweep is done once per direction, per player. It guarantees each maximal segment is counted exactly once, in exactly one direction.

### 7.3 Cross-direction overlap

A single piece may participate in lines in multiple directions simultaneously, and each such line is scored independently. For example, a Player 1 piece at the centre of an `X` of 5-runs (horizontal, vertical, both diagonals) contributes to four separate lines that score 2 points each, total 8. This is intended — it rewards "fulcrum" pieces that anchor multiple lines.

### 7.4 Length cap

The schedule explicitly caps at length 8. A run of length L > 8 is scored as if L = 8 (i.e. 5 points), with no further increment. On a 12×12 board the maximum theoretical run length is 12, so the cap actually bites; on an 8×8 board it never does.

## 8. Tunable parameters

These belong in `src/config.py` as the single source of truth:

```python
# Board
BOARD_SIZES          = [8, 9, 10, 11, 12]
DEFAULT_BOARD_SIZE   = 10

# Modes
MODE_FIRST_TO_FOUR   = "first_to_four"
MODE_POINTS_FULL     = "points_full"
DEFAULT_MODE         = MODE_POINTS_FULL

# Mode 2 scoring
SCORE_FOR_LENGTH = {3: 0.25, 4: 1.0, 5: 2.0, 6: 3.0, 7: 4.0, 8: 5.0}
MIN_SCORING_LENGTH = 3
MAX_SCORING_LENGTH = 8

# Tie-break
TIEBREAK_MAX_ROUNDS  = 3

# Optional rule toggles (see Section 9)
USE_SWAP_RULE        = False
USE_OPENING_RESTRICT = False
```

## 9. Suggested rule improvements

These suggestions are **advisory** — they are not part of the locked default ruleset, but each is a self-contained extension that can be toggled via `config.py`. Each is justified by what it solves; each comes with a cost.

### 9.1 Swap rule (pie rule) — fairness against first-mover advantage

After Player 1 places their first piece, Player 2 may either accept Player 1's move and play normally, or **swap colours** (so Player 2 inherits the piece Player 1 just placed, and Player 1 now plays second). The rule is borrowed from Hex and Renju, where it is the standard solution to first-mover advantage.

**Why include it.** On an open-placement N×N grid, the first move's location is meaningful — centre placements compound into stronger threats than corner placements. The swap rule pressures Player 1 into a "barely advantageous" first move, equalising the game without changing any other rule.

**Cost.** Adds a swap decision into the action space, which complicates the agent's state machine and the UI's turn-handling. Recommended only for competitive play, not for the AI training run (the agent already cycles which side it plays during self-play, which is a different but parallel mitigation).

**Recommendation.** `USE_SWAP_RULE = False` by default. Enable for human-vs-human tournaments.

### 9.2 Opening restriction — central or near-central first placements only

Player 1's first piece must be placed within a centred K×K sub-grid (e.g. K = 4 for a 10×10 board). Stops degenerate corner openings and concentrates early play.

**Cost.** Adds a check to the legal-moves enumerator on turn 1. Negligible to implement.

**Recommendation.** `USE_OPENING_RESTRICT = False` by default.

### 9.3 Hint system — single-player aid using a weak agent

In single-player mode, expose a "Hint" button that calls a weak snapshot's policy and highlights its top-1 recommended cell. Useful for novice players learning the game.

**Cost.** Free if the snapshot pool already exists. UI: one button, one highlight overlay.

**Recommendation.** Include. Off by default; toggle in the in-game pause menu.

### 9.4 Move replay — post-game review

After each game ends, the move list is preserved; the player can step through the game move-by-move via arrow keys, with each placement shown in order. Especially valuable for understanding why the AI won or lost.

**Cost.** Light. The move list is a simple append-only log; the replay viewer is a separate small screen.

**Recommendation.** Include in v1 if time permits, otherwise v1.1.

### 9.5 Threat highlighting (assist mode)

The UI optionally highlights cells where placing a piece would (a) complete a 4-line of your own, or (b) block an opponent's 4-line. Pure visual aid; does not change rules.

**Cost.** A pre-computed overlay each frame using the same legality functions the rules use. Light.

**Recommendation.** Include as an accessibility/training toggle. Off in competitive modes; on in tutorial mode.

### 9.6 Time control — per-turn or per-game clock

Optional Fischer-style increment per move or simple per-game cap. Adds tension; not core to the AI-learning goal.

**Cost.** UI clock, turn-timeout handler, possible auto-pass behaviour. Moderate.

**Recommendation.** Out of scope for the ITRI 616 submission; nice extension.

### 9.7 Mode 3 — Best-of-N match

A match of N (odd) games where the players alternate first-move, and the cumulative match score determines the winner. Useful for evaluating snapshots head-to-head.

**Cost.** Light — wraps the existing single-game loop. The evaluation harness already runs N-game series, so the engine logic exists.

**Recommendation.** Include in the evaluation tooling (already part of `evaluator.evaluate()`); optional in the UI.

### 9.8 Resignation

A player may resign at any time, immediately conceding the game. Standard in competitive game UIs.

**Cost.** One button, one confirmation dialog. Negligible.

**Recommendation.** Include — improves UX on long Mode-2 games.

### 9.9 Mode 4 — Asymmetric scoring (handicap mode)

Player 1 scores at 0.8× and Player 2 at 1.0× (or any tunable ratio) to compensate for skill differences in human play. Pure UX feature; AI training stays symmetric.

**Cost.** A single scoring multiplier in the score-computation path. Negligible.

**Recommendation.** Out of scope for v1. Nice to have for casual play.

## 10. Edge cases the engine must handle

This section enumerates the edge cases the implementation must explicitly cover; each appears as a unit test in `tests/test_rules.py` (see `07_test_plan.md`).

### 10.1 Boundary lines

Runs that touch the board edge are scored normally — a 4-in-a-row along the top edge is worth the same as a 4-in-a-row in the centre. The scanner must not try to read off-board cells.

### 10.2 Lines longer than 8

A 10-in-a-row scores 5 points, the same as an 8-in-a-row, because the schedule caps at length 8.

### 10.3 Multiple maximal runs of different lengths in the same direction

Pattern `XXXX.XXXXXX` on a single row: scan in the horizontal direction. The first run starts at column 0, length 4, scores 1 point. The second run starts at column 5, length 6, scores 3 points. Total: 4 points for that row, in the horizontal direction.

### 10.4 Same player has runs in multiple directions through the same cell

A single piece may participate in up to four direction-specific runs. Each is scored independently — see Section 7.3.

### 10.5 Tie-break on a full board

When the tie-break removes a piece, that cell becomes empty. The empty cell is **not** refilled; pieces are never replaced during tie-break, only removed.

### 10.6 Tie-break draws

If after three full rounds the scores are still tied, the game ends in a draw. The draw is recorded in the move log and result screen.

### 10.7 Mode 1 — line longer than 4

In Mode 1, the win condition is **a run of length >= 4**, not exactly 4. A 5-in-row also wins (the placing player completed a 4-line as a sub-segment in the moment they placed). The implementation checks length >= 4 after each placement.

### 10.8 Player can score by playing into an opponent's neighbourhood

A run is only the placing player's own pieces; the opponent's adjacent pieces are an interruption. No scoring happens for the opponent's lines on your turn.

### 10.9 Mid-game scoring (display only)

In Mode 2, the running score is displayed throughout the game for both players. The score is recomputed (cheaply — it's incremental, only the changed neighbourhood matters) after each placement. This is presentation-only and does not affect terminal logic.

## 11. Glossary

* **Run / line / segment** — interchangeable: a maximal uninterrupted sequence of one player's pieces along one direction.
* **Maximal** — extends as far as it can without hitting an opposing piece, an empty cell, or the board edge.
* **Cell** — one (row, column) position on the N×N grid.
* **Placement** — the only regular-play action; placing one piece on one empty cell.
* **Removal** — the tie-break sub-action; removing one of the opponent's pieces.
* **Snapshot / version / level** — the frozen agent state at a checkpoint in training, presented to the player as a difficulty level.

---

*End of game description.*
