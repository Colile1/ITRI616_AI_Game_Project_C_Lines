# C_lines — UI / UX Specification

**Author:** *----* *S----*
**Module:** ITRI 616
**Date:** 2026-05-28
**Theme:** Frosted glass · sheer · light-blue
**Status:** Visual design locked; implementation pending Phase 10

---

## 1. Design intent

The C_lines visual language is frosted-glass-on-deep-navy with light-blue accents. The intent is to evoke calm focus: dark, low-saturation background; thin, semi-transparent panels with a soft blur behind them; light-cyan glass-orb pieces for Player 1; warm-amber pieces for Player 2 for high contrast against the blue field. Visual hierarchy comes from layering, transparency, and a single accent colour — not from drop-shadows, bevels, or noisy textures.

The frosted-glass effect is achieved in PyGame by pre-rendering a Gaussian-blurred copy of the background and clipping it under each panel rectangle, then compositing a 20–30% white-alpha tint over the blur and stroking the panel edge at 30% white alpha to suggest the glass rim. This is the "backdrop-filter: blur" pattern translated into PyGame; it is cheap because the blurred backdrop is rendered once per screen and cached.

## 2. Colour tokens

All colours live in `src/ui/theme.py` as named constants. No hard-coded RGB tuples anywhere else.

```python
# Background field
BG_DEEP        = (10, 20, 40)       # darkest navy — base canvas
BG_MID         = (18, 32, 58)       # mid navy — radial-gradient centre
BG_TOP         = (28, 48, 80)       # lightest navy — sky-tinted vignette

# Frosted-glass panel
GLASS_TINT     = (200, 225, 255)    # warm white-blue tint; use at alpha 50..80
GLASS_EDGE     = (255, 255, 255)    # panel stroke; use at alpha 70..90 (1px)
GLASS_INSIDE   = (180, 210, 255)    # inner highlight on top edge; alpha 40

# Player accents
P1_ACCENT      = (127, 223, 255)    # light cyan — Player 1
P1_GLOW        = ( 60, 180, 255)    # deeper blue — P1 glow ring
P2_ACCENT      = (255, 184,  92)    # warm amber — Player 2
P2_GLOW        = (255, 130,  40)    # deeper amber — P2 glow ring

# Text
TEXT_BRIGHT    = (235, 242, 255)    # primary labels
TEXT_MUTED     = (155, 175, 210)    # secondary labels / hints
TEXT_DIM       = (100, 120, 150)    # tertiary / disabled

# State accents
HL_LEGAL       = (160, 220, 255)    # subtle highlight on legal cells (hover)
HL_THREAT_OWN  = ( 95, 220, 145)    # green — your 4-line completion cell
HL_THREAT_OPP  = (245, 100, 120)    # red   — opponent's 4-line completion cell
HL_LAST_MOVE   = (255, 220, 120)    # gold ring around the most-recent placement

# Functional
WIN_GLOW       = (110, 240, 170)    # full-board win-line halo
DRAW_GLOW      = (200, 200, 215)    # full-board draw halo
```

## 3. Typography

```
Display font   : Inter / system sans (bold, 36 px)
UI font        : Inter / system sans (regular, 18 px)
Mono font      : JetBrains Mono / system mono (regular, 14 px) — for score numbers
Fallback chain : Pygame default if Inter not present
```

All UI numbers (scores, turn count, snapshot generation IDs) render in the mono font for stability of layout when digits change.

## 4. Window layout

The window is divided into a board area (left) and an information sidebar (right). Sizes scale with the chosen board.

```
┌─────────────────────────────────────────┬────────────────────────────┐
│                                         │   C_LINES                  │
│                                         │   ─────────                │
│                                         │   Mode  : Points-Full      │
│                                         │   Board : 12 × 12          │
│         ┌──────────────────┐            │   Turn  : 47               │
│         │                  │            │                            │
│         │      board       │            │   ── Players ──            │
│         │   N × N cells    │            │   P1  (you)  · ● 12.5      │
│         │                  │            │   P2  (AI gen_007) · 11.0  │
│         │                  │            │                            │
│         └──────────────────┘            │   ── Last move ──          │
│                                         │   P2 placed at G7          │
│                                         │                            │
│                                         │   ── Controls ──           │
│                                         │   Click       Place        │
│                                         │   H           Hint         │
│                                         │   R           Replay last  │
│                                         │   ESC         Menu         │
│                                         │   Q           Resign       │
└─────────────────────────────────────────┴────────────────────────────┘
```

The board is centred in its area and never touches the window edge — there is at least 24 px breathing room on every side. The sidebar is a single frosted-glass panel; section dividers inside it are thin 1-px `GLASS_EDGE` lines at low alpha.

### 4.1 Window sizing

```
cell_px        = 56   # pixel size of one board cell
board_margin   = 24   # px padding around the board grid
sidebar_w      = 320  # px
window_w       = N * cell_px + 2 * board_margin + sidebar_w
window_h       = N * cell_px + 2 * board_margin
```

For N=12: window is `12*56 + 48 + 320 = 1040` px wide, `12*56 + 48 = 720` px tall — fits comfortably on a 1366×768 student-laptop screen with margin to spare.

### 4.2 Resize behaviour

When the player changes board size in the menu, the window resizes to the formula above. Resizing during a game is disabled.

## 5. Screens

The application is a linear state machine of screens. Transitions are listed in Section 6.

### 5.1 Main menu

```
                C_LINES
                ───────
              [ Play vs AI    ]
              [ Hot-seat      ]
              [ Settings      ]
              [ Quit          ]

         a light-blue line-art motif in the corner
```

A single centred column of buttons, each a frosted-glass panel. Hovered buttons brighten their tint by ~15% and gain a 1px inner glow.

### 5.2 Board-size picker

After selecting a play mode, the player picks a board size from {8, 9, 10, 11, 12}. UI: five circular frosted chips arranged horizontally, with the selected size highlighted. Enter confirms; left/right arrows or click to change.

### 5.3 Mode picker

A two-tile horizontal layout: "First to Four" on the left, "Points Until Full" on the right. Each tile shows a one-line summary of the mode. Selected tile gets the P1-accent stroke.

### 5.4 Level select (single-player only)

A grid of cards, one per available snapshot for the chosen board size. Each card shows:

```
┌─────────────────────────┐
│  gen_007                │
│  ─────────              │
│  Games trained : 7,000  │
│  WR vs random  : 92.5%  │
│  WR vs heur.   : 64.1%  │
│  Elo           : 1180   │
│  Trained       : 2026-05-26 │
│                         │
│  [ Play vs this level ] │
└─────────────────────────┘
```

Cards are tinted faintly by difficulty: lighter blue = weaker, deeper blue = stronger. The newest snapshot is flagged "★ Latest". Cards scroll horizontally if more than fit on screen.

### 5.5 In-game

The layout in Section 4. The board cells render as:

* Empty cell — a thin frosted-glass square (subtle stroke, near-transparent fill).
* P1 piece — a light-cyan glass orb with a soft inner glow.
* P2 piece — a warm-amber glass orb with a soft inner glow.
* Hovered legal cell — `HL_LEGAL` border (1 px) appears.
* Threat cells (assist mode) — `HL_THREAT_OWN` / `HL_THREAT_OPP` border.
* Last placed cell — gold ring (`HL_LAST_MOVE`) for one full turn.
* Lines counted toward score (Mode 2) — thin pulse along the line direction at score time.

### 5.6 Game over

A central frosted card overlays the dimmed board:

```
            P1 WINS
            ───────
       Final: P1 14.25  P2 11.50

   Lines: P1 had 1 × 5-line, 2 × 4-line, 2 × 3-line
          P2 had 0 × 5-line, 2 × 4-line, 4 × 3-line

       [ Replay ]   [ Rematch ]   [ Menu ]
```

For tie-break-resolved Mode 2 games, the "Final" line annotates "(after N tie-break rounds)".

### 5.7 Replay viewer

The game-over screen's [Replay] button opens a stepper: arrow keys advance/retreat one move at a time, the move number and player are shown in the sidebar, scoring deltas are surfaced as each line is formed.

### 5.8 Settings

A short list of toggles, each a frosted card:

```
 [ ] Show threat highlights (assist)
 [ ] Show legal-move highlights (assist)
 [ ] Music
 [x] Sound effects
 [ ] Enable swap rule (for hot-seat only)
 [ ] Enable opening restriction
```

## 6. Screen transition graph

```
   Main menu
   ┌───────────────┬─────────────────┬──────────┐
   │               │                 │          │
   v               v                 v          v
 Play vs AI    Hot-seat           Settings    Quit
   │               │                            │
   v               v                            v
 Board-size    Board-size                    (Main)
   │               │
   v               v
 Mode picker   Mode picker
   │               │
   v               v
 Level select   In-game (hot-seat)
   │               │
   v               v
 In-game (vs AI)   Game over
   │               │
   v               v
 Game over     (Replay / Rematch / Menu)
```

ESC during any screen returns to the previous screen; from in-game it pauses and offers Resign / Continue.

## 7. Frosted-glass rendering — implementation recipe

```python
# theme.py
def make_frost_surface(size, base_surface, blur_radius=10, tint=GLASS_TINT, tint_alpha=60):
    """Return a Surface that visually frosts the underlying base_surface."""
    w, h = size
    snippet = base_surface.copy()
    snippet = snippet.subsurface(snippet.get_rect().fit(snippet.get_rect()))
    blurred = pygame.transform.gaussian_blur(snippet, blur_radius)
    tint_surf = pygame.Surface((w, h), pygame.SRCALPHA)
    tint_surf.fill((*tint, tint_alpha))
    out = pygame.Surface((w, h), pygame.SRCALPHA)
    out.blit(blurred, (0, 0))
    out.blit(tint_surf, (0, 0))
    pygame.draw.rect(out, (*GLASS_EDGE, 70), out.get_rect(), width=1)
    return out
```

* `gaussian_blur` is in PyGame-CE 2.5+. On classic PyGame, fall back to downsample-by-4 then upsample, which produces a similar (cheaper) blur.
* Cache the result of `make_frost_surface` per panel rectangle per background — the background only changes between screens, so cache keys are stable.
* Re-render the panel only when its size or position changes, not every frame.

## 8. Piece rendering

A piece is a centred filled circle of radius `cell_px * 0.35`, with three composited layers:

1. A back glow — radial alpha falloff from `P1_GLOW`/`P2_GLOW` at the centre to transparent at radius `cell_px * 0.50`. Slight blur. Gives the piece its "lit-from-within" feel against the frosted board.
2. A solid disc in the accent colour.
3. A small white highlight crescent in the upper-left at ~12% alpha — the "glass" specular.

Animation: pieces fade in over 120 ms when placed (alpha 0 -> 1) and ride a 4-px-down-to-rest spring so they feel like they "settle" onto the board. This is the only animation in the game and is cheap.

## 9. Accessibility

* Colours pass WCAG AA contrast against the navy background (light cyan and warm amber are at +4.5:1 minimum).
* Player accents are red/blue *and* distinct shapes if a colour-blind mode is enabled (P1 = circle, P2 = octagon with the same fill colour). Toggle in Settings.
* All controls have a keyboard binding; mouse is not strictly required.
* Font sizes are scaled by a `UI_SCALE` constant in `theme.py` so users on 4K displays can boost everything by 1.5×.
* Animations can be disabled in Settings ("Reduce motion") — pieces appear instantly.

## 10. Sound (optional, off by default)

A minimal soundscape: one soft "click" on placement (P1 slightly higher pitch than P2), one chime on a 4-line completion, one swell on game-over. All sounds are short (<300 ms), 16-bit WAV, room for licence-free creation.

## 11. Asset list

The UI ships with:

* `assets/fonts/Inter-Regular.ttf`, `Inter-Bold.ttf`
* `assets/fonts/JetBrainsMono-Regular.ttf`
* `assets/icons/` — small SVGs converted to PNG for menu chrome (gear, back-arrow)
* Optional `assets/sounds/place_p1.wav`, `place_p2.wav`, `line.wav`, `gameover.wav`

If the fonts are not present, the UI falls back gracefully to `pygame.font.SysFont(None, size)`. The application must run with assets/ empty.

## 12. Out of scope

Animated backgrounds, particle systems, parallax, full audio mix with music. Online multiplayer chrome. Mobile-touch input modes. Theme switching at runtime (a "Dark / Light" toggle is intentionally not present — the game is dark-only by design).

---

*End of UI/UX specification.*
