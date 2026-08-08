"""The in-game sidebar: player cards, evaluation meter, move history, actions.

Drawing only.  `layout()` returns the clickable rects and `draw()` paints them,
so the app can hit-test without duplicating any geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pygame

from src.config import (
    PLAYER_1, PLAYER_2, MODE_FIRST_TO_FOUR,
)
from src.game.analysis import algebraic
from src.ui.theme import (
    BG_MID, TEXT_BRIGHT, TEXT_MUTED, TEXT_DIM,
    P1_ACCENT, P2_ACCENT,
    font, glyph, draw_button, draw_card, draw_balance_meter, truncate,
)

DIVIDER = (50, 70, 110)

PAD = 14
CARD_H = 68
CARD_GAP = 8
METER_H = 14
BTN_H = 36
BTN_GAP = 8

# Action buttons, top to bottom.  Each carries a preferred icon and a stand-in
# for fonts that lack it — plain Arial has none of the first choices.
# (key, preferred icon, stand-in, label, needs_active_game)
ACTIONS: list[tuple[str, str, str, str, bool]] = [
    ("undo",     "↶", "←",  "Undo   (U)",   True),
    ("hint",     "◎", "○",  "Hint   (H)",   True),
    ("resign",   "✕", "×",  "Resign  (Q)",  True),
    ("settings", "⚙", "≡",  "Settings",     False),
    ("menu",     "⌂", "▲",  "Main Menu",    False),
]


def action_label(icon: str, fallback: str, text: str) -> str:
    return f"{glyph(icon, fallback)}  {text}"


def action_enabled(key: str, state: "SidebarState") -> bool:
    """Whether *key* may be clicked right now.

    Single source of truth: both the drawing and the app's hit test call this,
    so a button can never look disabled and still fire.
    """
    if state.game_over:
        return False          # the result card owns input while it is up
    if key == "undo":
        return state.can_undo
    if key == "hint":
        return state.hint_available
    return True


@dataclass
class SidebarState:
    """Everything the sidebar renders.  Assembled by the app once per frame."""

    mode: str
    board_size: int
    turn: int
    current_player: int
    p1_score: float
    p2_score: float
    p1_label: str = "Player 1"
    p2_label: str = "Player 2"
    vs_ai: bool = False
    ai_thinking: bool = False
    game_over: bool = False

    evaluation: float | None = None          # +1 = P1 winning, -1 = P2 winning
    eval_source: str = ""                    # "network" | "heuristic" | ""
    show_eval: bool = True

    moves: list[tuple[int, int, int]] = field(default_factory=list)
    ai_last_sec: float | None = None
    ai_avg_sec: float | None = None

    can_undo: bool = False
    can_redo: bool = False
    hint_available: bool = True


@dataclass
class SidebarLayout:
    rect: pygame.Rect
    buttons: dict[str, pygame.Rect]
    history_rect: pygame.Rect


# ---------------------------------------------------------------------------

def layout(area: pygame.Rect) -> SidebarLayout:
    """Compute every rect in the sidebar for the given area."""
    x = area.x + PAD
    w = area.width - 2 * PAD

    buttons: dict[str, pygame.Rect] = {}
    y = area.bottom - PAD - BTN_H
    for action in reversed(ACTIONS):
        buttons[action[0]] = pygame.Rect(x, y, w, BTN_H)
        y -= BTN_H + BTN_GAP

    header_bottom = area.y + 72 + 2 * (CARD_H + CARD_GAP) + METER_H + 46
    history_rect = pygame.Rect(x, header_bottom, w, max(0, y - header_bottom - 4))
    return SidebarLayout(rect=area, buttons=buttons, history_rect=history_rect)


def draw(surface: pygame.Surface, area: pygame.Rect, state: SidebarState) -> SidebarLayout:
    """Paint the sidebar and return its layout (for hit testing)."""
    lay = layout(area)

    surface.fill(BG_MID, area)
    pygame.draw.line(surface, DIVIDER, (area.x, area.y), (area.x, area.bottom))

    _draw_header(surface, area, state)
    _draw_player_cards(surface, area, state)
    _draw_eval(surface, area, state)
    _draw_history(surface, lay.history_rect, state)
    _draw_actions(surface, lay, state)
    return lay


# ---------------------------------------------------------------------------

def _draw_header(surface: pygame.Surface, area: pygame.Rect, state: SidebarState) -> None:
    x = area.x + PAD + 6
    surface.blit(font("display").render("C_lines", True, TEXT_BRIGHT), (x, area.y + 12))

    mode_str = "First to Four" if state.mode == MODE_FIRST_TO_FOUR else "Points Until Full"
    sub = f"{mode_str}  ·  {state.board_size}×{state.board_size}  ·  turn {state.turn}"
    surface.blit(
        font("hint").render(truncate("hint", sub, area.width - 2 * PAD), True, TEXT_DIM),
        (x, area.y + 54),
    )


def _draw_player_cards(surface: pygame.Surface, area: pygame.Rect, state: SidebarState) -> None:
    x = area.x + PAD
    w = area.width - 2 * PAD

    rows = [
        (PLAYER_1, state.p1_label, state.p1_score, P1_ACCENT),
        (PLAYER_2, state.p2_label, state.p2_score, P2_ACCENT),
    ]
    for i, (pid, label, score, col) in enumerate(rows):
        y = area.y + 76 + i * (CARD_H + CARD_GAP)
        rect = pygame.Rect(x, y, w, CARD_H)
        active = (pid == state.current_player) and not state.game_over

        draw_card(
            surface, rect,
            stripe_color=col if active else None,
            fill_alpha=55 if active else 22,
            border_alpha=130 if active else 45,
        )
        pygame.draw.circle(surface, col, (rect.x + 22, rect.centery), 9)

        name_col = TEXT_BRIGHT if active else TEXT_MUTED
        surface.blit(
            font("ui").render(truncate("ui", label, w - 130), True, name_col),
            (rect.x + 40, y + 7),
        )
        surface.blit(
            font("mono").render(f"{score:.2f} pts", True, col),
            (rect.x + 40, y + 30),
        )

        if active:
            if state.ai_thinking and pid == PLAYER_2 and state.vs_ai:
                turn_text = "Thinking…"
            else:
                turn_text = f"{glyph('▶', '»', 'hint')}  to play"
            surface.blit(font("hint").render(turn_text, True, col), (rect.x + 40, y + 50))

        # AI timing lives on the AI's card, where it belongs
        if state.vs_ai and pid == PLAYER_2 and state.ai_last_sec is not None:
            timing = f"{state.ai_last_sec:.2f}s"
            if state.ai_avg_sec is not None:
                timing += f"  (avg {state.ai_avg_sec:.2f}s)"
            ts = font("hint").render(timing, True, TEXT_DIM)
            surface.blit(ts, (rect.right - ts.get_width() - 10, y + 50))


def _draw_eval(surface: pygame.Surface, area: pygame.Rect, state: SidebarState) -> None:
    x = area.x + PAD
    w = area.width - 2 * PAD
    y = area.y + 76 + 2 * (CARD_H + CARD_GAP) + 8

    if not state.show_eval or state.evaluation is None:
        return

    caption = "Evaluation"
    if state.eval_source == "network":
        caption = "Evaluation — network"
    elif state.eval_source == "heuristic":
        caption = "Evaluation — heuristic"
    surface.blit(font("hint").render(caption, True, TEXT_DIM), (x + 2, y))

    meter = pygame.Rect(x, y + 16, w, METER_H)
    draw_balance_meter(surface, meter, state.evaluation)

    pct = int(round(abs(state.evaluation) * 100))
    if pct < 4:
        verdict, col = "level", TEXT_MUTED
    elif state.evaluation > 0:
        verdict, col = f"{state.p1_label} +{pct}", P1_ACCENT
    else:
        verdict, col = f"{state.p2_label} +{pct}", P2_ACCENT
    vs = font("hint").render(truncate("hint", verdict, w), True, col)
    surface.blit(vs, (x + (w - vs.get_width()) // 2, meter.bottom + 3))


def _draw_history(surface: pygame.Surface, rect: pygame.Rect, state: SidebarState) -> None:
    if rect.height < 40:
        return

    pygame.draw.line(surface, DIVIDER, (rect.x, rect.y), (rect.right, rect.y))
    surface.blit(font("hint").render("Move history", True, TEXT_DIM), (rect.x + 2, rect.y + 6))

    line_h = 17
    top = rect.y + 24
    capacity = max(0, (rect.height - 28) // line_h)
    if capacity == 0:
        return

    # Pair the plies into turns: "12.  D4    E5"
    pairs: list[tuple[int, str, str]] = []
    for i in range(0, len(state.moves), 2):
        first = state.moves[i]
        second = state.moves[i + 1] if i + 1 < len(state.moves) else None
        pairs.append((
            i // 2 + 1,
            algebraic(first[1], first[2]),
            algebraic(second[1], second[2]) if second else "",
        ))

    if not pairs:
        surface.blit(font("hint").render("no moves yet", True, TEXT_DIM), (rect.x + 8, top))
        return

    visible = pairs[-capacity:]          # always show the latest
    f = font("mono_sm")
    for i, (num, a, b) in enumerate(visible):
        y = top + i * line_h
        surface.blit(f.render(f"{num:>3}.", True, TEXT_DIM), (rect.x + 4, y))
        surface.blit(f.render(a, True, P1_ACCENT), (rect.x + 42, y))
        if b:
            surface.blit(f.render(b, True, P2_ACCENT), (rect.x + 96, y))

    if len(pairs) > capacity:
        hidden = font("hint").render(f"+{len(pairs) - capacity} earlier", True, TEXT_DIM)
        surface.blit(hidden, (rect.right - hidden.get_width() - 4, rect.y + 6))


def _draw_actions(surface: pygame.Surface, lay: SidebarLayout, state: SidebarState) -> None:
    mx, my = pygame.mouse.get_pos()
    for key, icon, icon_alt, text, _ in ACTIONS:
        rect = lay.buttons[key]
        draw_button(
            surface, rect, action_label(icon, icon_alt, text),
            hovered=rect.collidepoint(mx, my),
            base_surf=surface,
            enabled=action_enabled(key, state),
        )

    hint_y = lay.buttons["undo"].y - 22
    hint = font("hint").render("← → ↑ ↓ + Enter to play  ·  F1 help", True, TEXT_DIM)
    surface.blit(hint, (lay.rect.centerx - hint.get_width() // 2, hint_y))
    pygame.draw.line(
        surface, DIVIDER,
        (lay.rect.x + PAD, hint_y - 8), (lay.rect.right - PAD, hint_y - 8),
    )
