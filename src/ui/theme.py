"""Colour tokens and frosted-glass rendering helpers."""

from __future__ import annotations
import pygame

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BG_DEEP = (10, 20, 40)
BG_MID = (18, 32, 58)
BG_TOP = (28, 48, 80)

GLASS_TINT = (200, 225, 255)
GLASS_EDGE = (255, 255, 255)
GLASS_INSIDE = (180, 210, 255)

P1_ACCENT = (127, 223, 255)
P1_GLOW = (60, 180, 255)
P2_ACCENT = (255, 184, 92)
P2_GLOW = (255, 130, 40)

TEXT_BRIGHT = (235, 242, 255)
TEXT_MUTED = (155, 175, 210)
TEXT_DIM = (100, 120, 150)

HL_LEGAL = (160, 220, 255)
HL_THREAT_OWN = (95, 220, 145)
HL_THREAT_OPP = (245, 100, 120)
HL_LAST_MOVE = (255, 220, 120)

WIN_GLOW = (110, 240, 170)
DRAW_GLOW = (200, 200, 215)

PLAYER_ACCENT = {1: P1_ACCENT, 2: P2_ACCENT}
PLAYER_GLOW = {1: P1_GLOW, 2: P2_GLOW}


# ---------------------------------------------------------------------------
# Fonts (lazy-loaded; call init_fonts() once after pygame.init())
# ---------------------------------------------------------------------------
_fonts: dict[str, pygame.font.Font] = {}


def init_fonts() -> None:
    _fonts["display"] = pygame.font.SysFont("Arial", 36, bold=True)
    _fonts["h2"]      = pygame.font.SysFont("Arial", 22, bold=True)
    _fonts["ui"]      = pygame.font.SysFont("Arial", 18)
    _fonts["score"]   = pygame.font.SysFont("Courier New", 20, bold=True)
    _fonts["mono"]    = pygame.font.SysFont("Courier New", 14)
    _fonts["small"]   = pygame.font.SysFont("Arial", 13)
    _fonts["hint"]    = pygame.font.SysFont("Arial", 12)
    _fonts["coord"]   = pygame.font.SysFont("Arial", 11)


def font(key: str) -> pygame.font.Font:
    return _fonts.get(key, _fonts.get("ui"))


# ---------------------------------------------------------------------------
# Frosted-glass panel
# ---------------------------------------------------------------------------

def make_frost_surface(
    size: tuple[int, int],
    base_surface: pygame.Surface,
    rect: pygame.Rect,
    blur_radius: int = 6,
    tint: tuple[int, int, int] = GLASS_TINT,
    tint_alpha: int = 60,
) -> pygame.Surface:
    """Render a frosted-glass rectangle.

    Clips *base_surface* at *rect*, applies a tint, draws a thin edge.
    Falls back gracefully if gaussian_blur is unavailable.
    """
    out = pygame.Surface(size, pygame.SRCALPHA)

    try:
        clip_rect = pygame.Rect(rect.x, rect.y, size[0], size[1])
        clip_rect.clamp_ip(base_surface.get_rect())
        snippet = base_surface.subsurface(clip_rect).copy()
        try:
            blurred = pygame.transform.gaussian_blur(snippet, blur_radius)
        except AttributeError:
            blurred = snippet
        out.blit(blurred, (0, 0))
    except (ValueError, pygame.error):
        out.fill((*BG_MID, 200))

    tint_surf = pygame.Surface(size, pygame.SRCALPHA)
    tint_surf.fill((*tint, tint_alpha))
    out.blit(tint_surf, (0, 0))

    pygame.draw.rect(out, (*GLASS_EDGE, 70), out.get_rect(), width=1, border_radius=8)
    return out


# ---------------------------------------------------------------------------
# Piece drawing
# ---------------------------------------------------------------------------

def draw_piece(
    surface: pygame.Surface,
    center: tuple[int, int],
    player: int,
    cell_px: int,
    alpha: int = 255,
) -> None:
    """Draw a player piece.  *alpha* scales the entire piece (disc + glow)."""
    radius = int(cell_px * 0.35)
    acc = PLAYER_ACCENT[player]
    glow_col = PLAYER_GLOW[player]
    scale = alpha / 255.0

    tmp = pygame.Surface((cell_px, cell_px), pygame.SRCALPHA)
    cx, cy = cell_px // 2, cell_px // 2

    # Back glow (scaled with alpha)
    for r in range(radius + 8, radius - 1, -2):
        a = max(0, int(40 * (1 - (r - radius) / 10) * scale))
        pygame.draw.circle(tmp, (*glow_col, a), (cx, cy), r)

    # Solid disc
    pygame.draw.circle(tmp, (*acc, alpha), (cx, cy), radius)

    # Specular crescent (scaled)
    spec_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(spec_surf, (255, 255, 255, int(30 * scale)), (radius, radius), radius)
    tmp.blit(spec_surf, (cx - radius, cy - radius))

    surface.blit(tmp, (center[0] - cell_px // 2, center[1] - cell_px // 2))


# ---------------------------------------------------------------------------
# Button
# ---------------------------------------------------------------------------

def draw_button(
    surface: pygame.Surface,
    rect: pygame.Rect,
    text: str,
    hovered: bool = False,
    base_surf: pygame.Surface | None = None,
) -> None:
    alpha = 95 if hovered else 50
    tint = (225, 238, 255) if hovered else GLASS_TINT
    frost = make_frost_surface(
        (rect.width, rect.height),
        base_surf or surface,
        rect,
        tint=tint,
        tint_alpha=alpha,
    )
    surface.blit(frost, (rect.x, rect.y))
    if hovered:
        pygame.draw.rect(surface, (*GLASS_EDGE, 90), rect, 1, border_radius=8)
    col = TEXT_BRIGHT if hovered else TEXT_MUTED
    label = font("ui").render(text, True, col)
    lx = rect.x + (rect.width - label.get_width()) // 2
    ly = rect.y + (rect.height - label.get_height()) // 2
    surface.blit(label, (lx, ly))


# ---------------------------------------------------------------------------
# Card panel
# ---------------------------------------------------------------------------

def draw_card(
    surface: pygame.Surface,
    rect: pygame.Rect,
    stripe_color: tuple[int, int, int] | None = None,
    fill_alpha: int = 30,
    border_alpha: int = 60,
    border_radius: int = 8,
) -> None:
    """Frosted-glass card with an optional left-side colour stripe."""
    card_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    card_surf.fill((*BG_TOP, fill_alpha))
    pygame.draw.rect(
        card_surf, (*GLASS_EDGE, border_alpha),
        card_surf.get_rect(), 1, border_radius=border_radius,
    )
    if stripe_color is not None:
        stripe_h = rect.height - border_radius
        stripe_surf = pygame.Surface((4, stripe_h), pygame.SRCALPHA)
        stripe_surf.fill((*stripe_color, 210))
        card_surf.blit(stripe_surf, (0, border_radius // 2))
    surface.blit(card_surf, (rect.x, rect.y))


# ---------------------------------------------------------------------------
# Toggle switch
# ---------------------------------------------------------------------------

def draw_toggle(
    surface: pygame.Surface,
    cx: int,
    cy: int,
    value: bool,
    color_on: tuple[int, int, int] = P1_ACCENT,
) -> None:
    """Pill-shaped on/off toggle centred at (cx, cy)."""
    tw, th = 40, 20
    rect = pygame.Rect(cx - tw // 2, cy - th // 2, tw, th)
    surf = pygame.Surface((tw, th), pygame.SRCALPHA)
    bg = (*color_on, 210) if value else (70, 90, 120, 210)
    pygame.draw.rect(surf, bg, surf.get_rect(), border_radius=th // 2)
    knob_x = tw - th // 2 - 2 if value else th // 2 + 2
    pygame.draw.circle(surf, (240, 248, 255, 240), (knob_x, th // 2), th // 2 - 2)
    surface.blit(surf, (rect.x, rect.y))
