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
    _fonts["ui"] = pygame.font.SysFont("Arial", 18)
    _fonts["mono"] = pygame.font.SysFont("Courier New", 14)
    _fonts["small"] = pygame.font.SysFont("Arial", 13)


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

    # Try to grab and blur the background snippet
    try:
        clip_rect = pygame.Rect(rect.x, rect.y, size[0], size[1])
        clip_rect.clamp_ip(base_surface.get_rect())
        snippet = base_surface.subsurface(clip_rect).copy()
        try:
            blurred = pygame.transform.gaussian_blur(snippet, blur_radius)
        except AttributeError:
            blurred = snippet  # fallback: no blur
        out.blit(blurred, (0, 0))
    except (ValueError, pygame.error):
        out.fill((*BG_MID, 200))

    # Tint layer
    tint_surf = pygame.Surface(size, pygame.SRCALPHA)
    tint_surf.fill((*tint, tint_alpha))
    out.blit(tint_surf, (0, 0))

    # Edge
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
    radius = int(cell_px * 0.35)
    acc = PLAYER_ACCENT[player]
    glow_col = PLAYER_GLOW[player]

    tmp = pygame.Surface((cell_px, cell_px), pygame.SRCALPHA)
    cx, cy = cell_px // 2, cell_px // 2

    # Back glow
    for r in range(radius + 8, radius - 1, -2):
        a = max(0, int(40 * (1 - (r - radius) / 10)))
        pygame.draw.circle(tmp, (*glow_col, a), (cx, cy), r)

    # Solid disc
    pygame.draw.circle(tmp, (*acc, alpha), (cx, cy), radius)

    # Specular crescent
    spec_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(spec_surf, (255, 255, 255, 30), (radius, radius), radius)
    tmp.blit(spec_surf, (cx - radius, cy - radius))

    surface.blit(tmp, (center[0] - cell_px // 2, center[1] - cell_px // 2))


def draw_button(
    surface: pygame.Surface,
    rect: pygame.Rect,
    text: str,
    hovered: bool = False,
    base_surf: pygame.Surface | None = None,
) -> None:
    alpha = 80 if hovered else 50
    tint = (220, 235, 255) if hovered else GLASS_TINT
    frost = make_frost_surface(
        (rect.width, rect.height),
        base_surf or surface,
        rect,
        tint=tint,
        tint_alpha=alpha,
    )
    surface.blit(frost, (rect.x, rect.y))
    label = font("ui").render(text, True, TEXT_BRIGHT if hovered else TEXT_MUTED)
    lx = rect.x + (rect.width - label.get_width()) // 2
    ly = rect.y + (rect.height - label.get_height()) // 2
    surface.blit(label, (lx, ly))
