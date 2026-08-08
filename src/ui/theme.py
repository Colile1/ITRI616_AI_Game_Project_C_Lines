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
HL_HINT = (185, 150, 255)
HL_CURSOR = (235, 245, 255)
HL_WIN_LINE = (255, 245, 190)

WIN_GLOW = (110, 240, 170)
DRAW_GLOW = (200, 200, 215)

PLAYER_ACCENT = {1: P1_ACCENT, 2: P2_ACCENT}
PLAYER_GLOW = {1: P1_GLOW, 2: P2_GLOW}

# Difficulty-band colours, shared by the level select and the stats screen
BAND_COLORS: dict[str, tuple[int, int, int]] = {
    "novice": (100, 200, 140),
    "easy":   (100, 180, 255),
    "medium": (200, 180,  80),
    "hard":   (220, 100,  80),
    "master": (200,  80, 220),
}


# ---------------------------------------------------------------------------
# Fonts (lazy-loaded; call init_fonts() once after pygame.init())
# ---------------------------------------------------------------------------
_fonts: dict[str, pygame.font.Font] = {}


def init_fonts() -> None:
    _glyph_cache.clear()
    _fonts["display"] = pygame.font.SysFont("Arial", 36, bold=True)
    _fonts["h2"]      = pygame.font.SysFont("Arial", 22, bold=True)
    _fonts["h3"]      = pygame.font.SysFont("Arial", 16, bold=True)
    _fonts["ui"]      = pygame.font.SysFont("Arial", 18)
    _fonts["score"]   = pygame.font.SysFont("Courier New", 20, bold=True)
    _fonts["mono"]    = pygame.font.SysFont("Courier New", 14)
    _fonts["mono_sm"] = pygame.font.SysFont("Courier New", 12)
    _fonts["small"]   = pygame.font.SysFont("Arial", 13)
    _fonts["hint"]    = pygame.font.SysFont("Arial", 12)
    _fonts["coord"]   = pygame.font.SysFont("Arial", 11)


def font(key: str) -> pygame.font.Font:
    return _fonts.get(key, _fonts.get("ui"))


# Decorative symbols are not guaranteed to exist in whatever the system resolves
# "Arial" to; an unsupported one renders as a tofu box.  `Font.metrics()` is no
# help — it happily reports the metrics of that box — so the only reliable probe
# is to render the character and compare it against a codepoint no font defines.
_glyph_cache: dict[tuple[str, str], str] = {}
_NOTDEF = ""        # private use area: never has a real glyph


def _renders_as_notdef(f: pygame.font.Font, text: str) -> bool:
    try:
        drawn = pygame.image.tostring(f.render(text, True, (255, 255, 255)), "RGBA")
        tofu = pygame.image.tostring(f.render(_NOTDEF, True, (255, 255, 255)), "RGBA")
    except (pygame.error, ValueError):
        return True
    return drawn == tofu


def glyph(preferred: str, fallback: str, font_key: str = "ui") -> str:
    """*preferred* if the font can really draw it, else the plain *fallback*."""
    cache_key = (preferred, font_key)
    if cache_key not in _glyph_cache:
        f = font(font_key)
        ok = f is not None and not any(_renders_as_notdef(f, ch) for ch in preferred)
        _glyph_cache[cache_key] = preferred if ok else fallback
    return _glyph_cache[cache_key]


def text_width(key: str, text: str) -> int:
    f = font(key)
    return f.size(text)[0] if f else 0


def truncate(key: str, text: str, max_px: int) -> str:
    """Shorten *text* with an ellipsis until it fits inside *max_px*."""
    if text_width(key, text) <= max_px:
        return text
    ell = "…"
    trimmed = text
    while trimmed and text_width(key, trimmed + ell) > max_px:
        trimmed = trimmed[:-1]
    return (trimmed + ell) if trimmed else ""


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
    radius_scale: float = 1.0,
) -> None:
    """Draw a player piece.

    *alpha* scales the entire piece (disc + glow); *radius_scale* scales only the
    disc, which is what the drop-in animation animates.
    """
    radius = max(2, int(cell_px * 0.35 * radius_scale))
    cell_px = max(cell_px, radius * 2 + 18)   # keep the glow inside the scratch surface
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
    enabled: bool = True,
    accent: tuple[int, int, int] | None = None,
    font_key: str = "ui",
) -> None:
    """Frosted button.  Disabled buttons keep their footprint but read as inert."""
    hovered = hovered and enabled
    alpha = 95 if hovered else (50 if enabled else 22)
    tint = (accent or (225, 238, 255)) if hovered else GLASS_TINT
    frost = make_frost_surface(
        (rect.width, rect.height),
        base_surf or surface,
        rect,
        tint=tint,
        tint_alpha=alpha,
    )
    surface.blit(frost, (rect.x, rect.y))
    if hovered:
        edge = accent or GLASS_EDGE
        pygame.draw.rect(surface, (*edge, 110), rect, 1, border_radius=8)
    if enabled:
        col = (accent or TEXT_BRIGHT) if hovered else TEXT_MUTED
    else:
        col = TEXT_DIM
    label = font(font_key).render(truncate(font_key, text, rect.width - 12), True, col)
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


# ---------------------------------------------------------------------------
# Segmented control (a row of mutually exclusive choices)
# ---------------------------------------------------------------------------

def segment_rects(rect: pygame.Rect, count: int) -> list[pygame.Rect]:
    """Split *rect* horizontally into *count* equal segments."""
    if count <= 0:
        return []
    seg_w = rect.width // count
    return [
        pygame.Rect(
            rect.x + i * seg_w,
            rect.y,
            seg_w if i < count - 1 else rect.width - seg_w * (count - 1),
            rect.height,
        )
        for i in range(count)
    ]


def draw_segmented(
    surface: pygame.Surface,
    rect: pygame.Rect,
    options: list[str],
    selected_index: int,
    hovered_index: int = -1,
    accent: tuple[int, int, int] = P1_ACCENT,
) -> list[pygame.Rect]:
    """Draw a segmented control and return the hit rect of each segment."""
    rects = segment_rects(rect, len(options))
    for i, (label, seg) in enumerate(zip(options, rects)):
        active = i == selected_index
        surf = pygame.Surface((seg.width, seg.height), pygame.SRCALPHA)
        if active:
            surf.fill((*accent, 70))
        elif i == hovered_index:
            surf.fill((*GLASS_TINT, 38))
        else:
            surf.fill((*BG_TOP, 26))
        surface.blit(surf, seg.topleft)
        col = TEXT_BRIGHT if active else TEXT_MUTED
        text = font("small").render(truncate("small", label, seg.width - 8), True, col)
        surface.blit(
            text,
            (seg.x + (seg.width - text.get_width()) // 2,
             seg.y + (seg.height - text.get_height()) // 2),
        )
    pygame.draw.rect(surface, (*GLASS_EDGE, 55), rect, 1, border_radius=6)
    return rects


# ---------------------------------------------------------------------------
# Meters
# ---------------------------------------------------------------------------

def draw_balance_meter(
    surface: pygame.Surface,
    rect: pygame.Rect,
    value: float,
    left_color: tuple[int, int, int] = P1_ACCENT,
    right_color: tuple[int, int, int] = P2_ACCENT,
) -> None:
    """A two-sided advantage bar.

    *value* runs from -1 (right side winning) to +1 (left side winning); the
    fill grows from the centre towards whoever is ahead.
    """
    value = max(-1.0, min(1.0, float(value)))
    track = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    track.fill((*BG_TOP, 70))
    surface.blit(track, rect.topleft)

    mid = rect.x + rect.width // 2
    half = rect.width // 2
    span = int(abs(value) * half)
    if span > 0:
        col = left_color if value > 0 else right_color
        fill = pygame.Surface((span, rect.height), pygame.SRCALPHA)
        fill.fill((*col, 190))
        surface.blit(fill, (mid - span if value > 0 else mid, rect.y))

    pygame.draw.line(surface, (*GLASS_EDGE, 120), (mid, rect.y), (mid, rect.bottom - 1))
    pygame.draw.rect(surface, (*GLASS_EDGE, 60), rect, 1, border_radius=4)


def draw_progress_bar(
    surface: pygame.Surface,
    rect: pygame.Rect,
    fraction: float,
    color: tuple[int, int, int] = P1_ACCENT,
) -> None:
    """A simple left-to-right fill bar; *fraction* is clamped to [0, 1]."""
    fraction = max(0.0, min(1.0, float(fraction)))
    track = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    track.fill((*BG_TOP, 70))
    surface.blit(track, rect.topleft)
    span = int(rect.width * fraction)
    if span > 0:
        fill = pygame.Surface((span, rect.height), pygame.SRCALPHA)
        fill.fill((*color, 190))
        surface.blit(fill, rect.topleft)
    pygame.draw.rect(surface, (*GLASS_EDGE, 55), rect, 1, border_radius=3)


# ---------------------------------------------------------------------------
# Misc drawing helpers
# ---------------------------------------------------------------------------

def draw_dim(surface: pygame.Surface, alpha: int = 170) -> None:
    """Darken the whole surface — the backdrop for any modal."""
    w, h = surface.get_size()
    dim = pygame.Surface((w, h), pygame.SRCALPHA)
    dim.fill((0, 0, 0, alpha))
    surface.blit(dim, (0, 0))


def blit_centered(
    surface: pygame.Surface, text: str, font_key: str,
    color: tuple[int, int, int], center_x: int, y: int,
) -> int:
    """Blit horizontally-centred text; returns its height."""
    s = font(font_key).render(text, True, color)
    surface.blit(s, (center_x - s.get_width() // 2, y))
    return s.get_height()


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_color(
    a: tuple[int, int, int], b: tuple[int, int, int], t: float
) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return (
        int(lerp(a[0], b[0], t)),
        int(lerp(a[1], b[1], t)),
        int(lerp(a[2], b[2], t)),
    )
