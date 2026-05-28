"""Menu screens: MainMenu, BoardSizePicker, ModePicker, SettingsScreen."""

from __future__ import annotations
import pygame

from src.ui.theme import (
    BG_DEEP, BG_TOP, TEXT_BRIGHT, TEXT_MUTED, TEXT_DIM,
    P1_ACCENT, P2_ACCENT, font, make_frost_surface, draw_button,
)
from src.config import BOARD_SIZES, MODE_FIRST_TO_FOUR, MODE_POINTS_FULL


class MainMenu:
    BUTTONS = ["Play vs AI", "Hot-seat", "Settings", "Quit"]
    BTN_W, BTN_H, BTN_GAP = 260, 52, 16

    def __init__(self):
        self._hovered: int = -1

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(BG_DEEP)
        w, h = surface.get_size()

        title = font("display").render("C_lines", True, TEXT_BRIGHT)
        surface.blit(title, ((w - title.get_width()) // 2, h // 5))

        sub = font("ui").render("AI Board Game — ITRI616", True, TEXT_MUTED)
        surface.blit(sub, ((w - sub.get_width()) // 2, h // 5 + 48))

        total_h = len(self.BUTTONS) * (self.BTN_H + self.BTN_GAP)
        start_y = (h - total_h) // 2 + 40
        for i, label in enumerate(self.BUTTONS):
            x = (w - self.BTN_W) // 2
            y = start_y + i * (self.BTN_H + self.BTN_GAP)
            rect = pygame.Rect(x, y, self.BTN_W, self.BTN_H)
            draw_button(surface, rect, label, hovered=(self._hovered == i), base_surf=surface)

    def handle_event(self, event: pygame.event.Event, surface: pygame.Surface) -> str | None:
        w, h = surface.get_size()
        total_h = len(self.BUTTONS) * (self.BTN_H + self.BTN_GAP)
        start_y = (h - total_h) // 2 + 40

        if event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            self._hovered = -1
            for i, _ in enumerate(self.BUTTONS):
                x = (w - self.BTN_W) // 2
                y = start_y + i * (self.BTN_H + self.BTN_GAP)
                if pygame.Rect(x, y, self.BTN_W, self.BTN_H).collidepoint(mx, my):
                    self._hovered = i

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            for i, label in enumerate(self.BUTTONS):
                x = (w - self.BTN_W) // 2
                y = start_y + i * (self.BTN_H + self.BTN_GAP)
                if pygame.Rect(x, y, self.BTN_W, self.BTN_H).collidepoint(mx, my):
                    return label.lower().replace(" ", "_")

        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            return "quit"
        return None


class BoardSizePicker:
    CHIP_R = 36

    def __init__(self):
        self._selected: int = 10
        self._hovered: int = -1

    @property
    def selected(self) -> int:
        return self._selected

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(BG_DEEP)
        w, h = surface.get_size()
        title = font("display").render("Board Size", True, TEXT_BRIGHT)
        surface.blit(title, ((w - title.get_width()) // 2, h // 4))
        hint = font("ui").render("Click to select, Enter to confirm", True, TEXT_MUTED)
        surface.blit(hint, ((w - hint.get_width()) // 2, h // 4 + 48))

        spacing = self.CHIP_R * 3
        total_w = len(BOARD_SIZES) * spacing
        start_x = (w - total_w) // 2 + self.CHIP_R
        cy = h // 2

        for i, sz in enumerate(BOARD_SIZES):
            cx = start_x + i * spacing
            selected = sz == self._selected
            hovered = i == self._hovered
            color = P1_ACCENT if selected else (TEXT_MUTED if hovered else TEXT_DIM)
            pygame.draw.circle(surface, color, (cx, cy), self.CHIP_R, 0 if selected else 2)
            label = font("display" if selected else "ui").render(str(sz), True,
                                                                  BG_DEEP if selected else color)
            surface.blit(label, (cx - label.get_width() // 2, cy - label.get_height() // 2))

    def handle_event(self, event: pygame.event.Event, surface: pygame.Surface) -> str | None:
        w, h = surface.get_size()
        spacing = self.CHIP_R * 3
        total_w = len(BOARD_SIZES) * spacing
        start_x = (w - total_w) // 2 + self.CHIP_R
        cy = h // 2

        if event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            self._hovered = -1
            for i, _ in enumerate(BOARD_SIZES):
                cx = start_x + i * spacing
                if (mx - cx) ** 2 + (my - cy) ** 2 <= self.CHIP_R ** 2:
                    self._hovered = i

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            for i, sz in enumerate(BOARD_SIZES):
                cx = start_x + i * spacing
                if (mx - cx) ** 2 + (my - cy) ** 2 <= self.CHIP_R ** 2:
                    self._selected = sz

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                return "confirm"
            if event.key == pygame.K_ESCAPE:
                return "back"
        return None


class ModePicker:
    BTN_W, BTN_H = 300, 80

    def __init__(self):
        self._selected: str = MODE_POINTS_FULL
        self._hovered: str | None = None

    @property
    def selected(self) -> str:
        return self._selected

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(BG_DEEP)
        w, h = surface.get_size()
        title = font("display").render("Game Mode", True, TEXT_BRIGHT)
        surface.blit(title, ((w - title.get_width()) // 2, h // 4))

        modes = [
            (MODE_FIRST_TO_FOUR, "First to Four", "Win with the first 4-in-a-row"),
            (MODE_POINTS_FULL, "Points Until Full", "Score lines — most points wins"),
        ]
        gap = 24
        total_h = len(modes) * (self.BTN_H + gap)
        start_y = (h - total_h) // 2

        for i, (mode, label, desc) in enumerate(modes):
            x = (w - self.BTN_W) // 2
            y = start_y + i * (self.BTN_H + gap)
            rect = pygame.Rect(x, y, self.BTN_W, self.BTN_H)
            selected = mode == self._selected
            hovered = mode == self._hovered
            draw_button(surface, rect, label, hovered=(selected or hovered), base_surf=surface)
            sub = font("small").render(desc, True, TEXT_DIM)
            surface.blit(sub, (x + (self.BTN_W - sub.get_width()) // 2, y + self.BTN_H - 4))

    def handle_event(self, event: pygame.event.Event, surface: pygame.Surface) -> str | None:
        w, h = surface.get_size()
        modes = [MODE_FIRST_TO_FOUR, MODE_POINTS_FULL]
        gap = 24
        total_h = len(modes) * (self.BTN_H + gap)
        start_y = (h - total_h) // 2

        if event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            self._hovered = None
            for i, mode in enumerate(modes):
                x = (w - self.BTN_W) // 2
                y = start_y + i * (self.BTN_H + gap)
                if pygame.Rect(x, y, self.BTN_W, self.BTN_H).collidepoint(mx, my):
                    self._hovered = mode

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            for i, mode in enumerate(modes):
                x = (w - self.BTN_W) // 2
                y = start_y + i * (self.BTN_H + gap)
                if pygame.Rect(x, y, self.BTN_W, self.BTN_H).collidepoint(mx, my):
                    self._selected = mode
                    return "confirm"

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return "back"
        return None


class SettingsScreen:
    TOGGLES = [
        ("show_legal", "Show legal moves"),
        ("show_threats", "Show threat highlights"),
        ("reduce_motion", "Reduce motion"),
        ("sound", "Sound (off by default)"),
    ]
    ROW_H = 48

    def __init__(self, settings: dict):
        self._settings = settings

    @property
    def settings(self) -> dict:
        return self._settings

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(BG_DEEP)
        w, h = surface.get_size()
        title = font("display").render("Settings", True, TEXT_BRIGHT)
        surface.blit(title, ((w - title.get_width()) // 2, 60))
        hint = font("ui").render("Click to toggle  |  ESC = back", True, TEXT_DIM)
        surface.blit(hint, ((w - hint.get_width()) // 2, 108))

        start_y = 160
        for i, (key, label) in enumerate(self.TOGGLES):
            y = start_y + i * self.ROW_H
            val = self._settings.get(key, False)
            color = P1_ACCENT if val else TEXT_MUTED
            indicator = font("ui").render("●  " + label, True, color)
            surface.blit(indicator, (w // 4, y))
            state_str = "ON" if val else "OFF"
            state = font("ui").render(state_str, True, color)
            surface.blit(state, (w * 3 // 4 - state.get_width(), y))

        back = font("ui").render("ESC — Back", True, TEXT_DIM)
        surface.blit(back, ((w - back.get_width()) // 2, h - 60))

    def handle_event(self, event: pygame.event.Event, surface: pygame.Surface) -> str | None:
        w, h = surface.get_size()
        start_y = 160

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            for i, (key, _) in enumerate(self.TOGGLES):
                y = start_y + i * self.ROW_H
                if y <= my <= y + self.ROW_H and w // 4 <= mx <= w * 3 // 4:
                    self._settings[key] = not self._settings.get(key, False)

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return "back"
        return None
