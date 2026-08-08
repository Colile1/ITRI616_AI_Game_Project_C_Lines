"""Load and save the UI settings dict.

Settings survive across sessions so a toggle set once stays set.  Unknown keys
in the file are dropped and missing keys fall back to DEFAULT_UI_SETTINGS, so an
old settings file always loads cleanly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.config import (
    UI_SETTINGS_PATH, DEFAULT_UI_SETTINGS, AI_SPEEDS, DEFAULT_AI_SPEED,
)


def load_settings(path: Path = UI_SETTINGS_PATH) -> dict:
    settings = dict(DEFAULT_UI_SETTINGS)
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return settings
    if not isinstance(data, dict):
        return settings

    for key, default in DEFAULT_UI_SETTINGS.items():
        if key not in data:
            continue
        value = data[key]
        if isinstance(default, bool):
            if isinstance(value, bool):
                settings[key] = value
        elif key == "ai_speed":
            if value in AI_SPEEDS:
                settings[key] = value
    return settings


def save_settings(settings: dict, path: Path = UI_SETTINGS_PATH) -> bool:
    """Atomically persist settings.  False if the write failed (never raises)."""
    payload = {
        key: settings.get(key, default)
        for key, default in DEFAULT_UI_SETTINGS.items()
    }
    if payload.get("ai_speed") not in AI_SPEEDS:
        payload["ai_speed"] = DEFAULT_AI_SPEED

    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, path)
        return True
    except OSError:
        return False
