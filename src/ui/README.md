# src/ui/

PyGame graphical interface. Requires a display (`SDL_VIDEODRIVER=dummy` for headless/training use).

| File | Class | Purpose |
|------|-------|---------|
| `app.py` | `App` | Main application loop and screen routing |
| `menus.py` | `MainMenu`, `BoardSizePicker`, `ModePicker`, `SettingsScreen` | Navigation screens |
| `level_select.py` | `LevelSelectScreen` | Scrollable snapshot cards — double-click or Play to start |
| `board_view.py` | `BoardView` | Renders the game board with piece animations |
| `training_board.py` | `TrainingBoard` | Live board display during human-assisted training |
| `replay_view.py` | `ReplayView` | Step through a saved game |
| `theme.py` | — | Colours, fonts, `draw_button()`, `draw_card()`, `draw_toggle()` |

### Launch
```powershell
python -m src.ui.app
```

### Controlling visible board sizes
In `src/config.py`: `UI_BOARD_SIZES = [8]` controls what the BoardSizePicker shows.
Add sizes back here once agents are trained for them.

### Level select interactions
- Mouse wheel or drag the right-side scrollbar to scroll snapshot cards
- Double-click any card to start playing against that agent
- Single-click the **Play** button in the card corner for the same effect
