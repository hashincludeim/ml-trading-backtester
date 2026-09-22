from __future__ import annotations

from django.http import HttpRequest
from plotly.offline import get_plotlyjs_version

from stockml.viz.theme import DARK_COLOR_MAP

NAV = [
    ("overview", "dashboard:overview", "Overview"),
    ("indicators", "dashboard:indicators", "Indicators"),
    ("exploration", "dashboard:exploration", "Exploration"),
    ("models", "dashboard:models", "Models"),
    ("backtest", "dashboard:backtest", "Backtest"),
    ("multi", "dashboard:multi_ticker", "Multi-ticker"),
]
THEME_MAP = {light.lower(): dark for light, dark in DARK_COLOR_MAP.items()}


def dashboard(request: HttpRequest) -> dict[str, object]:
    return {"nav_items": NAV, "plotlyjs_version": get_plotlyjs_version(), "theme_map": THEME_MAP}
