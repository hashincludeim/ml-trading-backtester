from __future__ import annotations

from django.http import HttpRequest
from plotly.offline import get_plotlyjs_version

NAV = [
    ("overview", "dashboard:overview", "Overview"),
    ("indicators", "dashboard:indicators", "Indicators"),
    ("exploration", "dashboard:exploration", "Exploration"),
    ("models", "dashboard:models", "Models"),
    ("backtest", "dashboard:backtest", "Backtest"),
    ("multi", "dashboard:multi_ticker", "Multi-ticker"),
]


def dashboard(request: HttpRequest) -> dict[str, object]:
    return {"nav_items": NAV, "plotlyjs_version": get_plotlyjs_version()}
