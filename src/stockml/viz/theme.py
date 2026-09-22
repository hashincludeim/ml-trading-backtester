"""Shared visual language for every chart: palette, fonts, hover, margins.

Categorical colours follow a fixed, colour-blind-validated order and are bound to *model
names*, so a model keeps its colour on every page regardless of which other models are shown.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from stockml.models.registry import MODEL_REGISTRY

FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8984"
GRID = "#e6e5e0"
AXIS_LINE = "#c9c8c2"

# Categorical slots, in validated order (identity only, never rank).
CATEGORICAL: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)

MODEL_COLORS: dict[str, str] = {
    name: CATEGORICAL[i % len(CATEGORICAL)] for i, name in enumerate(MODEL_REGISTRY)
}
BENCHMARK_COLOR = "#52514e"  # buy-and-hold / baselines: neutral ink, dashed
BENCHMARK_DASH = "dash"

UP_COLOR = "#1baf7a"
DOWN_COLOR = "#e34948"
PRICE_COLOR = "#2a78d6"
VOLUME_COLOR = "#b7d3f6"
BAND_FILL = "rgba(42, 120, 214, 0.08)"
REFERENCE_LINE = "#8a8984"

SEQUENTIAL_BLUE: list[tuple[float, str]] = [
    (0.0, "#f3f8fe"),
    (0.25, "#b7d3f6"),
    (0.5, "#6da7ec"),
    (0.75, "#256abf"),
    (1.0, "#0d366b"),
]
DIVERGING_BLUE_RED: list[tuple[float, str]] = [
    (0.0, "#184f95"),
    (0.25, "#6da7ec"),
    (0.5, "#f0efec"),
    (0.75, "#ee8a86"),
    (1.0, "#a8262a"),
]

TEMPLATE_NAME = "stockml"


def model_color(name: str) -> str:
    """Stable colour for a model name (benchmark grey for unknown names)."""
    return MODEL_COLORS.get(name, BENCHMARK_COLOR)


def _build_template() -> go.layout.Template:
    axis = {
        "showgrid": True,
        "gridcolor": GRID,
        "gridwidth": 1,
        "zeroline": False,
        "showline": True,
        "linecolor": AXIS_LINE,
        "ticks": "outside",
        "tickcolor": AXIS_LINE,
        "ticklen": 4,
        "automargin": True,
        "title": {"font": {"size": 12, "color": TEXT_SECONDARY}, "standoff": 8},
        "tickfont": {"size": 11, "color": TEXT_SECONDARY},
    }
    return go.layout.Template(
        layout={
            "font": {"family": FONT_FAMILY, "size": 12, "color": TEXT_PRIMARY},
            "title": {
                "font": {"size": 16, "color": TEXT_PRIMARY},
                "x": 0.0,
                "xanchor": "left",
                "xref": "paper",
                "y": 1.0,
                "yref": "container",
                "yanchor": "top",
                "pad": {"t": 16},
            },
            "paper_bgcolor": SURFACE,
            "plot_bgcolor": SURFACE,
            "colorway": list(CATEGORICAL),
            "margin": {"l": 64, "r": 24, "t": 92, "b": 48},
            "hovermode": "x unified",
            "hoverlabel": {
                "bgcolor": "#ffffff",
                "bordercolor": AXIS_LINE,
                "font": {"family": FONT_FAMILY, "size": 12, "color": TEXT_PRIMARY},
            },
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.0,
                "xanchor": "left",
                "x": 0.0,
                "font": {"size": 11, "color": TEXT_SECONDARY},
                "bgcolor": "rgba(0,0,0,0)",
            },
            "xaxis": axis,
            "yaxis": axis,
        }
    )


pio.templates[TEMPLATE_NAME] = _build_template()


def apply_theme(
    fig: go.Figure,
    title: str,
    x_title: str | None = None,
    y_title: str | None = None,
    height: int = 420,
) -> go.Figure:
    """Apply the shared template, title and axis labels to ``fig`` and return it.

    Args:
        fig: Figure to style (modified in place and returned for chaining).
        title: Chart title.
        x_title: Label for the first x-axis, including units.
        y_title: Label for the first y-axis, including units.
        height: Figure height in pixels.
    """
    fig.update_layout(template=TEMPLATE_NAME, title_text=title, height=height)
    if x_title is not None:
        fig.update_layout(xaxis_title_text=x_title)
    if y_title is not None:
        fig.update_layout(yaxis_title_text=y_title)
    return fig
