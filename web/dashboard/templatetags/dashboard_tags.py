from __future__ import annotations

from django import template
from django.utils.html import format_html
from django.utils.safestring import SafeString, mark_safe

register = template.Library()


@register.simple_tag
def plotly_chart(figure_json: str, chart_id: str, css_class: str = "") -> SafeString:
    """Render a chart container plus its figure JSON (already ``</``-escaped by the service)."""
    return format_html(
        '<div class="chart {}" data-figure="{}-data" id="{}" role="img"></div>'
        '<script type="application/json" id="{}-data">{}</script>',
        css_class,
        chart_id,
        chart_id,
        chart_id,
        mark_safe(figure_json),
    )


def _as_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


@register.filter
def pct(value: object, digits: int = 1) -> str:
    """Format a fraction as a percentage (``0.123`` -> ``12.3%``); ``–`` if not numeric."""
    v = _as_float(value)
    return "–" if v is None else f"{v * 100:.{digits}f}%"


@register.filter
def signed_pct(value: object, digits: int = 1) -> str:
    """Like :func:`pct` with an explicit sign."""
    v = _as_float(value)
    return "–" if v is None else f"{v * 100:+.{digits}f}%"


@register.filter
def num(value: object, digits: int = 3) -> str:
    """Fixed-point number; ``–`` if not numeric."""
    v = _as_float(value)
    return "–" if v is None else f"{v:.{digits}f}"
