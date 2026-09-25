"""Screenshot the running dashboard for the "explore" scene of the explainer video.

Start the site first (``python web/manage.py runserver``), then::

    python video/capture_pages.py --site http://127.0.0.1:8000

Writes PNGs and a manifest (``shots.js``) to ``video/build/shots/``. Uses the installed Google
Chrome through Playwright, so no browser download is needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import quote

from playwright.sync_api import Page, sync_playwright

from stockml.config import DataConfig

SHOTS = Path(__file__).resolve().parent / "build" / "shots"
DESKTOP = {"width": 1440, "height": 900}
PHONE = {"width": 390, "height": 844}
SCALE = 2  # device pixels per CSS pixel, so the frames stay sharp when scaled into the video
MAX_HEIGHT = 2400  # CSS px captured from the top of each page (the video scrolls within it)

# (slug, nav label, path, ticker index in the default universe or None, question on screen)
PAGES = (
    ("overview", "Overview", "/", 0, "How has this market behaved?"),
    ("indicators", "Indicators", "/indicators/", 1, "Do classic trading signals work?"),
    ("exploration", "Exploration", "/exploration/", 2, "Is there a signal to find?"),
    ("models", "Models", "/models/", 3, "Can a model beat a coin flip?"),
    ("backtest", "Backtest", "/backtest/", 0, "Would trading on it have paid?"),
    ("multi", "Multi-ticker", "/multi-ticker/", None, "Does anything work consistently?"),
)


def page_url(site: str, path: str, ticker_index: int | None) -> str:
    tickers = DataConfig().tickers
    if ticker_index is None:
        return site + path
    return f"{site}{path}?ticker={quote(tickers[ticker_index % len(tickers)])}"


def capture(page: Page, url: str, out: Path, theme: str) -> dict[str, int]:
    """Load ``url`` in ``theme``, wait for every chart to draw, and save the top of the page."""
    page.add_init_script(f"localStorage.setItem('stockml-theme', '{theme}')")
    page.goto(url, wait_until="networkidle")
    page.wait_for_function(
        "document.querySelectorAll('.chart[data-figure]:not(.is-ready)').length === 0",
        timeout=60_000,
    )
    page.wait_for_timeout(600)  # let Plotly finish its last layout pass
    width = page.viewport_size["width"] if page.viewport_size else DESKTOP["width"]
    height = min(MAX_HEIGHT, int(page.evaluate("document.documentElement.scrollHeight")))
    page.screenshot(
        path=str(out), full_page=True, clip={"x": 0, "y": 0, "width": width, "height": height}
    )
    return {"width": width * SCALE, "height": height * SCALE}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", default="http://127.0.0.1:8000", help="Running dashboard URL")
    args = parser.parse_args()
    site = args.site.rstrip("/")
    SHOTS.mkdir(parents=True, exist_ok=True)
    pages: list[dict[str, object]] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)

        def shoot(slug: str, url: str, viewport: dict[str, int], theme: str) -> dict[str, object]:
            context = browser.new_context(viewport=viewport, device_scale_factor=SCALE)
            size = capture(context.new_page(), url, SHOTS / f"{slug}.png", theme)
            context.close()
            print(f"captured {slug} ({url})")
            path = url.removeprefix(site) or "/"
            return {"file": f"build/shots/{slug}.png", "path": path, **size}

        for slug, label, path, ticker_index, question in PAGES:
            url = page_url(site, path, ticker_index)
            shot = shoot(slug, url, DESKTOP, "light")
            pages.append({"slug": slug, "label": label, "question": question, **shot})
        last_slug, _, last_path, last_index, _ = PAGES[-1]
        dark = shoot(f"{last_slug}-dark", page_url(site, last_path, last_index), DESKTOP, "dark")
        phone = shoot("phone", page_url(site, *PAGES[0][2:4]), PHONE, "light")
        browser.close()
    manifest = {"pages": pages, "dark": dark, "phone": phone}
    (SHOTS / "shots.js").write_text(
        "window.STOCKML_SHOTS = " + json.dumps(manifest, separators=(",", ":")) + ";\n"
    )
    print(f"Wrote {len(pages)} pages to {SHOTS}")


if __name__ == "__main__":
    main()
