"""Render ``video/explainer.html`` to an MP4 and a poster image, and copy both into the site.

Needs ``video/build/`` from ``export_data.py`` and ``capture_pages.py``, and the ``video``
extra (``pip install -e ".[video]"``)::

    python video/render_video.py                    # full render, about 3 minutes
    python video/render_video.py --stills 4 9 17.5  # a few frames as PNGs, to check a layout

The master copy goes to ``video/out/`` (git-ignored) and a copy for the About page to
``web/dashboard/static/dashboard/video/``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import Browser, Page, Playwright, sync_playwright

VIDEO_DIR = Path(__file__).resolve().parent
OUT = VIDEO_DIR / "out"
STATIC = VIDEO_DIR.parent / "web" / "dashboard" / "static" / "dashboard" / "video"
NAME = "stockml-explainer"
WIDTH, HEIGHT = 1920, 1080
POSTER_AT = 4.6  # seconds in: the title card with the question, logo and "up or down" fork
CRF = 20  # x264 quality: lower is sharper and larger; 20 keeps the text crisp at ~1 MB/min


def open_animation(p: Playwright) -> tuple[Browser, Page, float, int]:
    """Load the animation in headless Chrome and wait until fonts, data and images are ready."""
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=1)
    page.goto((VIDEO_DIR / "explainer.html").as_uri() + "?render")
    page.wait_for_function("window.STOCKML !== undefined")
    page.evaluate("window.STOCKML.ready")  # evaluate awaits the promise
    duration, fps = page.evaluate("[window.STOCKML.DURATION, window.STOCKML.FPS]")
    return browser, page, float(duration), int(fps)


def frame(page: Page, t: float, kind: str = "png") -> bytes:
    """Draw the animation at ``t`` seconds and screenshot it."""
    page.evaluate("t => window.STOCKML.renderAt(t)", t)
    if kind == "jpeg":
        return page.screenshot(type="jpeg", quality=90)
    return page.screenshot(type="png")


def encode(page: Page, duration: float, fps: int, target: Path) -> None:
    """Pipe every frame into ffmpeg as PNG and encode H.264 in BT.709 for web players."""
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "image2pipe",
        "-framerate",
        str(fps),
        "-c:v",
        "png",
        "-i",
        "-",
        "-vf",
        "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,"
        "setparams=range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-tune",
        "animation",
        "-crf",
        str(CRF),
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-color_range",
        "tv",
        "-movflags",
        "+faststart",
        str(target),
    ]
    n_frames = round(duration * fps)
    started = time.monotonic()
    with subprocess.Popen(command, stdin=subprocess.PIPE) as ffmpeg:
        assert ffmpeg.stdin is not None
        for i in range(n_frames + 1):
            ffmpeg.stdin.write(frame(page, i / fps))
            if i % (fps * 5) == 0:
                elapsed = time.monotonic() - started
                print(f"  {i / fps:5.1f}s of {duration:.1f}s rendered ({elapsed:.0f}s elapsed)")
        ffmpeg.stdin.close()
        if ffmpeg.wait() != 0:
            raise SystemExit("ffmpeg failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stills", nargs="+", type=float, help="Only save frames at these times")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser, page, duration, fps = open_animation(p)
        if args.stills:
            stills = OUT / "stills"
            stills.mkdir(exist_ok=True)
            for t in args.stills:
                (stills / f"t{t:05.1f}.png").write_bytes(frame(page, t))
            print(f"Saved {len(args.stills)} stills to {stills}")
            browser.close()
            return
        mp4, poster = OUT / f"{NAME}.mp4", OUT / f"{NAME}-poster.jpg"
        print(f"Rendering {duration:.1f}s at {fps} fps to {mp4}")
        encode(page, duration, fps, mp4)
        poster.write_bytes(frame(page, POSTER_AT, "jpeg"))
        browser.close()
    STATIC.mkdir(parents=True, exist_ok=True)
    for source in (mp4, poster):
        shutil.copy2(source, STATIC / source.name)
    print(f"Done: {mp4} ({mp4.stat().st_size / 1e6:.1f} MB), copied to {STATIC}")


if __name__ == "__main__":
    main()
