"""Figures for the README, drawn from the demo figures with the notebook's own
functions.

    uv run python tools/readme_figures.py

Writes docs/img/: tango.gif (the tango, looping), still_<demo>.png (one
finished frame per demo), how.png (cut-out, mask, what the text avoids, the
page).
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runaround import (
    PAPER,
    ROOT,
    Look,
    Page,
    Place,
    Runaround,
    Type,
    asset_text,
    figure_on_page,
    font_for,
    layout,
    load_demo,
    preview,
    render_frame,
)

OUT = ROOT / "docs" / "img"
TEXTS = {"tango": "love.txt", "spring": "spring.txt", "fox": "spring.txt"}
STILLS = {"tango": 93, "spring": 60, "fox": 52}


def frame(name, k, scale, previous=None, look=None, seed=None):
    info, frames = load_demo(name)
    page, typ, ra = Page(), Type(), Runaround()
    alpha, fig = figure_on_page(frames[k], page, Place(**info["place"]))
    text = asset_text(f"texts/{TEXTS[name]}")
    lay, blocked = layout(text, alpha, page, typ, ra, previous)
    lk = look or Look(**info["look"])
    img = render_frame(
        lay,
        page,
        typ,
        lk,
        figure=fig,
        rng=np.random.default_rng(k if seed is None else seed),
        scale=scale,
        region=alpha,
    )
    return Image.fromarray(img), lay, blocked, alpha, fig


def gif(name="tango", scale=0.32, first=0, count=72, fps=12):
    """`count` frames of a demo as a looping GIF, with ffmpeg's palette. The
    grain stays still from frame to frame here: moving grain does not compress."""
    with tempfile.TemporaryDirectory() as tmp:
        previous = None
        for i, k in enumerate(range(first, first + count)):
            _info, frames = load_demo(name)
            img, _, previous, _, _ = frame(name, k % len(frames), scale, previous, seed=0)
            img.save(Path(tmp) / f"{i:04d}.png")
        src = str(Path(tmp) / "%04d.png")
        out = OUT / f"{name}.gif"
        graph = "split[a][b];[a]palettegen=max_colors=48:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
        cmd = ["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", src]
        subprocess.run([*cmd, "-filter_complex", graph, "-loop", "0", str(out)], check=True)
    return out


def how(name="tango", k=93, scale=0.3):
    """Four panels: the cut-out, its mask, what the text avoids, the page."""
    img, lay, blocked, alpha, fig = frame(name, k, scale)
    page, typ = Page(), Type()
    W, H = round(page.width * scale), round(page.height * scale)
    cut = Image.new("RGBA", fig.size, PAPER["bg"])
    cut.alpha_composite(fig)
    mask = Image.fromarray((alpha * 255).astype(np.uint8)).convert("RGB")
    avoid = preview(lay, page, typ, blocked=blocked, scale=scale)
    panels = [cut.convert("RGB").resize((W, H)), mask.resize((W, H)), avoid, img]
    labels = ["the cut-out", "its mask", "what the text avoids", "the page"]
    pad, lab = 16, 36
    sheet = Image.new("RGB", (4 * W + 3 * pad, H + lab), "white")
    d = ImageDraw.Draw(sheet)
    f = font_for("IBMPlexMono-Regular.ttf", 20)
    for i, (p, t) in enumerate(zip(panels, labels)):
        x = i * (W + pad)
        d.text((x + 2, 6), t, font=f, fill=PAPER["ink"])
        sheet.paste(p, (x, lab))
    out = OUT / "how.png"
    sheet.save(out)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, k in STILLS.items():
        img = frame(name, k, 0.5)[0]
        img.save(OUT / f"still_{name}.png", optimize=True)
        print(OUT / f"still_{name}.png")
    print(how())
    print(gif())


if __name__ == "__main__":
    main()
