"""Light demo figures for the public notebook, from the private masks.

    uv run python tools/make_demo.py            # frames, window and loop
    uv run python tools/make_demo.py --loops    # window and loop only
    uv run python tools/make_demo.py --examples # example videos for cutout.py

Only pieces we may publish: the tango (The Gallopin' Gaucho, 1928, US public
domain), Spring (Blender Studio, CC-BY 4.0) and our own fox. Each becomes
demo/<name>/: cut-outs as lossy WebP (RGBA), the masks the text avoids as PNG,
every `STEP`-th frame, `WIDTH` px wide, and demo.json with the piece's place,
look and credit. A few MB in all.

Each demo also gets loop.mp4, the playground's loop with the notebook's
default settings (drawn by the notebook's own make_loop, so it is exactly what
"make the loop" would draw), and in demo.json `window`: the first of the
`Notebook.window` frames the sections study, the stretch where the text moves
most.
"""

import itertools
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from runaround import (
    MASKS,
    ROOT,
    Look,
    Notebook,
    Page,
    Place,
    Runaround,
    Type,
    asset_text,
    figure_on_page,
    layouts,
    load_demo,
    load_pieces,
    make_loop,
    moved,
    piece_params,
    resize_rgba,
)

DEMO = ROOT / "demo"
WIDTH = 640
STEP = 2  # 24 fps -> 12 fps
QUALITY = 82
PUBLIC = {
    "tango": "The Gallopin' Gaucho (1928), US public domain",
    "spring": "Spring, Blender Studio (2019), CC-BY 4.0, studio.blender.org",
    "fox": "Our own drawing (figures/fox), CC-BY 4.0",
}


# Short videos cutout.py starts from, cut from the same public pieces: the
# piece's own shot, H.264 at most `EXAMPLE_WIDTH` px wide, no sound.
# One shot each: ffmpeg's cut finder misses the tango's cuts (101.52, 103.02 s),
# so its example stops before them.
EXAMPLES = {"spring": None, "tango": 101.4}
EXAMPLE_WIDTH = 1280
EXAMPLE_CRF = 23


def examples():
    """demo/examples/<slug>.mp4 for each of EXAMPLES."""
    pieces = load_pieces()
    out = DEMO / "examples"
    out.mkdir(parents=True, exist_ok=True)
    for slug, stop in EXAMPLES.items():
        piece = pieces[slug]
        path = out / f"{slug}.mp4"
        end = stop or piece["end"]
        cmd = ["ffmpeg", "-v", "error", "-y", "-ss", str(piece["start"]), "-to", str(end)]
        cmd += ["-i", str(ROOT / "clips" / piece["clip"]), "-an"]
        cmd += ["-vf", f"scale='min({EXAMPLE_WIDTH},iw)':-2", "-c:v", "libx264", "-preset", "slow"]
        cmd += [
            "-crf",
            str(EXAMPLE_CRF),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        subprocess.run(cmd, check=True)
        print(f"{slug}: {path.stat().st_size / 1e6:.1f} MB, {PUBLIC[slug]}")


def loops_only():
    """Only the window and the loop of each demo, from the frames already
    there: after a change to the notebook's defaults or drawing."""
    for slug in PUBLIC:
        out = DEMO / slug
        info = json.loads((out / "demo.json").read_text())
        info["window"] = busiest_window(slug, info)
        (out / "demo.json").write_text(json.dumps(info, indent=2))
        write_loop(slug, info, out / "loop.mp4")
        mb = (out / "loop.mp4").stat().st_size / 1e6
        print(f"{slug}: window from frame {info['window']}, loop {mb:.1f} MB")


def main():
    if "--loops" in sys.argv:
        return loops_only()
    if "--examples" in sys.argv:
        return examples()
    pieces = load_pieces()
    for slug, credit in PUBLIC.items():
        src = MASKS / slug
        masks = sorted(src.glob("[0-9]*.png"))[::STEP]
        if not masks:
            raise SystemExit(f"no masks for {slug}: run masks.py first")
        out = DEMO / slug
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        for k, m in enumerate(masks):
            rgba = Image.open(src / "rgba" / m.name)
            h = round(rgba.height * WIDTH / rgba.width)
            resize_rgba(rgba, (WIDTH, h)).save(
                out / f"{k:04d}.webp", quality=QUALITY, exact=True
            )  # exact: keep the colour under transparent pixels ("ink" figures read it)
            Image.open(m).convert("L").resize((WIDTH, h), Image.BILINEAR).save(out / f"{k:04d}.png")
        p = piece_params(pieces[slug])
        info = dict(
            title=pieces[slug]["title"],
            credit=credit,
            fps=p["reel"].fps / STEP,
            place=asdict(p["place"]),
            look=asdict(p["look"]),
            text=pieces[slug]["text"],
            frames=len(masks),
        )
        info["window"] = busiest_window(slug, info)
        (out / "demo.json").write_text(json.dumps(info, indent=2))
        write_loop(slug, info, out / "loop.mp4")
        size = sum(f.stat().st_size for f in out.iterdir()) / 1e6
        print(f"{slug}: {len(masks)} frames, {size:.1f} MB")
    # the browser cannot list a folder: the notebook reads the demos from here
    (DEMO / "index.json").write_text(json.dumps(sorted(PUBLIC)))


def defaults(info):
    """What the notebook starts with for a demo: its text, the default page,
    type and runaround, the demo's look."""
    text = asset_text(f"texts/{info['text']}")
    return text, Page(), Type(), Runaround(), Look(**info["look"])


def busiest_window(slug, info):
    """The first frame of the `Notebook.window` frames in which the most
    text changes line, with the notebook's default settings."""
    load_demo.cache_clear()
    _, frames = load_demo(slug)
    text, page, typ, ra, _ = defaults(info)
    place = Place(**info["place"])
    alphas = [figure_on_page(frames[k], page, place)[0] for k in range(len(frames))]
    lays = layouts(text, alphas, page, typ, ra)
    share = np.array([moved(a, b) for a, b in itertools.pairwise(lays)])
    w = min(Notebook().window, len(lays)) - 1
    return int(np.argmax(np.convolve(share, np.ones(w), mode="valid")))


def write_loop(slug, info, path):
    """The whole clip as the playground's loop, H.264 at the notebook's scale."""
    nb = Notebook()
    _, frames = load_demo(slug)
    text, page, typ, ra, look = defaults(info)
    place = Place(**info["place"])
    imgs = make_loop(text, frames, place, page, typ, ra, look, range(len(frames)), nb.scale)
    h, w = imgs[0].shape[:2]
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24"]
    cmd += ["-s", f"{w}x{h}", "-r", str(info["fps"]), "-i", "-"]
    cmd += ["-c:v", "libx264", "-crf", str(nb.crf), "-pix_fmt", "yuv420p"]
    cmd += ["-movflags", "+faststart", str(path)]
    subprocess.run(cmd, input=b"".join(i.tobytes() for i in imgs), check=True)


if __name__ == "__main__":
    main()
