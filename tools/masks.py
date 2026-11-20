"""Masks: a clip in, one alpha mask per frame out, for the pieces of pieces.toml.

The command line of cutout.py: the models (BiRefNet, SAM 2) and the cleaning
live in that notebook, which is the only place that calls a segmentation
model; runaround.py only reads masks. Here: cutting a piece's clip into
frames, running every frame, and the contact sheets.

    uv sync --group masks
    uv run python tools/masks.py spring --method birefnet-general --sample 12
    uv run python tools/masks.py spring --method sam2 --sample 12
    uv run python tools/masks.py spring --sheet          # compare the sampled methods
    uv run python tools/masks.py spring --method birefnet-general   # every frame, final masks
    uv run python tools/masks.py fox --method alpha     # a drawn figure: its own alpha

A clip is a video file in clips/, or a folder of RGBA PNGs (a figure drawn
and rendered with HyperFrames, figures/<name>/): then the "alpha" method
takes the silhouette straight from the transparency, no model at all.

rembg runs on the GPU (onnxruntime-gpu with the CUDA DLLs that torch ships);
--cpu forces the CPU.

Layout of masks/<piece>/ (never committed):
    frames/%05d.png             the clip's frames between start and end
    clip.mp4                    the same frames as a video (SAM 2 reads video)
    compare/<method>/%05d.png   cleaned masks of the sampled frames
    %05d.png, rgba/%05d.png     final masks (what the text avoids, solidified)
                                and cut-outs (what is drawn), every frame
    masks.json                  how they were made
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # runaround.py is one level up

from cutout import (
    METHODS,
    alpha_masks,
    cleaned,
    cut_out_frame,
    ffmpeg,
    finish,
    on_paper,
    rembg_masks,
    sam2_masks,
    sam2_prompts,
    save_final,
    segment_from_frames,
)
from runaround import (
    FONTS,
    MASKS,
    PAPER,
    ROOT,
    load_alpha,
    load_pieces,
    piece_params,
    save_alpha,
    segment_clips,
    segment_cutout,
    segments,
)

# ------------------------------------------------------------------ frames


def extract(piece, out):
    """The frames of the piece's segments, one after the other, as PNGs, and
    each segment as a near-lossless video (SAM 2 reads video). Done once;
    reused by every method. source.json remembers which clips and times they
    come from, so masks of an old cut are never mixed with a new one; segments
    added at the end are extracted on their own, the others are kept.
    Returns (frames, spans): spans[k] = (first frame, frame count) of
    segment k."""
    frames = out / "frames"
    stamp = out / "source.json"
    spans_file = out / "spans.json"
    segs, clips = segments(piece), segment_clips(piece)
    if "segments" in piece:
        source = {"clip": piece["clip"], "segments": [list(x) for x in piece["segments"]]}
    else:  # the older single-cut form, kept so existing caches stay valid
        source = {k: piece[k] for k in ("clip", "start", "end")}
    old = json.loads(stamp.read_text()) if stamp.exists() else None
    have = frames.exists() and any(frames.iterdir())
    if have and old == source:
        found = sorted(frames.glob("*.png"))
        spans = json.loads(spans_file.read_text()) if spans_file.exists() else [[0, len(found)]]
        return found, [tuple(s) for s in spans]
    done = []
    if have:
        prefix = (
            old and "segments" in old and "segments" in source
            and old["clip"] == source["clip"]
            and old["segments"] == source["segments"][: len(old["segments"])]
        )  # fmt: skip
        if not prefix:
            raise SystemExit(
                f"{out} holds frames of {old}, the piece is now {source}: "
                "move that folder to the Recycle Bin and run again"
            )
        done = [tuple(s) for s in json.loads(spans_file.read_text())]
        print(f"keeping {len(done)} segments, adding {len(segs) - len(done)}")
    frames.mkdir(parents=True, exist_ok=True)
    fps = piece_params(piece)["reel"].fps
    spans = list(done)
    first = sum(n for _, n in done)
    for k in range(len(done), len(segs)):
        s, e = segs[k]
        clip = ROOT / "clips" / clips[k]
        if clip.is_dir():  # a rendered PNG sequence: take the frames as they are
            src = sorted(clip.glob("*.png"))[round(s * fps) : round(e * fps)]
            for i, f in enumerate(src):
                shutil.copyfile(f, frames / f"{first + i:05d}.png")
        else:
            # resampled to the reel's rate: a 60 fps or 25 fps clip keeps its real speed
            cut = ["-ss", str(s), "-to", str(e), "-i", str(clip), "-an", "-vf", f"fps={fps}"]
            run = [ffmpeg(), "-v", "error", "-y"]
            subprocess.run(
                [*run, *cut, "-start_number", str(first), str(frames / "%05d.png")], check=True
            )
        count = len(list(frames.glob("*.png"))) - first
        if not clip.is_dir():
            segment_from_frames(frames, first, count, fps, out / segment_video(k, len(segs)))
        spans.append((first, count))
        first += count
    stamp.write_text(json.dumps(source))
    spans_file.write_text(json.dumps(spans))
    return sorted(frames.glob("*.png")), spans


def segment_video(k, n):
    """File name of segment k's video ("clip.mp4" when there is only one)."""
    return "clip.mp4" if n == 1 else f"segment{k}.mp4"


def sample(n, count):
    """`count` frame indices spread evenly over n frames, first and last
    included."""
    return sorted({int(np.rint(i)) for i in np.linspace(0, n - 1, count)})


# ------------------------------------------------------------------ models


# ------------------------------------------------------------------ cleaning


# ------------------------------------------------------------------ outputs


def sheet(out, tile=480, cols=6):
    """One row per compared method, `cols` sampled frames each."""
    compare = out / "compare"
    methods = [m for m in METHODS if (compare / m).exists()]
    if not methods:
        raise SystemExit("no compare/ masks yet: run with --sample first")
    idx = sorted(int(p.stem) for p in (compare / methods[0]).glob("*.png"))
    idx = [idx[i] for i in sample(len(idx), cols)]
    frames = out / "frames"
    font = ImageFont.truetype(str(FONTS / "IBMPlexMono-Regular.ttf"), 20)
    rows = []
    for m in methods:
        tiles = []
        for i in idx:
            p = compare / m / f"{i:05d}.png"
            img = on_paper(cut_out_frame(frames / f"{i:05d}.png", load_alpha(p)), tile)
            d = ImageDraw.Draw(img)
            d.text((8, 6), f"{m}  #{i}", font=font, fill=PAPER["ink"])
            tiles.append(img)
        rows.append(tiles)
    th = rows[0][0].height
    gap = 6
    W = cols * tile + (cols - 1) * gap
    H = len(rows) * th + (len(rows) - 1) * gap
    canvas = Image.new("RGB", (W, H), "#ffffff")
    for r, tiles in enumerate(rows):
        for c, img in enumerate(tiles):
            canvas.paste(img, (c * (tile + gap), r * (th + gap)))
    path = out / "compare_sheet.png"
    canvas.save(path)
    return path


def method_sheet(out, method, tile=480, cols=4):
    """Every sampled frame of one method (12 by default), 4 per row."""
    compare = out / "compare" / method
    paths = sorted(compare.glob("*.png"))
    frames = out / "frames"
    imgs = [on_paper(cut_out_frame(frames / p.name, load_alpha(p)), tile) for p in paths]
    th = imgs[0].height
    rows = -(-len(imgs) // cols)
    canvas = Image.new("RGB", (cols * tile, rows * th), "#ffffff")
    for k, img in enumerate(imgs):
        canvas.paste(img, ((k % cols) * tile, (k // cols) * th))
    path = out / f"compare_{method}.png"
    canvas.save(path)
    return path


# ------------------------------------------------------------------ main


def main():
    pieces = load_pieces()
    ap = argparse.ArgumentParser()
    ap.add_argument("piece", choices=list(pieces))
    ap.add_argument("--method", choices=METHODS)
    ap.add_argument("--sample", type=int, help="only this many frames, into compare/<method>/")
    ap.add_argument("--sheet", action="store_true", help="contact sheet of compare/")
    ap.add_argument(
        "--segment", type=int, action="append", help="redo only this segment (repeatable)"
    )
    ap.add_argument("--cpu", action="store_true", help="rembg on the CPU (about 50x slower)")
    a = ap.parse_args()

    piece = pieces[a.piece]
    out = MASKS / a.piece
    out.mkdir(parents=True, exist_ok=True)
    if a.sheet:
        print(sheet(out))
        for m in METHODS:
            if (out / "compare" / m).exists():
                print(method_sheet(out, m))
        return
    if not a.method:
        ap.error("--method or --sheet")

    frames, spans = extract(piece, out)
    cut = piece_params(piece)["cutout"]
    t0 = time.perf_counter()

    def raw_masks(k, idx):
        """Raw model masks; a segment method "a+b" is the union of two methods
        (SAM 2 keeps thin props like guns, BiRefNet keeps whole bodies)."""
        method = piece.get("methods", {}).get(str(k), a.method)
        if "+" in method:
            print(f"segment {k}: {method}")
            parts = [one_raw_masks(k, idx, mm) for mm in method.split("+")]
            return [np.maximum.reduce(ms) for ms in zip(*parts)]
        return one_raw_masks(k, idx, method)

    def one_raw_masks(k, idx, method):
        """Raw model masks of frames `idx` (all in segment k for SAM 2). A
        segment may use another method: [piece] methods = {"<k>" = "<method>"}."""
        if method != a.method:
            print(f"segment {k}: {method}")
        if method == "sam2":  # SAM 2 cannot follow across a cut: one run per segment
            first, count = spans[k]
            # boxes the author adds by hand, in clip pixels: [piece.prompts] "<k>" = [[x0, y0, x1, y1]];
            # segments listed in `prompts_only` use those boxes alone (BiRefNet finds a mailbox)
            hand = [list(map(int, x)) for x in piece.get("prompts", {}).get(str(k), [])]
            only = str(k) in map(str, piece.get("prompts_only", []))
            b = hand if only else sam2_prompts(frames[first], cut) + hand
            print(f"SAM 2 prompts (boxes on frame {first}): {b}")
            video = out / segment_video(k, len(spans))
            got = list(sam2_masks(video, b))
            if len(got) != count:  # an older cache cut the video apart from the PNGs
                fps = piece_params(piece)["reel"].fps
                segment_from_frames(out / "frames", first, count, fps, video)
                got = list(sam2_masks(video, b))
            if len(got) != count:
                raise SystemExit(f"SAM 2 gave {len(got)} masks for {count} frames in segment {k}")
            boxes.append(b)
            return [got[i - first] for i in idx]
        if method == "alpha":
            return [alpha_masks(frames[i]) for i in idx]
        if method == "frame":
            h, w = np.asarray(Image.open(frames[idx[0]])).shape[:2]
            return [np.ones((h, w), np.float32) for _ in idx]
        return list(rembg_masks(method, [frames[i] for i in idx], cpu=a.cpu))

    boxes, done = [], 0
    if a.sample:
        keep = sample(len(frames), a.sample)
        dest = out / "compare" / a.method
        dest.mkdir(parents=True, exist_ok=True)
        for k, (first, count) in enumerate(spans):
            idx = [i for i in keep if first <= i < first + count]
            if idx:
                for i, m in zip(
                    idx, cleaned(frames, idx, raw_masks(k, idx), segment_cutout(piece, k))
                ):
                    save_alpha(m, dest / f"{i:05d}.png")
                done += len(idx)
    else:
        # one segment at a time: a 2560x1440 clip of 500 frames does not fit in
        # memory at once, and a median or a hold across a cut mixes two shots
        for k, (first, count) in enumerate(spans):
            if a.segment and k not in a.segment:
                continue
            idx = list(range(first, first + count))
            ck = segment_cutout(piece, k)
            avoid, draw = finish(frames, idx, raw_masks(k, idx), ck)
            save_final(out, frames, idx, avoid, draw)
            done += count
            del avoid, draw
    secs = time.perf_counter() - t0
    if not a.sample:
        record = dict(
            piece=a.piece,
            clip=piece["clip"],
            segments=[list(s) for s in segments(piece)],
            spans=[list(s) for s in spans],
            frames=len(frames),
            method=a.method,
            sam2_boxes=boxes or None,
            cutout=cut.__dict__,
            seconds=round(secs, 1),
        )
        (out / "masks.json").write_text(json.dumps(record, indent=2))
    print(f"{a.method}: {done} masks in {secs:.1f} s ({secs / max(done, 1):.2f} s/frame)")


if __name__ == "__main__":
    main()
