"""Reels. Everything about the page, the runaround and the look comes from
runaround.py; the values of each piece from pieces.toml.

    uv run python tools/render.py <piece> --draft   # 1080x1920, for deciding
    uv run python tools/render.py <piece>           # 2160x3840 master
    uv run python tools/render.py <piece> --remux   # only the sound, on the silent master

Output in out/<piece>/: <piece>_reel_<palette>[_draft].mp4 and, next to it, the
same name with .json: every resolved parameter of the render, so every render
can be reproduced. A draft never overwrites a master. x264 crf 14, preset slow,
bt709 tags, AAC 48 kHz at -14 LUFS.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # runaround.py is one level up

from runaround import (
    MASKS,
    ROOT,
    TEXTS,
    caption_at,
    caption_box,
    held,
    highlight_ranges,
    layout,
    load_alpha,
    load_pieces,
    marks_at,
    piece_params,
    place_alpha,
    place_figure,
    render_frame,
    scene_bands,
    segment_of,
    segment_place,
    with_variant,
)

# Not every core, never all at once (unbroken-line: hours at 100% on all 16
# threads shut the author's PC down). RUNAROUND_THREADS overrides it.
THREADS = int(os.environ.get("RUNAROUND_THREADS", "6"))


def encoder(path, w, h, reel):
    """An ffmpeg process that takes raw RGB frames on stdin."""
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(reel.fps), "-i", "-",
        "-vf", "scale=in_range=full:out_range=tv:out_color_matrix=bt709",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(reel.crf), "-preset", reel.preset,
        "-threads", str(THREADS),
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-color_range", "tv", "-movflags", "+faststart",
        str(path),
    ]  # fmt: skip
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def loudness(path, start, dur, stream=None):
    """Integrated loudness (LUFS) of a stretch of an audio file (ffmpeg ebur128);
    `stream` picks one audio stream of a film."""
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-ss", str(start), "-t", str(dur), "-i", str(path)]
    if stream is not None:
        cmd += ["-map", f"0:a:{stream}"]
    cmd += ["-af", "ebur128", "-f", "null", "-"]
    log = subprocess.run(cmd, check=True, capture_output=True, text=True).stderr
    found = re.findall(r"I:\s+(-?[\d.]+) LUFS", log)
    if not found:
        raise SystemExit(f"could not measure the loudness of {path}")
    return float(found[-1])


def add_music(silent, video, music, dur):
    """Mux the soundtrack under a finished silent video:
    the music from `music.start`, one fixed gain to `music.lufs` (measured
    once), fades, stopped at `music.cut_at` if set; the voice pieces of
    `music.voices` from the film `music.voice`, levelled once on their words;
    a limiter; 48 kHz AAC. Returns the music gain (None without music)."""
    ceiling = 10 ** (music.ceiling_db / 20)
    inputs, labels, graph, gain = ["-i", str(silent)], [], [], None
    n = 0  # index of the last ffmpeg input
    if music.file:
        track = ROOT / "music" / music.file
        end = music.cut_at or dur
        gain = music.lufs - loudness(track, music.start, end)
        fade_out = music.cut_fade if music.cut_at else music.fade_out
        n += 1
        inputs += ["-ss", str(music.start), "-t", f"{dur:.3f}", "-i", str(track)]
        bed = ""
        if music.voices and music.bed_db:  # under the voice, back to full once it ends
            v_end = max(at + b - a for a, b, at in music.voices)
            low = 10 ** (music.bed_db / 20)
            ramp = f"clip((t-{v_end:.3f})/{music.bed_rise},0,1)"
            bed = f"volume='{low:.4f}+(1-{low:.4f})*{ramp}':eval=frame,"
        graph.append(
            f"[{n}:a]atrim=0:{end:.3f},asetpts=PTS-STARTPTS,volume={gain:.2f}dB,{bed}"
            f"afade=t=in:st=0:d={music.fade_in},"
            f"afade=t=out:st={max(0.0, end - fade_out):.3f}:d={fade_out},"
            "aresample=48000,aformat=channel_layouts=stereo[m]"
        )
        labels.append("[m]")
    if music.voices:
        film = ROOT / "clips" / music.voice
        lo = music.voice_level_from or min(v[0] for v in music.voices)
        hi = music.voice_level_to or max(v[1] for v in music.voices)
        vgain = music.lufs - loudness(film, lo, hi - lo, music.voice_stream)
        for k, (a, b, at) in enumerate(music.voices):
            n += 1
            inputs += ["-ss", str(a), "-t", f"{b - a:.3f}", "-i", str(film)]
            delay = round(at * 1000)
            graph.append(
                f"[{n}:a:{music.voice_stream}]volume={vgain:.2f}dB,aresample=48000,"
                "aformat=channel_layouts=stereo,afade=t=in:d=0.05,"
                f"afade=t=out:st={max(0.0, b - a - 0.08):.3f}:d=0.08,adelay={delay}|{delay}[v{k}]"
            )
            labels.append(f"[v{k}]")
    graph.append(
        f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0:duration=longest,"
        f"aresample=48000:async=1:first_pts=0,apad,atrim=0:{dur:.3f},"
        f"alimiter=limit={ceiling:.4f}:level=disabled[a]"  # audio from t=0, silence padded
    )
    cmd = [
        "ffmpeg", "-v", "error", "-y", *inputs,
        "-filter_complex", ";".join(graph), "-map", "0:v", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", music.bitrate, "-ar", "48000",
        "-movflags", "+faststart", str(video),
    ]  # fmt: skip
    subprocess.run(cmd, check=True)
    return gain


def main():
    pieces = load_pieces()
    ap = argparse.ArgumentParser()
    ap.add_argument("piece", choices=list(pieces))
    ap.add_argument("--draft", action="store_true", help="1080 wide, its own file name")
    ap.add_argument("--frames", type=int, help="only the first N frames (quick test)")
    ap.add_argument(
        "--step", type=int, help="hold each picture this many frames (default: the piece's)"
    )
    ap.add_argument("--music", help="another file in music/ (default: the piece's)")
    ap.add_argument(
        "--variant", help="a [piece.variants.<name>] of the piece: other sound or captions"
    )
    ap.add_argument(
        "--remux",
        action="store_true",
        help="only put the music on the existing _nomusic master: no frames are drawn",
    )
    a = ap.parse_args()

    piece = pieces[a.piece]
    if a.variant:
        piece = with_variant(piece, a.variant)
    p = piece_params(piece)
    if a.step:
        p["reel"] = replace(p["reel"], step=a.step)
    if a.music:
        p["music"] = replace(p["music"], file=a.music, start=0.0)
    page, typ, ra, look, reel = (p[k] for k in ("page", "type", "runaround", "look", "reel"))
    masks = sorted((MASKS / a.piece).glob("[0-9]*.png"))
    if not masks:
        raise SystemExit(f"no masks in masks/{a.piece}/: run masks.py first")
    masks = masks[: a.frames] if a.frames else masks
    text = (TEXTS / piece["text"]).read_text(encoding="utf-8")
    scale = 1 if a.draft else 2
    w, h = page.width * scale, page.height * scale

    out = ROOT / "out" / a.piece
    out.mkdir(parents=True, exist_ok=True)
    name = f"{a.piece}_reel_{look.palette}"
    name += "_draft" if a.draft else ""
    name += "_test" if a.frames else ""
    name += f"_s{reel.step}" if reel.step > 1 else ""
    if a.variant and "captions" in piece["variants"][a.variant]:
        name += f"_{a.variant}"  # other captions, other frames: its own silent master
    music = p["music"]
    # with music, the silent master is kept too ("_nomusic"): Instagram gets it
    # and its own licensed track added in the app
    silent = out / f"{name}_nomusic.mp4"
    if a.music:  # an alternative soundtrack gets its own file name
        name += "_" + Path(a.music).stem
    if a.variant and "captions" not in piece["variants"][a.variant]:
        name += f"_{a.variant}"  # same frames as the piece, other sound
    video = out / f"{name}.mp4"
    if not (music.file or music.voices):
        silent = video

    t0 = time.perf_counter()
    if a.remux:
        if not (music.file or music.voices) or not silent.exists():
            raise SystemExit(f"--remux needs music and {silent.name}: render the piece first")
        dur = float(
            subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                 str(silent)],
                check=True, capture_output=True, text=True,
            ).stdout
        )  # fmt: skip
        gain = add_music(silent, video, music, dur)
        print(f"{video} (music {music.file or 'none'}, voices {len(music.voices)})")
        return
    enc = encoder(silent, w, h, reel)
    previous, exhausted = None, 0
    info = json.loads((MASKS / a.piece / "masks.json").read_text())
    spans = info.get("spans") or [[0, len(masks)]]
    shown = None
    sub = p["subtitles"]
    captions = piece.get("captions", [])
    bands = scene_bands(captions, sub, page, typ, spans, reel.fps)
    marked = highlight_ranges(text, piece.get("highlights", []))
    for i in range(len(masks)):
        j = held(i, spans, reel.step)
        caption = caption_at(captions, i / reel.fps)
        band = bands[i] if i < len(bands) else None
        if (j, caption) != shown:  # a new picture or caption: new layout
            m = masks[j]
            pl = segment_place(piece, segment_of(j, spans))
            alpha = place_alpha(load_alpha(m), page, pl)
            if band:
                extra = [band]
            else:
                extra = [caption_box(caption, sub, page, typ)] if caption else []
            lay, previous = layout(text, alpha, page, typ, ra, previous, extra=extra)
            figure = place_figure(MASKS / a.piece / "rgba" / m.name, page, pl)
            shown = (j, caption)
        exhausted += lay.exhausted
        rng = np.random.default_rng([piece["seed"], i])
        frame = render_frame(
            lay, page, typ, look, figure=figure, rng=rng, scale=scale, region=alpha,
            caption=caption, sub=sub, band=band, marks=marks_at(marked, i / reel.fps),
        )  # fmt: skip
        enc.stdin.write(frame.tobytes())
        if i % 24 == 0:
            print(f"frame {i}/{len(masks)}", flush=True)
    enc.stdin.close()
    if enc.wait():
        raise SystemExit("ffmpeg failed")
    gain = None
    if music.file or music.voices:
        gain = add_music(silent, video, music, len(masks) / reel.fps)
    secs = time.perf_counter() - t0

    record = dict(
        piece=a.piece,
        video=video.name,
        size=[w, h],
        frames=len(masks),
        seconds=round(secs, 1),
        text_ran_out_in=exhausted,
        music_gain_db=gain,
        masks=json.loads((MASKS / a.piece / "masks.json").read_text()),
        **{k: asdict(v) for k, v in p.items()},
    )
    (out / f"{name}.json").write_text(json.dumps(record, indent=2))
    print(f"{video} ({len(masks)} frames in {secs:.0f} s)")


if __name__ == "__main__":
    main()
