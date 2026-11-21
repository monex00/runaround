# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
#     "pillow",
#     "anywidget",
#     "imageio-ffmpeg",
#     "rembg[gpu]>=2.0.60",
#     "ultralytics>=8.3",
#     "torch>=2.7",
#     "torchvision>=0.22",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="Runaround: cut a figure out",
    css_file="notebook.css",
)

with app.setup:
    # The companion of runaround.py that does run models: a video in, a mask
    # per frame out, in masks/cutout/<name>/<shot>/, laid out as masks/<piece>/.
    # The models and the cleaning of their masks live here, below the cells,
    # and tools/masks.py imports them; the layout comes from runaround.py.
    import io
    import itertools
    import json
    import math
    import os
    import re
    import shutil
    import subprocess
    from dataclasses import asdict, dataclass, replace
    from pathlib import Path

    import anywidget
    import numpy as np
    import traitlets
    from PIL import Image

    ROOT = Path(__file__).parent
    from runaround import (
        MASKS,
        PAPER,
        Cutout,
        Look,
        Notebook,
        Page,
        Place,
        Runaround,
        Type,
        asset_text,
        clean_mask,
        glow_fill,
        hold_masks,
        ink_fill,
        load_demo,
        loop_webp,
        make_loop,
        resize_rgba,
        save_alpha,
        smooth_masks,
        solidify,
    )

    # Not every core and not all the GPU (the author's PC has shut down under
    # heavy load): thread pools capped before torch / onnxruntime / numpy load.
    THREADS = os.environ.setdefault("RUNAROUND_THREADS", "6")
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(_v, THREADS)
    # Ultralytics sends anonymous usage events unless it believes it is offline.
    os.environ.setdefault("YOLO_OFFLINE", "1")
    # rembg keeps its model files inside the project, not in the home folder.
    MODELS = ROOT / "models"
    os.environ.setdefault("U2NET_HOME", str(MODELS / "rembg"))

    REMBG = ["isnet-general-use", "isnet-anime", "birefnet-general"]
    SAM2_WEIGHTS = "sam2.1_b.pt"
    # "frame": the whole picture is the figure, a press photo the text runs around
    METHODS = [*REMBG, "sam2", "alpha", "frame"]
    # onnxruntime on the GPU. Its default memory arena grows in big steps; on
    # Windows BiRefNet then spills past the 16 GB of VRAM into system memory and
    # one frame takes 4-9 s instead of 0.3 s. Growing only by what is asked, under
    # a cap, keeps it on the card.
    CUDA_OPTIONS = {
        "arena_extend_strategy": "kSameAsRequested",
        "gpu_mem_limit": str(10 * 1024**3),
        "cudnn_conv_algo_search": "HEURISTIC",
    }

    CLIPS = ROOT / "clips"
    WORK = MASKS / "cutout"  # apart from the pieces' masks/<slug>/, never committed


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Cut a figure out

    The page in `runaround.py` never runs a model: it reads a *mask* per
    frame, white where the figure is. This notebook makes those masks from a
    video of your own, with two segmentation models on a GPU: **BiRefNet**
    finds the figure on the first frame, you check or draw the boxes around
    it, and **SAM 2** follows each box through the shot. The masks are
    cleaned as `tools/masks.py` cleans them, then laid out on the page.

    Use only clips you may use: your own, Blender's open movies (CC-BY,
    credit them), public-domain cartoons.
    """)
    return


@app.cell
def _(mo):
    import torch

    gpu = torch.cuda.is_available()
    _name = torch.cuda.get_device_name(0) if gpu else None
    mo.md(
        f"**GPU:** {_name}."
        if gpu
        else "**No GPU found.** BiRefNet and SAM 2 will run on the CPU, about fifty times slower: "
        "keep the clip to a few seconds."
    )
    return (gpu,)


@app.cell
def _(mo):
    _local = sorted(p.name for p in CLIPS.iterdir() if p.is_file()) if CLIPS.exists() else []
    video_upload = mo.ui.anywidget(VideoDrop(WORK, Scan()))
    video_pick = mo.ui.dropdown(
        ["", *_local], value="", label="or one from clips/" if _local else "(no clips/ here)"
    )
    fps = mo.ui.dropdown({"12": 12, "24": 24}, value="12", label="frames a second")
    width = mo.ui.dropdown(
        {"960": 960, "1280": 1280, "1920": 1920}, value="1280", label="width (px)"
    )
    mo.vstack(
        [
            video_upload,
            mo.hstack([video_pick, fps, width] if _local else [fps, width], justify="start"),
        ]
    )
    return fps, video_pick, video_upload, width


@app.cell
def _(mo, video_pick, video_upload):
    # The source video on disk (an upload is kept in masks/cutout/<name>/).
    if video_upload.value["path"]:
        source = Path(video_upload.value["path"])
        name = source.parent.name
    elif video_pick.value:
        source = CLIPS / video_pick.value
        name = slug(source.stem)
    else:
        mo.stop(True, mo.md("*Pick a video to start.*"))
    duration = video_duration(source)
    return duration, name, source


@app.cell
def _(mo, name, source):
    # The video read once: its cuts, and the small pictures the strip shows.
    with mo.status.progress_bar(
        total=100,
        title="Reading the video",
        subtitle="looking for the cuts",
        show_rate=False,
        show_eta=False,
        remove_on_exit=True,
    ) as _bar:
        _shown = [0]

        def _report(share, what):
            _percent = int(share * 100)
            if _percent > _shown[0]:
                _bar.update(increment=_percent - _shown[0], subtitle=what)
                _shown[0] = _percent

        _scan = scan_video(source, WORK / name, Scan(), _report)
    picker = mo.ui.anywidget(ShotPicker(**_scan))
    mo.vstack(
        [
            mo.md(
                "**Pick the shot.** The strip is the whole video, divided at its cuts. Click "
                "a shot, then trim it. Stay inside one: SAM 2 cannot follow a figure across "
                "a cut."
            ),
            picker,
        ]
    )
    return (picker,)


@app.cell
def _(duration, fps, name, picker, width):
    # Each shot gets its own folder, so nothing made before is overwritten.
    shot = (round(picker.value["start"], 1), round(min(picker.value["end"], duration), 1))
    work = WORK / name / f"{shot[0]:.1f}-{shot[1]:.1f}s_{fps.value}fps_{width.value}px"
    return shot, work


@app.cell
def _(mo):
    take = mo.ui.run_button(label="take the frames")
    take
    return (take,)


@app.cell
def _(fps, mo, shot, source, take, width, work):
    # The frames of the shot, once: kept on disk with a stamp of where they
    # come from, so a change of shot never mixes old frames with new ones.
    _wanted = dict(source=str(source), start=shot[0], end=shot[1], fps=fps.value, width=width.value)
    _stamp = work / "source.json"
    _have = _stamp.exists() and json.loads(_stamp.read_text()) == _wanted
    if not _have:
        mo.stop(not take.value, mo.md("*Press **take the frames** when the shot is right.*"))
        work.mkdir(parents=True, exist_ok=True)
        with mo.status.spinner(title="Taking the frames"):
            take_frames(source, work, **{k: _wanted[k] for k in ("start", "end", "fps", "width")})
        _stamp.write_text(json.dumps(_wanted))
    frames = sorted((work / "frames").glob("*.png"))
    mo.md(
        f"{len(frames)} frames at {fps.value} a second, in `{work.relative_to(ROOT).as_posix()}`."
    )
    return (frames,)


@app.cell
def _(mo):
    mo.md(r"""
    ## The boxes

    SAM 2 needs to be told what to follow: a box around each part of the
    figure. BiRefNet proposes them on the first frame. If someone comes in
    later, move along the shot to a frame where they are in view and drag a
    box around them: SAM 2 follows every box forward and backward from its
    frame. A part with no box is not followed. Back on a shot already cut
    out, the boxes are the ones you used.
    """)
    return


@app.cell
def _(fps, frames, mo, shot, work):
    # The boxes of the last cut-out of this shot, else BiRefNet's on the first frame.
    _record = work / "masks.json"
    _last = json.loads(_record.read_text()).get("boxes") if _record.exists() else None
    if _last:
        start_boxes = as_prompts(_last)
    else:
        with mo.status.spinner(title="BiRefNet looks for the figure"):
            try:
                start_boxes = as_prompts(sam2_prompts(frames[0], Cutout()))
            except SystemExit:  # nobody on the first frame: the boxes are up to you
                start_boxes = []
    boxer = mo.ui.anywidget(BoxDrawer(frames, shot[0], fps.value, start_boxes))
    boxer
    return (boxer,)


@app.cell
def _(mo):
    method = mo.ui.radio(
        {
            "SAM 2: follows the boxes": "sam2",
            "BiRefNet: every frame on its own": "birefnet-general",
            "both, joined": "sam2+birefnet-general",
        },
        value="SAM 2: follows the boxes",
        label="model",
    )
    line_art = mo.ui.checkbox(
        value=False, label="black outlines on a light background (old cartoons)"
    )
    cut_run = mo.ui.run_button(label="cut out")
    mo.vstack([method, line_art, cut_run])
    return cut_run, line_art, method


@app.cell
def _(boxer, cut_run, fps, frames, line_art, method, mo, work):
    # Raw masks from the model(s), cleaned and finished as tools/masks.py does,
    # saved in masks/<name>/: %05d.png the text avoids, rgba/ the cut-out.
    _boxes = as_prompts(boxer.value["boxes"])
    cut = replace(Cutout(), ink=Scan().line_art if line_art.value else 0.0)
    _wanted = dict(method=method.value, boxes=_boxes, cutout=asdict(cut), frames=len(frames))
    _record = work / "masks.json"
    _have = (
        _record.exists() and {k: json.loads(_record.read_text()).get(k) for k in _wanted} == _wanted
    )
    if not _have:
        mo.stop(not cut_run.value, mo.md("*Press **cut out** when the boxes are right.*"))
        mo.stop("sam2" in method.value and not _boxes, mo.md("*SAM 2 needs at least one box.*"))
        _idx = list(range(len(frames)))
        _raws = []
        for _m in method.value.split("+"):
            if _m == "sam2":
                with mo.status.progress_bar(
                    total=tracking_steps(len(frames), _boxes), title="SAM 2 follows the figure"
                ) as _bar:
                    _raws.append(follow_boxes(frames, _boxes, fps.value, work, _bar.update))
            else:
                _gen = rembg_masks(_m, frames)
                _raws.append(
                    list(
                        mo.status.progress_bar(
                            _gen, total=len(frames), title="BiRefNet, frame by frame"
                        )
                    )
                )
        _raw = [np.maximum.reduce(ms) for ms in zip(*_raws)]
        with mo.status.spinner(title="Cleaning the masks"):
            _avoid, _draw = finish(frames, _idx, _raw, cut)
            save_final(work, frames, _idx, _avoid, _draw)
        _record.write_text(json.dumps(_wanted, indent=2))
    figure_full = [Image.open(work / "rgba" / f.name).convert("RGBA") for f in frames]
    _n, _w = Notebook().sheet
    _sample = [figure_full[k] for k in np.linspace(0, len(frames) - 1, _n).astype(int)]
    mo.vstack(
        [
            mo.md(f"{_n} frames of the cut-out, on paper:"),
            mo.hstack([on_paper(r, _w) for r in _sample], justify="start", wrap=True),
        ]
    )
    return (figure_full,)


@app.cell
def _(mo):
    mo.md(r"""
    ## On the page

    The masks laid out as `runaround.py` lays them out. Place the figure,
    then make the loop.
    """)
    return


@app.cell
def _(mo):
    scale = mo.ui.slider(
        0.5, 1.6, step=0.05, value=Place().scale, label="figure width (share of the page)"
    )
    bottom = mo.ui.slider(
        0.6, 1.2, step=0.02, value=Place().bottom, label="bottom edge (share of the page)"
    )
    loop_run = mo.ui.run_button(label="make the loop")
    mo.vstack([mo.hstack([scale, bottom], justify="start"), loop_run])
    return bottom, loop_run, scale


@app.cell
def _(bottom, figure_full, fps, frames, loop_run, mo, name, scale, work):
    # The figure at the notebook's size, (cut-out, mask the text avoids) per
    # frame, the shape runaround.py's playground takes.
    _nb, _page = Notebook(), Page()
    _w = _nb.figure_width
    figure = []
    for _rgba, _f in zip(figure_full, frames):
        _h = round(_rgba.height * _w / _rgba.width)
        _mask = Image.open(work / _f.name).convert("L").resize((_w, _h), Image.BILINEAR)
        figure.append((resize_rgba(_rgba, (_w, _h)), np.asarray(_mask, np.float32) / 255.0))
    _place = Place(scale=scale.value, bottom=bottom.value)
    _text = asset_text("texts/love.txt")
    _look = Look(**{**load_demo("fox")[0]["look"], "figure": "colour"})  # the demos' print look
    if loop_run.value:
        _frames = make_loop(
            _text,
            figure,
            _place,
            _page,
            Type(),
            Runaround(),
            _look,
            range(len(figure)),
            _nb.scale,
            progress=lambda ks: mo.status.progress_bar(
                ks, title="Making the loop", remove_on_exit=True
            ),
        )
        _view = mo.image(
            loop_webp(_frames, fps.value, _nb.quality), width=round(_page.width * _nb.scale)
        )
    else:
        _still = make_loop(
            _text, figure, _place, _page, Type(), Runaround(), _look, [0], _nb.scale
        )[0]
        _view = mo.image(Image.fromarray(_still), width=round(_page.width * _nb.scale))
    _cutout = cutout_webp([r for r, _ in figure], fps.value)
    mo.vstack(
        [
            _view,
            mo.md(
                "Take the cut-out to the playground of `runaround.py` (*your own cut-out*): "
                "an animated WebP whose transparency is the mask."
            ),
            mo.download(
                _cutout,
                filename=f"{name}_cutout.webp",
                mimetype="image/webp",
                label="cut-out (WebP)",
            ),
        ]
    )
    return


@app.function
def slug(text):
    """A folder name from a file name: lower case, letters, digits and _."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "clip"


@app.function
def video_duration(path):
    """Length of a video in seconds, from ffmpeg's own report (ffprobe may be
    missing where only ffmpeg is)."""
    r = subprocess.run([ffmpeg(), "-i", str(path)], capture_output=True, text=True, check=False)
    h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r.stderr).groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


@app.function
def take_frames(source, work, start, end, fps, width):
    """The shot as PNG frames in work/frames/ (at `fps`, at most `width` px
    wide) and as work/clip.mp4, made from those very PNGs, for SAM 2."""
    frames = work / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    vf = f"fps={fps},scale='min({width},iw)':-2"
    cmd = [ffmpeg(), "-v", "error", "-y", "-ss", str(start), "-to", str(end), "-i", str(source)]
    cmd += ["-an", "-vf", vf, "-start_number", "0", str(frames / "%05d.png")]
    subprocess.run(cmd, check=True)
    count = len(list(frames.glob("*.png")))
    segment_from_frames(frames, 0, count, fps, work / "clip.mp4")


@app.function
def frame_jpeg(path):
    """A frame of the shot as JPEG bytes, for the box drawer."""
    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, format="JPEG", quality=Scan().frame_quality)
    return buf.getvalue()


@app.function
def cutout_webp(rgbas, fps):
    """Cut-outs as an animated WebP with transparency, the format the
    playground of runaround.py takes as "your own cut-out"."""
    buf = io.BytesIO()
    rgbas[0].save(
        buf,
        format="WEBP",
        save_all=True,
        append_images=rgbas[1:],
        duration=round(1000 / fps),
        loop=0,
        quality=85,
        exact=True,
    )
    return buf.getvalue()


@app.function
def ffmpeg():
    """The ffmpeg program: the one on PATH, else the binary imageio-ffmpeg
    ships (a cloud notebook may have no ffmpeg installed)."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


@app.function
def segment_from_frames(frames, first, count, fps, path):
    """The segment's video for SAM 2, made from the very PNGs the masks are
    matched to: two separate ffmpeg cuts can round to different frame counts."""
    cmd = [ffmpeg(), "-v", "error", "-y", "-framerate", str(fps), "-start_number", str(first)]
    cmd += ["-i", str(frames / "%05d.png"), "-frames:v", str(count)]
    cmd += ["-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)


@app.function
def rembg_masks(method, paths, cpu=False):
    """Raw alpha per frame from a rembg model, on the GPU unless `cpu`."""
    if not cpu:
        # torch ships the CUDA 13 and cuDNN 9 DLLs that onnxruntime-gpu needs;
        # importing it first loads them into the process.
        import torch  # noqa: F401
    from rembg import new_session, remove

    providers = ["CPUExecutionProvider"] if cpu else [("CUDAExecutionProvider", CUDA_OPTIONS)]
    session = new_session(method, providers=providers)
    used = session.inner_session.get_providers()
    if not cpu and "CUDAExecutionProvider" not in used:
        raise SystemExit(f"{method}: CUDA not available ({used}); use --cpu to run slowly")
    for p in paths:
        m = remove(Image.open(p).convert("RGB"), session=session, only_mask=True)
        yield np.asarray(m, dtype=np.float32) / 255.0


@app.function
def alpha_masks(path):
    """The alpha channel of an RGBA frame, as float alpha in [0, 1]."""
    img = Image.open(path)
    if "A" not in img.getbands():
        raise SystemExit(f"{path} has no alpha channel: use a segmentation method")
    return np.asarray(img.getchannel("A"), dtype=np.float32) / 255.0


@app.function
def sam2_prompts(first_frame, cut):
    """Boxes around each part of the figure on the first frame, found by
    BiRefNet: SAM 2 then follows each part through the shot."""
    from scipy import ndimage

    alpha = clean_mask(next(rembg_masks("birefnet-general", [first_frame])), cut)
    solid = alpha >= cut.threshold
    labels, n = ndimage.label(solid)
    if not n:
        raise SystemExit("BiRefNet found no figure on the first frame to give SAM 2")
    areas = ndimage.sum_labels(solid, labels, index=range(1, n + 1))
    boxes = []
    for k, sl in enumerate(ndimage.find_objects(labels)):
        # small parts (a crack in the wall, a speck) are not followed
        if sl is not None and areas[k] >= cut.prompt_part * areas.max():
            ys, xs = sl
            boxes.append([xs.start, ys.start, xs.stop, ys.stop])
    return boxes


@app.function
def sam2_masks(video, boxes):
    """Binary alpha per frame from SAM 2.1 (GPU), one tracked object per box,
    merged into one figure."""
    from ultralytics.models.sam import SAM2VideoPredictor
    from ultralytics.utils.downloads import attempt_download_asset

    MODELS.mkdir(exist_ok=True)
    weights = attempt_download_asset(str(MODELS / SAM2_WEIGHTS))
    predictor = SAM2VideoPredictor(
        overrides=dict(model=str(weights), imgsz=1024, conf=0.25, save=False, verbose=False)
    )
    for r in predictor(source=str(video), bboxes=boxes, stream=True):
        h, w = r.orig_shape
        if r.masks is None:
            yield np.zeros((h, w), np.float32)
            continue
        m = r.masks.data.cpu().numpy().astype(bool).any(axis=0)
        if m.shape != (h, w):  # masks come back at the model's size
            m = np.asarray(Image.fromarray(m).resize((w, h), Image.NEAREST))
        yield m.astype(np.float32)


@app.function
def as_prompts(boxes):
    """Boxes as {"frame": k, "box": [x0, y0, x1, y1]}. A plain box (BiRefNet's
    proposals, cut-outs saved before boxes had a frame) is on the first frame."""
    out = []
    for b in boxes:
        frame, box = (b["frame"], b["box"]) if isinstance(b, dict) else (0, b)
        out.append(dict(frame=int(frame), box=[int(v) for v in box]))
    return out


@app.function
def prompt_groups(prompts):
    """{frame: [boxes on it]}, frames in order: SAM 2 starts once from each."""
    groups = {}
    for p in sorted(prompts, key=lambda p: p["frame"]):
        groups.setdefault(p["frame"], []).append(p["box"])
    return groups


@app.function
def tracking_steps(count, prompts):
    """How many frames SAM 2 goes through: from each frame with boxes to the
    end, and back to the start (the frame itself twice)."""
    return sum(count + (1 if k else 0) for k in prompt_groups(prompts))


@app.function
def clip_from(frames, k, fps, work, backward=False):
    """A video for SAM 2, which starts from its first frame: the shot from
    frame k to the end, or from k back to the first frame."""
    path = work / f"track_{k:05d}_{'back' if backward else 'on'}.mp4"
    if not backward:
        segment_from_frames(frames[0].parent, k, len(frames) - k, fps, path)
        return path
    cmd = [ffmpeg(), "-v", "error", "-y", "-framerate", str(fps), "-start_number", "0"]
    # cut before turning it round: the video must begin on frame k
    cmd += ["-i", str(frames[0].parent / "%05d.png"), "-vf", f"trim=end_frame={k + 1},reverse"]
    cmd += ["-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)
    return path


@app.function
def follow_boxes(frames, prompts, fps, work, tick=lambda: None):
    """One mask per frame from SAM 2, for boxes drawn on any frame: it follows
    each group of boxes from its frame to the end and back to the start, and
    the results are joined. Someone who walks in later is simply empty before."""
    joined = None
    for k, boxes in prompt_groups(prompts).items():
        runs = [(1, len(frames) - k, clip_from(frames, k, fps, work))]
        if k:
            runs.append((-1, k + 1, clip_from(frames, k, fps, work, backward=True)))
        for step, length, video in runs:
            seen = 0
            for i, mask in enumerate(sam2_masks(video, boxes)):
                if joined is None:
                    joined = np.zeros((len(frames), *mask.shape), bool)
                joined[k + step * i] |= mask >= 0.5
                seen += 1
                tick()
            if seen != length:
                raise RuntimeError(f"SAM 2 gave {seen} masks for {length} frames")
    return [m.astype(np.float32) for m in joined]


@app.function
def cleaned(frames, idx, raw, cut):
    """Raw masks of frames `idx` cleaned one by one: ink and glow recovered,
    specks dropped, small holes filled."""
    out = []
    for i, m in zip(idx, raw):
        img = Image.open(frames[i]).convert("RGB")
        gray = np.asarray(img.convert("L"), np.float32) / 255.0
        m = glow_fill(ink_fill(m, gray, cut), np.asarray(img), cut)
        out.append(clean_mask(m, cut))
    return out


@app.function
def finish(frames, idx, raw, cut):
    """The final masks of one shot from its raw masks: (avoid, draw), the
    mask the text keeps away from (solid, held across a bad frame) and the
    one the cut-out is drawn with, as stacks of float arrays."""
    clean = smooth_masks(np.stack(cleaned(frames, idx, raw, cut)), cut)
    avoid = hold_masks(np.stack([solidify(m, cut) for m in clean]), cut)
    drawn = replace(cut, solidify=cut.draw_close)
    draw = np.stack([solidify(m, drawn) if cut.draw_close else m for m in clean])
    if cut.draw_hold:
        draw = hold_masks(draw, replace(cut, hold=cut.draw_hold))
    return avoid, draw


@app.function
def save_final(out, frames, idx, avoid, draw):
    """masks/<piece>/%05d.png (what the text avoids) and rgba/%05d.png (the
    frame cut out, what is drawn)."""
    (out / "rgba").mkdir(parents=True, exist_ok=True)
    for i, d, v in zip(idx, draw, avoid):
        save_alpha(v, out / f"{i:05d}.png")
        cut_out_frame(frames[i], d).save(out / "rgba" / f"{i:05d}.png")


@app.function
def cut_out_frame(frame_path, alpha):
    """The frame with the mask as its alpha channel."""
    rgb = np.asarray(Image.open(frame_path).convert("RGB"))
    a = np.clip(np.rint(alpha * 255), 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, a]))


@app.function
def on_paper(rgba, width):
    """A cut-out over the paper colour, scaled to `width`: how the author
    judges the edges."""
    h = round(rgba.height * width / rgba.width)
    bg = Image.new("RGBA", rgba.size, PAPER["bg"])
    return Image.alpha_composite(bg, rgba).convert("RGB").resize((width, h), Image.LANCZOS)


@app.class_definition
@dataclass(frozen=True)
class Scan:
    """How a video is read for the shot picker: its cuts and a sheet of small
    pictures the browser scrubs through. Times in seconds."""

    cut: float = 0.3  # how much a frame must differ from the last to be a cut (0..1)
    width: int = 160  # px, the frames compared to find the cuts
    min_shot: float = 0.25  # cuts closer than this are one cut (a flash)
    thumb: int = 320  # px, the pictures of the strip
    rate: float = 4.0  # pictures a second, at most
    thumbs: int = 900  # pictures in all, at most
    cols: int = 30  # pictures per row of the sheet
    quality: int = 6  # JPEG quality of the sheet (ffmpeg: 2 best, 31 worst)
    default: float = 3.0  # length of the first selection
    shortest: float = 0.5  # the selection cannot be shorter
    longest: float = 20.0  # or longer
    kinds: tuple = (".mp4", ".mov", ".mkv", ".webm")  # videos that can be uploaded
    upload: int = 500_000_000  # bytes, the longest upload
    piece: int = 2_000_000  # bytes sent at a time, so the bar can move
    reading: float = 0.85  # share of the bar for finding the cuts; the rest is the strip
    frame_quality: int = 85  # JPEG quality of the frames the boxes are drawn on
    line_art: float = 0.35  # darkness that counts as an outline, for old cartoons


@app.function
def clean_cuts(times, duration, gap):
    """The cut times in order, without those at the very ends of the video
    and with cuts closer than `gap` counted once."""
    out = []
    for t in sorted(times):
        if gap <= t <= duration - gap and (not out or t - out[-1] >= gap):
            out.append(t)
    return out


@app.function
def default_shot(cuts, duration, scan):
    """The first selection: the start of the longest shot, on tenths of a
    second, never past its cut."""
    bounds = [0.0, *cuts, duration]
    s, e = max(itertools.pairwise(bounds), key=lambda p: p[1] - p[0])
    first = math.ceil(round(s * 10, 6)) / 10
    last = math.floor(round(e * 10, 6)) / 10
    return first, round(min(last, first + scan.default), 1)


@app.function
def upload_path(folder, filename, size):
    """Where an uploaded video is kept: <folder>/<name>/source.<ext>. A
    different video with the same file name gets a folder of its own, so its
    shots never mix with the other's."""
    path = Path(filename)
    name = slug(path.stem)
    target = folder / name / f"source{path.suffix.lower()}"
    if target.exists() and target.stat().st_size != size:
        target = folder / f"{name}_{size}" / target.name
    return target


@app.class_definition
class VideoDrop(anywidget.AnyWidget):
    """A place to drop a video, with a bar. The file travels in small pieces
    straight to disk, so the page can say how far it has got and a long video
    is never held in memory. `path` changes when the whole file has arrived."""

    _css = r"""
    .vd {
      display: block; box-sizing: border-box; width: 100%; padding: 18px 16px;
      border: 1.5px dashed #383b3e; border-radius: 4px; color: #383b3e;
      cursor: pointer; text-align: center; outline: none; background: transparent;
    }
    .vd:hover, .vd:focus-visible, .vd.vd-over { background: rgba(242, 217, 78, .28); }
    .vd.vd-busy { cursor: progress; border-style: solid; }
    .vd-main { font-size: 16px; }
    .vd-sub { font: 12px 'IBM Plex Mono', ui-monospace, monospace; margin-top: 4px; opacity: .75; }
    .vd-bar { display: none; height: 6px; margin-top: 10px; background: rgba(56, 59, 62, .18); }
    .vd-bar i { display: block; height: 100%; width: 0; background: #383b3e; transition: width .15s; }
    .vd-busy .vd-bar { display: block; }
    .vd-error .vd-sub { opacity: 1; background: #f2d94e; display: inline-block; padding: 1px 6px; }
    """

    _esm = r"""
    const mb = (n) => (n / 1e6).toFixed(n < 1e7 ? 1 : 0);

    function render({ model, el }) {
      const kinds = model.get("kinds"), limit = model.get("limit"), piece = model.get("piece");
      el.innerHTML = `<div class="vd" role="button" tabindex="0">
        <input type="file" accept="${kinds.join(",")}" hidden>
        <div class="vd-main"></div><div class="vd-sub"></div><div class="vd-bar"><i></i></div>
      </div>`;
      const box = el.querySelector(".vd"), input = el.querySelector("input");
      const main = el.querySelector(".vd-main"), sub = el.querySelector(".vd-sub");
      const bar = el.querySelector(".vd-bar i");
      let file = null, started = 0;

      function idle(title, note, error) {
        file = null;
        box.className = error ? "vd vd-error" : "vd";
        main.textContent = title; sub.textContent = note;
      }
      const hint = `${kinds.map((k) => k.slice(1)).join(", ")}, up to ${mb(limit)} MB`;
      const empty = () => idle("Drop a video here, or click to choose one", hint);
      const shown = model.get("shown");
      if (shown) idle(`${shown} is here`, "Drop another video to replace it"); else empty();

      function begin(f) {
        if (file || !f) return;
        const dot = f.name.lastIndexOf(".");
        if (!kinds.includes(f.name.slice(dot).toLowerCase())) return idle(f.name, `not a video this page can read (${hint})`, true);
        if (f.size > limit) return idle(f.name, `${mb(f.size)} MB is too long: the limit is ${mb(limit)} MB`, true);
        file = f; started = performance.now();
        box.className = "vd vd-busy"; bar.style.width = "0";
        main.textContent = `Uploading ${f.name}`; sub.textContent = `0 of ${mb(f.size)} MB`;
        model.send({ kind: "start", name: f.name, size: f.size });
      }
      async function next(at) {
        const share = at / file.size, secs = (performance.now() - started) / 1000;
        bar.style.width = `${share * 100}%`;
        const left = share > 0.03 ? Math.ceil((secs * (1 - share)) / share) : null;
        sub.textContent = `${mb(at)} of ${mb(file.size)} MB` + (left === null ? "" : ` · about ${left} s left`);
        if (at >= file.size) { main.textContent = `Saving ${file.name}`; model.send({ kind: "end" }); return; }
        const sent = file;
        const buffer = await file.slice(at, at + piece).arrayBuffer();
        if (sent === file) model.send({ kind: "piece", at }, undefined, [buffer]);
      }
      const onMessage = (msg) => {
        if (!file) return;
        if (msg.kind === "next") next(msg.at);
        else if (msg.kind === "error") idle(file.name, msg.text, true);
        else if (msg.kind === "saved") {
          const name = file.name;
          idle(`${name} is here`, "Drop another video to replace it");
          model.set("shown", name); model.set("path", msg.path); model.save_changes();
        }
      };
      model.on("msg:custom", onMessage);

      box.addEventListener("click", () => { if (!file) input.click(); });
      box.addEventListener("keydown", (e) => {
        if ((e.key === "Enter" || e.key === " ") && !file) { e.preventDefault(); input.click(); }
      });
      input.addEventListener("change", () => { begin(input.files[0]); input.value = ""; });
      for (const type of ["dragenter", "dragover"]) {
        box.addEventListener(type, (e) => { e.preventDefault(); if (!file) box.classList.add("vd-over"); });
      }
      box.addEventListener("dragleave", () => box.classList.remove("vd-over"));
      box.addEventListener("drop", (e) => {
        e.preventDefault(); box.classList.remove("vd-over"); begin(e.dataTransfer.files[0]);
      });
      return () => model.off("msg:custom", onMessage);
    }
    export default { render };
    """
    kinds = traitlets.List([]).tag(sync=True)
    limit = traitlets.Int(0).tag(sync=True)
    piece = traitlets.Int(0).tag(sync=True)
    path = traitlets.Unicode("").tag(sync=True)
    shown = traitlets.Unicode("").tag(sync=True)

    def __init__(self, folder, scan):
        super().__init__(kinds=list(scan.kinds), limit=scan.upload, piece=scan.piece)
        self._folder, self._target, self._part, self._size, self._at = folder, None, None, 0, 0
        self.on_msg(self._on_piece)

    def _on_piece(self, _widget, msg, buffers):
        """One message of an upload: start, a piece of the file, or the end."""
        try:
            if msg["kind"] == "start":
                self._begin(msg["name"], int(msg["size"]))
            elif msg["kind"] == "piece":
                self._write(int(msg["at"]), bytes(buffers[0]))
            elif msg["kind"] == "end":
                self._finish()
        except (OSError, ValueError) as e:
            self.send({"kind": "error", "text": str(e)})

    def _begin(self, filename, size):
        if Path(filename).suffix.lower() not in self.kinds or not 0 < size <= self.limit:
            raise ValueError("this file cannot be uploaded here")
        self._target = upload_path(self._folder, filename, size)
        self._target.parent.mkdir(parents=True, exist_ok=True)
        self._part = self._target.with_suffix(self._target.suffix + ".part")
        self._part.write_bytes(b"")
        self._size, self._at = size, 0
        self.send({"kind": "next", "at": 0})

    def _write(self, at, data):
        if self._part is None or at != self._at:
            raise ValueError("a piece of the file arrived out of order: try again")
        with self._part.open("ab") as f:
            f.write(data)
        self._at += len(data)
        self.send({"kind": "next", "at": self._at})

    def _finish(self):
        if self._part is None or self._at != self._size:
            raise ValueError("the file arrived incomplete: try again")
        self._part.replace(self._target)
        self._part = None
        self.send({"kind": "saved", "path": str(self._target)})


@app.function
def find_cuts(source, duration, scan, report):
    """The times where the picture changes at once. ffmpeg prints a score for
    every frame, so `report(share of the video read)` can follow it."""
    find = (
        f"scale={scan.width}:-2,select='gte(scene,0)',metadata=print:key=lavfi.scene_score:file=-"
    )
    cmd = [ffmpeg(), "-v", "error", "-i", str(source), "-an", "-vf", find, "-f", "null", "-"]
    times, now = [], 0.0
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as p:
        for raw in p.stdout:
            line = raw.decode("ascii", "replace").strip()
            if line.startswith("frame:"):
                now = float(line.rsplit("pts_time:", 1)[1])
                report(min(now / duration, 1.0))
            elif line.startswith("lavfi.scene_score=") and float(line.split("=")[1]) > scan.cut:
                times.append(now)
    if p.returncode:
        raise RuntimeError(f"ffmpeg could not read {source.name}")
    return clean_cuts(times, duration, scan.min_shot)


@app.function
def scan_video(source, folder, scan, report=lambda share, what: None):
    """What the shot picker needs: the video's cuts and a JPEG sheet of small
    pictures. Kept in `folder`, under the size and date of the video, so a
    video is read once. `report(share, what)` follows the work."""
    info = source.stat()
    stem = folder / f"scan_{info.st_size}_{int(info.st_mtime)}"
    meta_file, sheet_file = stem.with_suffix(".json"), stem.with_suffix(".jpg")
    if meta_file.exists() and sheet_file.exists():
        saved = json.loads(meta_file.read_text())
        return dict(saved, sprite=sheet_file.read_bytes())
    duration = video_duration(source)
    cuts = find_cuts(
        source, duration, scan, lambda d: report(d * scan.reading, "looking for the cuts")
    )
    report(scan.reading, "making the strip")
    rate = min(scan.rate, scan.thumbs / duration)
    count = max(1, int(duration * rate))
    rows = math.ceil(count / scan.cols)
    tile = f"fps={rate:.5f},scale={scan.thumb}:-2,tile={scan.cols}x{rows}"
    cmd = [ffmpeg(), "-v", "error", "-i", str(source), "-an", "-vf", tile, "-frames:v", "1"]
    cmd += ["-q:v", str(scan.quality), "-f", "image2pipe", "-c:v", "mjpeg", "-"]
    sprite = subprocess.run(cmd, capture_output=True, check=True).stdout
    start, end = default_shot(cuts, duration, scan)
    meta = dict(
        duration=duration,
        rate=rate,
        count=count,
        cols=scan.cols,
        rows=rows,
        shortest=scan.shortest,
        longest=scan.longest,
    )
    saved = dict(meta=meta, cuts=cuts, start=start, end=end)
    folder.mkdir(parents=True, exist_ok=True)
    sheet_file.write_bytes(sprite)
    meta_file.write_text(json.dumps(saved))
    return dict(saved, sprite=sprite)


@app.class_definition
class ShotPicker(anywidget.AnyWidget):
    """The video as a strip cut into its shots. Click a shot, then drag the
    two ends of the selection in the close-up below. `start` and `end` reach
    Python only when the mouse is released."""

    _css = r"""
    .sp { color: #383b3e; max-width: 100%; position: relative; }
    .sp-frames { display: flex; gap: 8px; margin-bottom: 8px; }
    .sp-frames figure { flex: 0 1 400px; margin: 0; min-width: 0; }
    .sp-frames canvas { display: block; width: 100%; background: #383b3e; }
    .sp-frames figcaption, .sp-label, .sp-read {
      font: 12px 'IBM Plex Mono', ui-monospace, monospace;
    }
    .sp-frames figcaption { margin-top: 3px; opacity: .7; }
    .sp-label { margin: 10px 0 3px; opacity: .7; }
    .sp-strip { display: block; width: 100%; touch-action: none; outline: none; }
    .sp-all { height: 44px; cursor: pointer; }
    .sp-near { height: 76px; cursor: ew-resize; }
    .sp-near:focus-visible { outline: 2px solid #383b3e; }
    .sp-read { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin-top: 8px; }
    .sp-read b { font-weight: 600; }
    .sp-read button {
      font: inherit; color: inherit; background: transparent; cursor: pointer;
      border: 1px solid #383b3e; border-radius: 3px; padding: 2px 10px;
    }
    .sp-warn { background: #f2d94e; padding: 1px 6px; }
    .sp-tip {
      position: absolute; display: none; pointer-events: none; z-index: 5;
      background: #383b3e; color: #f4f0e8; padding: 3px;
      font: 11px 'IBM Plex Mono', ui-monospace, monospace; text-align: center;
    }
    .sp-tip canvas { display: block; width: 160px; }
    """

    _esm = r"""
    const INK = "#383b3e", MARK = "#f2d94e";
    const up10 = (t) => Math.ceil(Math.round(t * 1e6) / 1e5) / 10;
    const down10 = (t) => Math.floor(Math.round(t * 1e6) / 1e5) / 10;
    const near10 = (t) => Math.round(t * 10) / 10;
    const clamp = (v, a, b) => Math.min(Math.max(v, a), b);

    function render({ model, el }) {
      const m = model.get("meta"), cuts = model.get("cuts"), dur = m.duration;
      const bounds = [0, ...cuts, dur];
      let a = model.get("start"), b = model.get("end");
      let view = [0, dur], hover = -1, drag = null, side = "b", playing = 0, playT = null, saveTimer = 0;

      el.innerHTML = `<div class="sp">
        <div class="sp-frames">
          <figure><canvas data-f="a"></canvas><figcaption>first frame</figcaption></figure>
          <figure><canvas data-f="b"></canvas><figcaption>last frame</figcaption></figure>
        </div>
        <div class="sp-read"><button type="button">play</button><span class="sp-time"></span><span class="sp-note"></span></div>
        <div class="sp-label">the whole video: click a shot</div>
        <canvas class="sp-strip sp-all"></canvas>
        <div class="sp-label">this shot: drag the two ends, or the middle to move them together (arrow keys: a tenth of a second)</div>
        <canvas class="sp-strip sp-near" tabindex="0"></canvas>
        <div class="sp-tip"><canvas></canvas><span></span></div>
      </div>`;
      const root = el.querySelector(".sp");
      const [fa, fb] = el.querySelectorAll(".sp-frames canvas");
      const all = el.querySelector(".sp-all"), near = el.querySelector(".sp-near");
      const time = el.querySelector(".sp-time"), note = el.querySelector(".sp-note");
      const play = el.querySelector("button"), tip = el.querySelector(".sp-tip");
      const tipPic = tip.querySelector("canvas"), tipTime = tip.querySelector("span");

      const raw = model.get("sprite");
      const bytes = raw.buffer ? new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength) : raw;
      const url = URL.createObjectURL(new Blob([bytes], { type: "image/jpeg" }));
      const sheet = new Image();
      let tw = 0, th = 0;

      const shotAt = (t) => {
        for (let i = 0; i < bounds.length - 1; i++) if (t < bounds[i + 1]) return i;
        return bounds.length - 2;
      };
      let cur = shotAt((a + b) / 2);

      function pic(x, t, dx, dy, w, h) {
        if (!tw) return;
        const i = clamp(Math.round(t * m.rate), 0, m.count - 1);
        x.drawImage(sheet, (i % m.cols) * tw, Math.floor(i / m.cols) * th, tw, th, dx, dy, w, h);
      }
      function fit(c) {
        const w = c.clientWidth || 800, h = c.clientHeight, r = window.devicePixelRatio || 1;
        if (c.width !== Math.round(w * r) || c.height !== Math.round(h * r)) {
          c.width = Math.round(w * r); c.height = Math.round(h * r);
        }
        const x = c.getContext("2d");
        x.setTransform(r, 0, 0, r, 0, 0);
        return [x, w, h];
      }
      function film(x, w, h, t0, t1) {
        const step = Math.max(8, h * (tw / (th || 1)));
        x.fillStyle = INK; x.fillRect(0, 0, w, h);
        for (let p = 0; p < w; p += step) pic(x, t0 + ((p + step / 2) / w) * (t1 - t0), p, 0, step, h);
      }
      function setView() {
        const s = bounds[cur], e = bounds[cur + 1], pad = Math.max(1, (e - s) * 0.15);
        let f = Math.min(s, a) - pad, t = Math.max(e, b) + pad;
        if (t - f > 60) { const mid = (a + b) / 2; f = mid - 30; t = mid + 30; }
        view = [Math.max(0, f), Math.min(dur, t)];
      }
      const crossing = () => cuts.filter((c) => c > a + 0.05 && c < b - 0.05).length;

      function drawFrames() {
        for (const [c, t] of [[fa, playT ?? a], [fb, Math.max(a, b - 1 / m.rate)]]) {
          if (!tw) continue;
          const w = c.clientWidth || 400, h = w * (th / tw), r = window.devicePixelRatio || 1;
          c.style.height = h + "px";
          if (c.width !== Math.round(w * r)) { c.width = Math.round(w * r); c.height = Math.round(h * r); }
          const x = c.getContext("2d");
          x.setTransform(r, 0, 0, r, 0, 0);
          x.imageSmoothingQuality = "high";
          pic(x, t, 0, 0, w, h);
        }
      }
      function drawAll() {
        const [x, w, h] = fit(all), X = (t) => (t / dur) * w;
        film(x, w, h, 0, dur);
        x.fillStyle = "rgba(244,240,232,.62)";
        for (let i = 0; i < bounds.length - 1; i++) {
          if (i === cur || i === hover) continue;
          x.fillRect(X(bounds[i]), 0, X(bounds[i + 1]) - X(bounds[i]), h);
        }
        x.fillStyle = "#fff";
        for (const c of cuts) x.fillRect(X(c) - 0.5, 0, 1, h);
        x.strokeStyle = MARK; x.lineWidth = 3;
        x.strokeRect(X(a) + 1, 1.5, Math.max(3, X(b) - X(a) - 2), h - 3);
      }
      function drawNear() {
        const [x, w, h] = fit(near), [f, t] = view, X = (v) => ((v - f) / (t - f)) * w;
        film(x, w, h, f, t);
        x.fillStyle = "rgba(244,240,232,.72)";
        x.fillRect(0, 0, X(a), h); x.fillRect(X(b), 0, w - X(b), h);
        x.fillStyle = "#fff";
        for (const c of cuts) if (c > f && c < t) x.fillRect(X(c) - 1, 0, 2, h);
        x.fillStyle = MARK;
        x.fillRect(X(a), 0, X(b) - X(a), 3); x.fillRect(X(a), h - 3, X(b) - X(a), 3);
        for (const v of [a, b]) {
          const p = X(v) + (v === a ? 0 : -9);
          x.fillStyle = MARK; x.fillRect(p, 0, 9, h);
          x.fillStyle = INK; x.fillRect(p + 3, h / 2 - 10, 1, 20); x.fillRect(p + 5, h / 2 - 10, 1, 20);
        }
        if (playT !== null) { x.fillStyle = "#fff"; x.fillRect(X(playT) - 1, 0, 2, h); }
      }
      function drawText() {
        time.innerHTML = `<b>${a.toFixed(1)} s → ${b.toFixed(1)} s</b> · ${(b - a).toFixed(1)} s · shot ${cur + 1} of ${bounds.length - 1}`;
        const n = crossing();
        note.className = n ? "sp-note sp-warn" : "sp-note";
        note.textContent = n ? (n === 1 ? "crosses a cut: SAM 2 will lose the figure" : `crosses ${n} cuts: SAM 2 will lose the figure`) : "";
      }
      function draw() { drawFrames(); drawAll(); drawNear(); drawText(); }

      const commit = () => { model.set("start", a); model.set("end", b); model.save_changes(); };
      function stop() {
        if (playing) cancelAnimationFrame(playing);
        playing = 0; playT = null; play.textContent = "play";
      }
      function pick(i) {
        stop();
        cur = i;
        let s = up10(bounds[i]), e = down10(bounds[i + 1]);
        if (e - s > m.longest) e = near10(s + m.longest);
        if (e - s < m.shortest) { e = near10(Math.min(dur, s + m.shortest)); s = near10(Math.max(0, e - m.shortest)); }
        a = s; b = e; setView(); draw(); commit();
      }
      function snap(t, which, perSec) {
        const reach = 6 / perSec;
        for (const c of cuts) if (Math.abs(c - t) <= reach) return which === "a" ? up10(c) : down10(c);
        return near10(t);
      }
      function setA(t) { a = near10(clamp(t, Math.max(0, b - m.longest), b - m.shortest)); }
      function setB(t) { b = near10(clamp(t, a + m.shortest, Math.min(down10(dur), a + m.longest))); }

      function showTip(e, c, t) {
        const r = c.getBoundingClientRect(), o = root.getBoundingClientRect();
        if (tw) {
          const i = clamp(Math.round(t * m.rate), 0, m.count - 1);
          tipPic.width = 320; tipPic.height = Math.round(320 * th / tw);
          tipPic.getContext("2d").drawImage(sheet, (i % m.cols) * tw, Math.floor(i / m.cols) * th, tw, th, 0, 0, tipPic.width, tipPic.height);
        }
        tipTime.textContent = `${t.toFixed(1)} s`;
        tip.style.display = "block";
        tip.style.left = clamp(e.clientX - o.left - 83, 0, o.width - 166) + "px";
        tip.style.top = r.top - o.top - tip.offsetHeight - 4 + "px";
      }
      const hideTip = () => { tip.style.display = "none"; };

      all.addEventListener("pointermove", (e) => {
        const r = all.getBoundingClientRect(), t = clamp(((e.clientX - r.left) / r.width) * dur, 0, dur);
        const i = shotAt(t);
        if (i !== hover) { hover = i; drawAll(); }
        showTip(e, all, t);
      });
      all.addEventListener("pointerleave", () => { hover = -1; drawAll(); hideTip(); });
      all.addEventListener("click", (e) => {
        const r = all.getBoundingClientRect();
        pick(shotAt(clamp(((e.clientX - r.left) / r.width) * dur, 0, dur)));
      });

      const nearT = (e) => {
        const r = near.getBoundingClientRect();
        return view[0] + clamp((e.clientX - r.left) / r.width, 0, 1) * (view[1] - view[0]);
      };
      near.addEventListener("pointerdown", (e) => {
        stop(); hideTip(); near.focus({ preventScroll: true }); near.setPointerCapture(e.pointerId);
        const r = near.getBoundingClientRect(), perSec = r.width / (view[1] - view[0]);
        const t = nearT(e), da = Math.abs(t - a) * perSec, db = Math.abs(t - b) * perSec;
        if (Math.min(da, db) <= 12) drag = da <= db ? "a" : "b";
        else if (t > a && t < b) drag = { off: t - a, len: b - a };
        else drag = t < a ? "a" : "b";
        if (typeof drag === "string") { side = drag; move(e); }
      });
      function move(e) {
        const r = near.getBoundingClientRect(), perSec = r.width / (view[1] - view[0]), t = nearT(e);
        if (drag === "a") setA(snap(t, "a", perSec));
        else if (drag === "b") setB(snap(t, "b", perSec));
        else { a = near10(clamp(t - drag.off, 0, down10(dur) - drag.len)); b = near10(a + drag.len); }
        cur = shotAt((a + b) / 2);
        drawFrames(); drawAll(); drawNear(); drawText();
      }
      near.addEventListener("pointermove", (e) => {
        if (drag) { move(e); return; }
        const perSec = near.getBoundingClientRect().width / (view[1] - view[0]), t = nearT(e);
        near.style.cursor = Math.min(Math.abs(t - a), Math.abs(t - b)) * perSec <= 12 ? "ew-resize" : t > a && t < b ? "grab" : "pointer";
        showTip(e, near, t);
      });
      near.addEventListener("pointerleave", hideTip);
      const release = () => { if (drag) { drag = null; setView(); draw(); commit(); } };
      near.addEventListener("pointerup", release);
      near.addEventListener("pointercancel", release);
      near.addEventListener("keydown", (e) => {
        if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
        e.preventDefault(); stop();
        const d = (e.key === "ArrowLeft" ? -1 : 1) * (e.shiftKey ? 1 : 0.1);
        if (side === "a") setA(a + d); else setB(b + d);
        cur = shotAt((a + b) / 2); draw();
        clearTimeout(saveTimer); saveTimer = setTimeout(commit, 350);
      });

      play.addEventListener("click", () => {
        if (playing) { stop(); draw(); return; }
        play.textContent = "stop";
        const t0 = performance.now();
        const tick = (now) => {
          playT = a + (((now - t0) / 1000) % (b - a));
          drawFrames(); drawNear();
          playing = requestAnimationFrame(tick);
        };
        playing = requestAnimationFrame(tick);
      });

      sheet.onload = () => {
        tw = sheet.naturalWidth / m.cols; th = sheet.naturalHeight / m.rows;
        setView(); draw();
      };
      sheet.src = url;
      const seen = new ResizeObserver(() => { if (tw) draw(); });
      seen.observe(root);
      drawText();
      return () => { stop(); clearTimeout(saveTimer); seen.disconnect(); URL.revokeObjectURL(url); };
    }
    export default { render };
    """
    sprite = traitlets.Bytes(b"").tag(sync=True)
    meta = traitlets.Dict({}).tag(sync=True)
    cuts = traitlets.List([]).tag(sync=True)
    start = traitlets.Float(0.0).tag(sync=True)
    end = traitlets.Float(0.0).tag(sync=True)


@app.class_definition
class BoxDrawer(anywidget.AnyWidget):
    """The frames of the shot, to draw boxes on. Move along the shot, drag on
    the picture to add a box on that frame. `boxes` are
    {"frame": k, "box": [x0, y0, x1, y1]} in the picture's own pixels. A frame
    is sent only when it is looked at."""

    _css = r"""
    .bx { color: #383b3e; max-width: 100%; }
    .bx-pic { position: relative; display: inline-block; max-width: 100%; line-height: 0;
      cursor: crosshair; user-select: none; background: #383b3e; min-height: 80px; min-width: 200px; }
    .bx-pic img { display: block; max-width: 100%; }
    .bx-pic canvas { position: absolute; left: 0; top: 0; width: 100%; height: 100%; touch-action: none; }
    .bx-bar, .bx-row { display: flex; align-items: center; gap: 8px; margin-top: 6px;
      font: 12px 'IBM Plex Mono', ui-monospace, monospace; }
    .bx-row { flex-wrap: wrap; }
    .bx-track { position: relative; flex: 1; min-width: 80px; padding-bottom: 9px; }
    .bx-track input { display: block; width: 100%; margin: 0; accent-color: #383b3e; }
    .bx-marks { position: absolute; left: 0; right: 0; bottom: 0; height: 8px; }
    .bx-marks button { position: absolute; top: 0; width: 7px; height: 8px; margin-left: -3px;
      padding: 0; border: 0; background: #f2d94e; outline: 1px solid #383b3e; cursor: pointer; }
    .bx button.bx-b { font: inherit; color: inherit; background: transparent; cursor: pointer;
      border: 1px solid #383b3e; border-radius: 3px; padding: 2px 8px; }
    .bx-time { white-space: nowrap; }
    .bx-note { flex: 1; min-width: 160px; opacity: .75; }
    """

    _esm = r"""
    const KEEP = 80;

    function render({ model, el }) {
      const count = model.get("count"), first = model.get("start"), fps = model.get("fps");
      el.innerHTML = `<div class="bx">
        <div class="bx-pic"><img draggable="false" alt="a frame of the shot"><canvas></canvas></div>
        <div class="bx-bar">
          <button type="button" class="bx-b" data-go="-1" title="previous frame">&lsaquo;</button>
          <div class="bx-track"><input type="range" min="0" max="${count - 1}" step="1" value="0" aria-label="frame of the shot"><div class="bx-marks"></div></div>
          <button type="button" class="bx-b" data-go="1" title="next frame">&rsaquo;</button>
          <span class="bx-time"></span>
        </div>
        <div class="bx-row"><span class="bx-note"></span>
          <button type="button" class="bx-b" data-clear="here">clear this frame</button>
          <button type="button" class="bx-b" data-clear="all">clear all</button>
        </div>
      </div>`;
      const img = el.querySelector("img"), canvas = el.querySelector("canvas"), ctx = canvas.getContext("2d");
      const range = el.querySelector("input"), marks = el.querySelector(".bx-marks");
      const time = el.querySelector(".bx-time"), note = el.querySelector(".bx-note");
      const pictures = new Map();
      let now = 0, shown = -1, asked = -1, drag = null;

      const boxes = () => model.get("boxes");
      function draw() {
        if (!img.naturalWidth) return;
        canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
        ctx.lineWidth = Math.max(2, canvas.width / 300);
        ctx.strokeStyle = "#f2d94e";
        const here = boxes().filter((b) => b.frame === shown).map((b) => b.box);
        if (drag) here.push([drag.a[0], drag.a[1], drag.b[0], drag.b[1]]);
        for (const [x0, y0, x1, y1] of here) ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      }
      function words() {
        const all = boxes(), onThis = all.filter((b) => b.frame === now).length;
        const frames = new Set(all.map((b) => b.frame)).size;
        const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "es"}`;
        note.textContent = all.length
          ? `${plural(onThis, "box")} here · ${plural(all.length, "box")} in all, on ${frames} frame${frames === 1 ? "" : "s"}`
          : "no boxes yet: drag on the picture";
        time.textContent = `${(first + now / fps).toFixed(2)} s · frame ${now + 1} of ${count}`;
      }
      function drawMarks() {
        marks.replaceChildren();
        for (const k of [...new Set(boxes().map((b) => b.frame))].sort((a, b) => a - b)) {
          const m = document.createElement("button");
          m.type = "button"; m.title = `boxes at ${(first + k / fps).toFixed(2)} s`;
          m.style.left = `calc(8px + (100% - 16px) * ${count > 1 ? k / (count - 1) : 0})`;
          m.addEventListener("click", () => go(k));
          marks.append(m);
        }
      }
      function show(k) {
        shown = k;
        img.src = pictures.get(k);
      }
      function ask(k) {
        if (asked >= 0) return;
        asked = k;
        model.send({ kind: "frame", k });
      }
      function go(k) {
        now = Math.min(Math.max(k, 0), count - 1);
        range.value = now;
        words();
        if (pictures.has(now)) show(now); else ask(now);
      }
      const onMessage = (msg, buffers) => {
        if (msg.kind !== "frame") return;
        const raw = buffers[0];
        const bytes = raw.buffer ? new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength) : raw;
        pictures.set(msg.k, URL.createObjectURL(new Blob([bytes], { type: "image/jpeg" })));
        if (pictures.size > KEEP) {
          const old = [...pictures.keys()].find((k) => k !== now);
          URL.revokeObjectURL(pictures.get(old)); pictures.delete(old);
        }
        asked = -1;
        if (pictures.has(now)) { if (shown !== now) show(now); } else ask(now);
      };
      model.on("msg:custom", onMessage);
      img.onload = draw;

      const at = (e) => {
        const r = canvas.getBoundingClientRect();
        return [(e.clientX - r.left) * img.naturalWidth / r.width, (e.clientY - r.top) * img.naturalHeight / r.height];
      };
      canvas.addEventListener("pointerdown", (e) => {
        if (shown !== now || !img.naturalWidth) return;
        drag = { a: at(e), b: at(e) };
        canvas.setPointerCapture(e.pointerId);
      });
      canvas.addEventListener("pointermove", (e) => { if (drag) { drag.b = at(e); draw(); } });
      const put = (list) => { model.set("boxes", list); model.save_changes(); changed(); };
      function changed() { drawMarks(); words(); draw(); }
      canvas.addEventListener("pointerup", () => {
        if (!drag) return;
        const [ax, ay] = drag.a, [bx, by] = drag.b;
        drag = null;
        if (Math.abs(bx - ax) > 4 && Math.abs(by - ay) > 4) {
          const box = [Math.min(ax, bx), Math.min(ay, by), Math.max(ax, bx), Math.max(ay, by)].map(Math.round);
          put([...boxes(), { frame: now, box }]);
        } else draw();
      });
      range.addEventListener("input", () => go(Number(range.value)));
      el.querySelectorAll("[data-go]").forEach((b) => b.addEventListener("click", () => go(now + Number(b.dataset.go))));
      el.querySelector('[data-clear="here"]').addEventListener("click", () => put(boxes().filter((b) => b.frame !== now)));
      el.querySelector('[data-clear="all"]').addEventListener("click", () => put([]));
      model.on("change:boxes", changed);
      drawMarks();
      go(boxes().length ? Math.min(...boxes().map((b) => b.frame)) : 0);
      return () => {
        model.off("msg:custom", onMessage); model.off("change:boxes", changed);
        for (const url of pictures.values()) URL.revokeObjectURL(url);
      };
    }
    export default { render };
    """
    count = traitlets.Int(0).tag(sync=True)
    start = traitlets.Float(0.0).tag(sync=True)
    fps = traitlets.Float(12.0).tag(sync=True)
    boxes = traitlets.List([]).tag(sync=True)

    def __init__(self, frames, start, fps, boxes):
        super().__init__(count=len(frames), start=start, fps=fps, boxes=boxes)
        self._frames = frames
        self.on_msg(self._on_ask)

    def _on_ask(self, _widget, msg, _buffers):
        """The page asks for a frame; it travels as a JPEG."""
        if msg.get("kind") == "frame":
            k = max(0, min(int(msg["k"]), len(self._frames) - 1))
            self.send({"kind": "frame", "k": k}, [frame_jpeg(self._frames[k])])


if __name__ == "__main__":
    app.run()
