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
    import base64
    import io
    import json
    import os
    import re
    import shutil
    import subprocess
    from dataclasses import asdict, replace
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
    video_upload = mo.ui.file(
        filetypes=[".mp4", ".mov", ".mkv", ".webm"],
        max_size=500_000_000,
        label="a video",
    )
    video_pick = mo.ui.dropdown(
        ["", *_local], value="", label="or one from clips/" if _local else "(no clips/ here)"
    )
    fps = mo.ui.dropdown({"12": 12, "24": 24}, value="12", label="frames a second")
    width = mo.ui.dropdown(
        {"960": 960, "1280": 1280, "1920": 1920}, value="1280", label="width (px)"
    )
    mo.vstack(
        [
            mo.hstack([video_upload, video_pick], justify="start"),
            mo.hstack([fps, width], justify="start"),
        ]
    )
    return fps, video_pick, video_upload, width


@app.cell
def _(mo, video_pick, video_upload):
    # The source video on disk (an upload is kept in masks/cutout/<name>/).
    if video_upload.value:
        name = slug(Path(video_upload.name()).stem)
        (WORK / name).mkdir(parents=True, exist_ok=True)
        source = WORK / name / f"source{Path(video_upload.name()).suffix.lower()}"
        if not source.exists() or source.stat().st_size != len(video_upload.contents()):
            source.write_bytes(video_upload.contents())
    elif video_pick.value:
        source = CLIPS / video_pick.value
        name = slug(source.stem)
    else:
        mo.stop(True, mo.md("*Pick a video to start.*"))
    duration = video_duration(source)
    return duration, name, source


@app.cell
def _(duration, mo):
    start = mo.ui.number(0.0, round(duration, 1), step=0.1, value=0.0, label="start (s)")
    length = mo.ui.number(
        0.5, 20.0, step=0.5, value=max(0.5, min(3.0, duration // 0.5 * 0.5)), label="length (s)"
    )
    mo.hstack([start, length, mo.md(f"<small>of {duration:.1f} s</small>")], justify="start")
    return length, start


@app.cell
def _(duration, length, mo, name, source, start, fps, width):
    # Each shot gets its own folder, so nothing made before is overwritten.
    shot = (start.value, min(start.value + length.value, duration))
    work = WORK / name / f"{shot[0]:.1f}-{shot[1]:.1f}s_{fps.value}fps_{width.value}px"
    mo.vstack(
        [
            mo.md(
                "The first and the last frame of the shot. Keep it to one shot: "
                "SAM 2 cannot follow a figure across a cut."
            ),
            mo.hstack(
                # the last frame starts one frame before the end: at the end there is none
                [
                    frame_at(source, t, width.value // 3)
                    for t in (shot[0], max(shot[0], shot[1] - 1 / fps.value))
                ],
                justify="start",
            ),
        ]
    )
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
    figure on the first frame. BiRefNet proposes them; drag on the picture
    to add one, *clear* to start over. A part with no box is not followed.
    Back on a shot already cut out, the boxes are the ones you used.
    """)
    return


@app.cell
def _(frames, mo, work):
    # The boxes of the last cut-out of this shot, else BiRefNet's.
    _record = work / "masks.json"
    _last = json.loads(_record.read_text()).get("boxes") if _record.exists() else None
    if _last:
        start_boxes = _last
    else:
        with mo.status.spinner(title="BiRefNet looks for the figure"):
            start_boxes = sam2_prompts(frames[0], Cutout())
    boxer = mo.ui.anywidget(BoxDrawer(src=data_url(frames[0]), boxes=start_boxes))
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
    ink = mo.ui.slider(
        0.0, 0.6, step=0.05, value=Cutout().ink, label="line art: ink recovered (0 = off)"
    )
    cut_run = mo.ui.run_button(label="cut out")
    mo.vstack([method, ink, cut_run])
    return cut_run, ink, method


@app.cell
def _(boxer, cut_run, frames, ink, method, mo, work):
    # Raw masks from the model(s), cleaned and finished as tools/masks.py does,
    # saved in masks/<name>/: %05d.png the text avoids, rgba/ the cut-out.
    _boxes = [list(map(int, b)) for b in boxer.value["boxes"]]
    cut = replace(Cutout(), ink=ink.value)
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
            _title = "SAM 2 follows the figure" if _m == "sam2" else "BiRefNet, frame by frame"
            _gen = (
                sam2_masks(work / "clip.mp4", _boxes) if _m == "sam2" else rembg_masks(_m, frames)
            )
            _raws.append(list(mo.status.progress_bar(_gen, total=len(frames), title=_title)))
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
def frame_at(path, t, width):
    """One frame of a video at t seconds, `width` px wide."""
    cmd = [ffmpeg(), "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1"]
    cmd += ["-vf", f"scale={width}:-2", "-f", "image2pipe", "-c:v", "png", "-"]
    return Image.open(io.BytesIO(subprocess.run(cmd, capture_output=True, check=True).stdout))


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
def data_url(path, width=None):
    """A picture as a data: URL, for the box drawer."""
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


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
class BoxDrawer(anywidget.AnyWidget):
    """A picture you draw boxes on: drag to add one, "clear" to remove them
    all. `boxes` are [x0, y0, x1, y1] in the picture's own pixels."""

    _esm = r"""
    function render({ model, el }) {
      const wrap = document.createElement("div");
      wrap.style.cssText = "position:relative;display:inline-block;cursor:crosshair;user-select:none;max-width:100%";
      const img = document.createElement("img");
      img.draggable = false;
      img.style.cssText = "display:block;max-width:100%";
      const canvas = document.createElement("canvas");
      canvas.style.cssText = "position:absolute;left:0;top:0;width:100%;height:100%;touch-action:none";
      wrap.append(img, canvas);
      const clear = document.createElement("button");
      clear.textContent = "clear";
      clear.style.cssText = "margin-top:6px;font-family:'IBM Plex Mono',monospace";
      el.append(wrap, clear);
      let drag = null;
      const ctx = canvas.getContext("2d");
      function draw() {
        canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.lineWidth = Math.max(2, canvas.width / 300);
        ctx.strokeStyle = "#f2d94e";
        const all = [...model.get("boxes")];
        if (drag) all.push([drag.a[0], drag.a[1], drag.b[0], drag.b[1]]);
        for (const [x0, y0, x1, y1] of all) ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      }
      function at(e) {
        const r = canvas.getBoundingClientRect();
        return [(e.clientX - r.left) * img.naturalWidth / r.width,
                (e.clientY - r.top) * img.naturalHeight / r.height];
      }
      canvas.addEventListener("pointerdown", (e) => {
        drag = { a: at(e), b: at(e) };
        canvas.setPointerCapture(e.pointerId);
      });
      canvas.addEventListener("pointermove", (e) => { if (drag) { drag.b = at(e); draw(); } });
      canvas.addEventListener("pointerup", () => {
        if (!drag) return;
        const [ax, ay] = drag.a, [bx, by] = drag.b;
        drag = null;
        if (Math.abs(bx - ax) > 4 && Math.abs(by - ay) > 4) {
          const box = [Math.min(ax, bx), Math.min(ay, by), Math.max(ax, bx), Math.max(ay, by)].map(Math.round);
          model.set("boxes", [...model.get("boxes"), box]);
          model.save_changes();
        }
        draw();
      });
      clear.addEventListener("click", () => { model.set("boxes", []); model.save_changes(); });
      img.onload = draw;
      img.src = model.get("src");
      model.on("change:src", () => { img.src = model.get("src"); });
      model.on("change:boxes", draw);
    }
    export default { render };
    """
    src = traitlets.Unicode("").tag(sync=True)
    boxes = traitlets.List([]).tag(sync=True)


if __name__ == "__main__":
    app.run()
