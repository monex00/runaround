# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "numpy", "scipy", "pillow"]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="Runaround",
    css_file="notebook.css",
)

with app.setup:
    # Everything defined here and in @app.function / @app.class_definition cells
    # is importable:  from runaround import Page, free_spans, fill, ...
    # render.py and the tests import from this file. It is the single source
    # of truth for the page, the runaround and the look.
    import functools
    import io
    import itertools
    import json
    import tomllib
    import urllib.request
    from dataclasses import dataclass, replace
    from pathlib import Path

    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from scipy import ndimage

    ROOT = Path(__file__).parent
    PIECES = ROOT / "pieces.toml"
    # Fonts, texts and demo figures sit at the top of the private repo and under
    # public/ in the published one, which GitHub Pages serves next to the page.
    BASE = ROOT / "public" if (ROOT / "public" / "fonts").exists() else ROOT
    FONTS = BASE / "fonts"
    MASKS = ROOT / "masks"
    TEXTS = BASE / "texts"
    DEMO = BASE / "demo"  # light figures for the notebook (tools/make_demo.py)
    # Where a server copy of the notebook (molab) finds what it was not given.
    PUBLISHED = "https://raw.githubusercontent.com/monex00/runaround/main/"

    # Two palettes, paper and night, shared by the author's math-art projects.
    # Paper and ink stay monochrome; only the figure keeps its own colours.
    PAPER = dict(bg="#f4f0e8", ink="#383b3e")
    NIGHT = dict(bg="#1d252b", ink="#fbfcfc")
    PALETTES = dict(paper=PAPER, night=NIGHT)

    # Bottom-right signature on reels; None leaves them unsigned.
    SIGNATURE = "monex00, 2026"


# ------------------------------------------------------------- the notebook


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Runaround

    A page of newspaper text with a figure living inside it. Every frame the
    text gets out of the figure's way: each line is cut where the figure is
    and poured into what is left, so the columns flow around the silhouette.
    Typesetters call text wrapped around a picture a *runaround*. Here the
    picture never stops moving.

    **Play with it first.** Pick a figure, or drop in your own cut-out, and a
    text: the loop below plays them as a reel would. Change the lines, the
    margin or the type and press *make the loop* to see your own page move;
    it is drawn right here, in your browser, so it takes a little while. How
    it works is further down.
    """)
    return


@app.cell
def _(mo):
    notebook = Notebook()
    _titles = {load_demo(n)[0]["title"]: n for n in demo_names()}
    _local = {f"{load_pieces()[n]['title']} (local)": f"local:{n}" for n in local_names()}
    figure_pick = mo.ui.dropdown(
        {**_titles, **_local, "A shape (no clip)": "shape"},
        value="Tango" if "Tango" in _titles else next(iter(_titles), "A shape (no clip)"),
        label="figure",
    )
    upload = mo.ui.file(
        filetypes=[".png", ".webp", ".gif"],
        label="your own cut-out (PNG, WebP or GIF)",
    )
    text_pick = mo.ui.dropdown(
        {
            "The figure's own": "figure",
            "Love poems (public domain)": "love.txt",
            "So far, so good": "haine.txt",
            "The thaw": "spring.txt",
            "My own text": "",
        },
        value="The figure's own",
        label="text",
    )
    own_text = mo.ui.text_area(
        placeholder="Paste your text here (a blank line starts a paragraph)", full_width=True
    )
    mode = mo.ui.radio(
        {"break anywhere": "char", "whole words": "word"},
        value="break anywhere",
        label="lines",
        inline=True,
    )
    margin = mo.ui.slider(0.0, 2.0, step=0.1, value=Runaround().margin, label="margin (em)")
    size = mo.ui.slider(18, 44, step=1, value=Type().size, label="type size")
    printed = mo.ui.checkbox(value=True, label="print look")
    palette = mo.ui.dropdown(["paper", "night"], value="paper", label="palette")
    length = mo.ui.dropdown(
        {"4 seconds": 4, "8 seconds": 8, "the whole clip": 0}, value="4 seconds", label="length"
    )
    make = mo.ui.run_button(label="make the loop")
    return (
        figure_pick,
        length,
        make,
        margin,
        mode,
        notebook,
        own_text,
        palette,
        printed,
        size,
        text_pick,
        upload,
    )


@app.cell
def _(figure_pick, notebook, upload):
    # Every figure is a sequence of (cut-out, mask) with where it sits on the
    # page, its look, its rate, its own text and the frames the sections study.
    _print = {**load_demo(demo_names()[0])[0]["look"], "figure": "colour"}
    figure_demo, figure_text, figure_window = None, None, 0
    figure_fps = notebook.fps
    if upload.value:
        figure = frames_from_upload(upload.contents())
        figure_place = Place(scale=0.9, bottom=0.78)
        figure_look = _print
        figure_credit = upload.name()
    elif figure_pick.value == "shape":
        figure = shape_frames(Page())
        figure_place = Place(scale=1.0, cx=0.5, bottom=1.0)
        figure_look = _print
        figure_credit = "a synthetic shape"
    elif figure_pick.value.startswith("local:"):
        _slug = figure_pick.value[6:]
        _info, figure, figure_place = load_local(_slug)
        figure_look = _info["look"]
        figure_credit = _info["credit"]
        figure_text = load_pieces()[_slug].get("text")
    else:
        figure_demo = figure_pick.value
        _info, figure = load_demo(figure_demo)
        figure_place = Place(**_info["place"])
        figure_look = _info["look"]
        figure_credit = _info["credit"]
        figure_fps = _info["fps"]
        figure_text = _info["text"]
        figure_window = _info.get("window", 0)
    return (
        figure,
        figure_credit,
        figure_demo,
        figure_fps,
        figure_look,
        figure_place,
        figure_text,
        figure_window,
    )


@app.cell
def _(
    figure_credit,
    figure_look,
    figure_pick,
    figure_text,
    length,
    make,
    margin,
    mo,
    mode,
    own_text,
    palette,
    printed,
    size,
    text_pick,
    upload,
):
    # The settings, and the controls that set them. Nothing here depends on
    # time: the loop below and the sections read these values.
    page_now = Page()
    type_now = Type(size=float(size.value), mode=mode.value, justify=mode.value == "word")
    runaround_now = Runaround(margin=margin.value)
    if text_pick.value == "figure":
        text_now = asset_text(f"texts/{figure_text or 'love.txt'}")
    elif text_pick.value:
        text_now = asset_text(f"texts/{text_pick.value}")
    else:
        text_now = own_text.value or "Paste a text above."
    if printed.value:
        look_now = Look(**{**figure_look, "palette": palette.value})
    else:
        look_now = Look(palette=palette.value)
    mo.vstack(
        [
            mo.hstack([figure_pick, upload], justify="start"),
            mo.hstack([text_pick, mode, palette, printed], justify="start"),
            mo.md("") if text_pick.value else own_text,
            mo.hstack([margin, size], justify="start"),
            mo.hstack([length, make], justify="start"),
            mo.md(f"<small>Figure: {figure_credit}.</small>"),
        ]
    )
    return look_now, page_now, runaround_now, text_now, type_now


@app.cell
def _(
    figure,
    figure_demo,
    figure_fps,
    figure_look,
    figure_place,
    figure_text,
    length,
    look_now,
    make,
    mo,
    notebook,
    page_now,
    runaround_now,
    text_now,
    type_now,
):
    # The loop. A demo with the settings it starts with plays the loop that
    # ships with it (tools/make_demo.py draws it with make_loop, as below);
    # anything else is drawn here when you press "make the loop".
    _width = round(page_now.width * notebook.scale)
    _shipped = (
        figure_demo is not None
        and type_now == Type()
        and runaround_now == Runaround()
        and look_now == Look(**figure_look)
        and text_now == asset_text(f"texts/{figure_text}")
    )
    if make.value:
        _n = len(figure) if not length.value else min(len(figure), round(length.value * figure_fps))
        _frames = make_loop(
            text_now,
            figure,
            figure_place,
            page_now,
            type_now,
            runaround_now,
            look_now,
            range(_n),
            notebook.scale,
            progress=lambda ks: mo.status.progress_bar(
                ks, title="Making the loop", subtitle=f"{_n} frames", remove_on_exit=True
            ),
        )
        _view = mo.image(loop_webp(_frames, figure_fps, notebook.quality), width=_width)
    elif _shipped:
        _view = mo.video(
            asset_bytes(f"demo/{figure_demo}/loop.mp4"),
            autoplay=True,
            loop=True,
            muted=True,
            controls=False,
            width=_width,
        )
    else:
        _still = make_loop(
            text_now,
            figure,
            figure_place,
            page_now,
            type_now,
            runaround_now,
            look_now,
            [0],
            notebook.scale,
        )[0]
        _view = mo.vstack(
            [
                mo.md(
                    "*New settings: this is the first frame. Press **make the loop** to see it move.*"
                ),
                mo.image(Image.fromarray(_still), width=_width),
            ]
        )
    _view
    return


@app.cell
def _(
    figure,
    figure_place,
    figure_window,
    notebook,
    page_now,
    runaround_now,
    text_now,
    type_now,
):
    # The stretch the sections study: `notebook.window` frames from the
    # figure's window (for the demos, the two seconds in which the text moves
    # most), laid out with the settings above. `still` is the frame in the
    # middle, the one the single pictures show.
    _start = figure_window if figure_window + 1 < len(figure) else 0
    study_frames = list(range(_start, min(_start + notebook.window, len(figure))))
    _placed = [figure_on_page(figure[k], page_now, place_at(figure_place, k)) for k in study_frames]
    study_alphas = [a for a, _ in _placed]
    study_figures = [f for _, f in _placed]
    study_smooth, study_bare = breathing(text_now, study_alphas, page_now, type_now, runaround_now)
    still = len(study_frames) // 2
    return still, study_alphas, study_bare, study_figures, study_frames, study_smooth


@app.cell
def _(figure_fps, mo, study_frames):
    mo.md(rf"""
    ## 1. The cut-out

    Everything starts from a figure without its background: for every frame,
    a *mask*, white where the figure is and black elsewhere. The demo figures
    were cut out of their films with two segmentation models, BiRefNet and
    SAM 2 (`tools/masks.py`); a drawn figure brings its own transparency. The
    notebook never runs a model: it only reads masks.

    From here on the pictures look at {len(study_frames)} frames of the
    figure you picked (frames {study_frames[0]} to {study_frames[-1]} of the
    clip, {len(study_frames) / figure_fps:.1f} s), with the settings above;
    for the demos, the stretch where the text moves most. Left, the figure as it is drawn on
    the page in the middle of it; right, the mask the text keeps away from.
    """)
    return


@app.cell
def _(mo, notebook, still, study_alphas, study_figures):
    _mask = Image.fromarray((study_alphas[still] * 255).astype(np.uint8)).convert("RGB")
    _cut = Image.new("RGBA", study_figures[still].size, PAPER["bg"])
    _cut.alpha_composite(study_figures[still])
    _size = (round(_mask.width * notebook.small), round(_mask.height * notebook.small))
    mo.hstack([_cut.convert("RGB").resize(_size), _mask.resize(_size)], justify="start")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. The page

    A newspaper page is a grid: two columns, a margin all round, and in each
    column a stack of lines one under the other at a fixed distance, the
    *leading*. Every letter sits on one of those lines. The first letter of the
    text is set big, three lines tall: the *drop cap*. The grid follows the
    type size above, and nothing else: it does not care about the figure.
    """)
    return


@app.cell
def _(notebook, page_now, type_now):
    grid_preview(page_now, type_now, scale=notebook.small)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Keeping away

    The mask is grown by a margin, so the text keeps a little air around the
    figure: a pixel $p$ is *blocked* when its distance to the figure is

    $$d(p) \le m \cdot s,$$

    with $m$ the margin in ems and $s$ the type size. A line is blocked where
    any pixel of its height is blocked, so no letter ever touches the figure;
    what is left of each line are the *free spans*, on either side. In grey
    below, what is blocked in the middle frame.

    A figure trembles a little from frame to frame, and a span that moved by one
    pixel would push a letter onto the next line, and all the text after it.
    Two guards stop that: span edges move in steps of a quarter of an em
    (*snapped*), and a blocked pixel is only given back when the figure has
    moved half an em further away (*latched*).

    The chart follows one line over the whole stretch, the one where the
    guards have most to do: where its text stops, to the pixel, then
    snapped, then snapped and latched, as the reels do. A flat stretch is a
    line that holds still. Under it, one bar per frame: how many lines get
    new spans and have their text poured again, with no guard and with both.
    The guards do not stop the text from breathing when the figure really
    moves; they stop it from shivering when it does not.
    """)
    return


@app.cell
def _(
    mo,
    notebook,
    page_now,
    runaround_now,
    still,
    study_bare,
    study_smooth,
    type_now,
):
    _blocked = preview(
        study_smooth[still][0],
        page_now,
        type_now,
        blocked=study_smooth[still][1],
        scale=notebook.small,
    )
    if len(study_smooth) < 2:
        _chart = mo.md("*A single frame: nothing moves, nothing to guard against.*")
    else:
        _snap = runaround_now.snap * type_now.size
        _col, _line = guarded_line(
            [b for _, b in study_bare], [b for _, b in study_smooth], page_now, type_now, _snap
        )

        def _edges(seq, snap):
            return [text_edge(b, page_now, type_now, _col, _line, snap) for _, b in seq]

        def _poured(seq):
            return [lines_changed(a, b) for (a, _), (b, _) in itertools.pairwise(seq)]

        _bare, _both = _poured(study_bare), _poured(study_smooth)
        _chart = mo.vstack(
            [
                breath_chart(
                    {
                        "to the pixel": _edges(study_bare, 0.0),
                        "snapped": _edges(study_bare, _snap),
                        "snapped+latched": _edges(study_smooth, _snap),
                    },
                    {"no guard": _bare, "both guards": _both},
                    notebook.chart,
                ),
                mo.md(
                    f"The edges follow line {_line + 1} of column {_col + 1}. From one "
                    f"frame to the next, **{np.mean(_bare):.0f}** lines are poured again "
                    f"with no guard, **{np.mean(_both):.0f}** with both."
                ),
            ]
        )
    mo.hstack([_blocked, _chart], justify="start", align="center")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Pouring the text

    The text is one long stream of letters that never changes. Each frame it is
    poured into the free spans in reading order: down the first column, the
    left span before the right one, then the second column. Left, **break
    anywhere**: a letter that does not fit goes to the next span, so words break
    wherever the figure is and the silhouette stays sharp. Right, **whole
    words**: words move whole, a gap too narrow for the next word stays empty.
    """)
    return


@app.cell
def _(
    notebook,
    page_now,
    runaround_now,
    still,
    study_alphas,
    text_now,
    type_now,
):
    _a, _ = layout(
        text_now,
        study_alphas[still],
        page_now,
        replace(type_now, mode="char", justify=False),
        runaround_now,
    )
    _b, _ = layout(
        text_now,
        study_alphas[still],
        page_now,
        replace(type_now, mode="word", justify=True),
        runaround_now,
    )
    _left = preview(_a, page_now, type_now, scale=notebook.small)
    _right = preview(_b, page_now, type_now, scale=notebook.small)
    _both = Image.new("RGB", (_left.width * 2 + 12, _left.height), "white")
    _both.paste(_left, (0, 0))
    _both.paste(_right, (_left.width + 12, 0))
    _both
    return


@app.cell
def _(mo, notebook):
    mo.md(rf"""
    When the figure moves, one span changes and everything after it shifts:
    that is the text breathing. Below, {notebook.strip} moments of the
    stretch, {notebook.strip_step} frames apart; in yellow, the letters that
    changed line since the picture before.
    """)
    return


@app.cell
def _(
    look_now,
    notebook,
    page_now,
    study_figures,
    study_smooth,
    type_now,
):
    # The pictures go where the text moves most within the stretch.
    _step, _lays = notebook.strip_step, [lay for lay, _ in study_smooth]
    _span = _step * (notebook.strip - 1)
    _starts = list(range(_step, len(_lays) - _span)) or [0]

    def _score(first):
        return sum(
            moved(_lays[k - _step], _lays[k])
            for k in range(max(first, _step), first + _span + 1, _step)
        )

    _first = max(_starts, key=_score)
    _picks = [k for k in range(_first, len(_lays), _step)][: notebook.strip]
    filmstrip(
        _lays,
        study_figures,
        page_now,
        type_now,
        _picks,
        _step,
        notebook.strip_scale,
        mark=look_now.highlight,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. The print look

    The reels imitate cheap print: a fine grain that changes every frame and is
    heavier on the figure, and the figure's red and blue plates printed a hair
    out of register, which leaves a coloured fringe on its edges. A
    black-and-white cartoon can be printed in the page's own ink instead of its
    greys, as the tango is. Every number lives in the `Look` dataclass. Left,
    the clean page; right, the print look, close up, at the reel's own size.
    """)
    return


@app.cell
def _(
    figure_look,
    mo,
    notebook,
    page_now,
    palette,
    still,
    study_alphas,
    study_figures,
    study_frames,
    study_smooth,
    type_now,
):
    _alpha = study_alphas[still]
    _ys, _xs = np.nonzero(_alpha >= 0.5)
    # Centred on the figure's left edge: half text, half figure.
    _cx = int(_xs.min()) if len(_xs) else page_now.width // 2
    _cy = int(_ys.mean()) if len(_ys) else page_now.height // 2
    _h = notebook.crop // 2
    _x0 = min(max(_cx - _h, 0), page_now.width - notebook.crop)
    _y0 = min(max(_cy - _h, 0), page_now.height - notebook.crop)
    _box = (_x0, _y0, _x0 + notebook.crop, _y0 + notebook.crop)
    _crops = [
        Image.fromarray(
            render_frame(
                study_smooth[still][0],
                page_now,
                type_now,
                _look,
                figure=study_figures[still],
                rng=np.random.default_rng(study_frames[still]),
                region=_alpha,
            )
        ).crop(_box)
        for _look in (
            Look(palette=palette.value),
            Look(**{**figure_look, "palette": palette.value}),
        )
    ]
    mo.hstack(_crops, justify="start")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Your own figure

    Cutting a figure out of a video takes two segmentation models and a GPU,
    more than a browser page can run. A second notebook does it on
    [molab](https://molab.marimo.io), marimo's free cloud, which lends you
    the GPU:

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/monex00/runaround/blob/main/cutout.py)

    Pick a video and a shot, check the boxes around the figure, and it gives
    the cut-out back as an animated WebP. Drop that file into *your own
    cut-out* at the top of this page, and the text flows around your figure.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Appendix: the code

    Everything below is the library the playground and the sections above use,
    and that `tools/render.py` and the tests import from this same file. Every
    number the drawing uses lives in one of its dataclasses.
    """)
    return


@app.function
def load_pieces(path=PIECES):
    """The pieces of the series, from pieces.toml, keyed by slug."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return {p["slug"]: p for p in data.get("piece", [])}


@app.function
@functools.cache
def asset_bytes(rel):
    """The bytes of an asset (fonts/..., texts/..., demo/...): from disk, or,
    when the notebook runs in the browser, over HTTP from the public/ folder
    served next to the page. A server copy missing the file (molab copies
    public/ without demo/) takes it from the published repository."""
    local = BASE / rel
    if local.exists():
        return local.read_bytes()
    import marimo as mo

    here = str(mo.notebook_location())
    base = here if here.startswith(("http://", "https://")) else PUBLISHED
    with urllib.request.urlopen(f"{base.rstrip('/')}/public/{rel}") as f:
        return f.read()


@app.function
def asset_text(rel):
    """An asset as UTF-8 text."""
    return asset_bytes(rel).decode("utf-8")


@app.function
def use_fonts():
    """Instrument Serif (the voice) and IBM Plex Mono (labels, numbers), from
    fonts/ (SIL OFL), registered with matplotlib. Returns their names."""
    from matplotlib import font_manager

    for f in sorted(FONTS.glob("*.ttf")):
        font_manager.fontManager.addfont(str(f))
    return dict(serif="Instrument Serif", mono="IBM Plex Mono")


# ------------------------------------------------------------- parameters
# Every number the drawing uses lives in one of these dataclasses.


@app.class_definition
@dataclass(frozen=True)
class Cutout:
    """How a raw mask from a segmentation model is cleaned.

    A model's alpha has specks of background it was unsure about, small holes
    inside the figure and a slight shimmer from frame to frame. The figure may
    be made of several parts (a girl and her dog), so we keep every part that
    is big enough, not only the largest."""

    threshold: float = 0.5  # alpha above this counts as figure
    min_part: float = 0.02  # a part smaller than this share of the biggest one is a speck
    max_hole: float = 0.0005  # holes smaller than this share of the frame are filled
    edge_px: int = 2  # soft edge kept around each part, in pixels
    window: int = 3  # frames in the temporal median (odd; 1 = off)
    thin: int = 0  # parts thinner than twice this many pixels (a painted horizon line) are dropped
    prompt_part: float = 0.1  # SAM 2 follows only parts this big next to the biggest
    # Line-art clips (dark outlines on a light background): pixels darker than
    # `ink` near the model's mask join it, and whatever the outlines enclose is
    # filled. Recovers legs, tails and white gloves a model drops. 0 = off.
    ink: float = 0.0  # darkness threshold, 0..1 of full white
    ink_reach: int = 24  # how far from the model's mask ink is taken, in pixels
    ink_close: int = 3  # gaps in the outlines this wide are closed, in pixels
    ink_open: int = 0  # strokes thinner than twice this, enclosing nothing, are dropped (0 = keep)
    # The mask the text avoids can be fuller than the cut-out that is drawn:
    # gaps up to `solidify` pixels are closed and every hole filled, so a
    # figure a model broke into pieces never lets text in. 0 = off.
    solidify: int = 0
    # Frames on each side whose (solid) masks are joined into the mask the
    # text avoids, so a single frame where the model fails leaves no hole.
    hold: int = 0
    # Light near the figure joins it (a spark between two heads): pixels whose
    # brightest channel is above `glow` (0..1), within `ink_reach` px. 0 = off.
    glow: float = 0.0
    # The drawn cut-out may also be closed (gaps up to `draw_close` px), so a
    # railing in front does not slice the figure into strips. 0 = off.
    draw_close: int = 0
    # The drawn cut-out may borrow from `draw_hold` frames on each side, so a
    # thin prop the model drops for a few frames (a gun) does not blink. 0 = off.
    draw_hold: int = 0


@app.function
def clean_mask(alpha, cut=None):
    """One frame. alpha: float array (H, W) in [0, 1]. Returns the cleaned
    alpha, same shape: specks dropped, small holes filled, soft edges kept."""
    cut = cut or Cutout()
    alpha = np.asarray(alpha, dtype=np.float32)
    solid = alpha >= cut.threshold
    if cut.thin:
        solid = ndimage.binary_opening(solid, iterations=cut.thin)
    labels, n = ndimage.label(solid)
    if n == 0:
        return np.zeros_like(alpha)
    areas = ndimage.sum_labels(solid, labels, index=np.arange(1, n + 1))
    keep = np.flatnonzero(areas >= cut.min_part * areas.max()) + 1
    parts = np.isin(labels, keep)

    # Holes are background pockets enclosed by the figure. Small ones are
    # model noise; big ones are real (the sky between an arm and a staff).
    holes = ndimage.binary_fill_holes(parts) & ~parts
    hole_labels, m = ndimage.label(holes)
    if m:
        hole_areas = ndimage.sum_labels(holes, hole_labels, index=np.arange(1, m + 1))
        small = np.flatnonzero(hole_areas < cut.max_hole * alpha.size) + 1
        parts |= np.isin(hole_labels, small)

    near = ndimage.binary_dilation(parts, iterations=cut.edge_px) if cut.edge_px else parts
    out = np.where(near, alpha, 0.0).astype(np.float32)
    out[parts] = 1.0  # solid inside, filled holes included; soft only at the rim
    return out


@app.function
def glow_fill(alpha, rgb, cut=None):
    """One frame. Adds bright light (sparks, glows) near the model's mask:
    pixels whose brightest channel is above `cut.glow`, within `cut.ink_reach`
    pixels of the mask. Off when `cut.glow` is 0."""
    cut = cut or Cutout()
    alpha = np.asarray(alpha, dtype=np.float32)
    if not cut.glow:
        return alpha
    near = ndimage.binary_dilation(alpha >= cut.threshold, iterations=cut.ink_reach)
    light = (np.asarray(rgb, np.float32).max(axis=2) / 255.0 >= cut.glow) & near
    return np.maximum(alpha, light.astype(np.float32))


@app.function
def with_variant(piece, name):
    """The piece with one of its [piece.variants.<name>] laid over it: a
    table (music, look...) is merged key by key, anything else replaced."""
    over = piece.get("variants", {}).get(name)
    if over is None:
        raise SystemExit(f"{piece['slug']} has no variant {name!r}")
    out = dict(piece)
    for k, v in over.items():
        out[k] = {**piece.get(k, {}), **v} if isinstance(v, dict) else v
    return out


@app.function
def segment_place(piece, k):
    """The piece's Place with segment k's overrides: [piece.places."<k>"]
    (a press photo shown smaller than the cut-out figures, for instance)."""
    place = piece_params(piece)["place"]
    over = piece.get("places", {}).get(str(k), {})
    return replace(place, **over) if over else place


@app.function
def segment_of(i, spans):
    """Index of the segment frame i belongs to."""
    for k, (first, count) in enumerate(spans):
        if first <= i < first + count:
            return k
    return len(spans) - 1


@app.function
def segment_cutout(piece, k):
    """The piece's Cutout with segment k's overrides:
    [piece.cutouts."<k>"] in pieces.toml, same field names."""
    cut = piece_params(piece)["cutout"]
    over = piece.get("cutouts", {}).get(str(k), {})
    return replace(cut, **over) if over else cut


@app.function
def ink_fill(alpha, gray, cut=None):
    """One frame of a line-art clip. alpha: the model's mask, gray: the
    frame in [0, 1]. Adds the dark strokes near the mask and fills every
    region the strokes enclose. Off when `cut.ink` is 0."""
    cut = cut or Cutout()
    alpha = np.asarray(alpha, dtype=np.float32)
    if not cut.ink:
        return alpha
    solid = alpha >= cut.threshold
    near = ndimage.binary_dilation(solid, iterations=cut.ink_reach)
    ink = (np.asarray(gray) < cut.ink) & near
    shape = solid | ink
    if cut.ink_close:
        shape = ndimage.binary_closing(shape, iterations=cut.ink_close)
    shape = ndimage.binary_fill_holes(shape)
    if cut.ink_open:  # thin strokes that enclose nothing (a floor line) are dropped
        shape = ndimage.binary_opening(shape, iterations=cut.ink_open) | solid
    return np.maximum(alpha, shape.astype(np.float32))


@app.function
def solidify(alpha, cut=None):
    """The mask the text keeps away from: `alpha` with gaps up to
    `cut.solidify` pixels closed and all holes filled. The drawn cut-out
    keeps the finer `alpha`. Off when `cut.solidify` is 0."""
    cut = cut or Cutout()
    alpha = np.asarray(alpha, dtype=np.float32)
    if not cut.solidify:
        return alpha
    shape = alpha >= cut.threshold
    r = cut.solidify
    shape = np.pad(shape, r)  # so closing does not stick to the frame edge
    shape = ndimage.binary_closing(shape, iterations=r)[r:-r, r:-r]
    shape = ndimage.binary_fill_holes(shape)
    return np.maximum(alpha, shape.astype(np.float32))


@app.function
def hold_masks(stack, cut=None):
    """Many frames. Each mask joined with `cut.hold` frames before and after
    (a running maximum): one bad frame cannot open a hole in the figure."""
    cut = cut or Cutout()
    stack = np.asarray(stack, dtype=np.float32)
    if not cut.hold or len(stack) < 2:
        return stack
    return ndimage.maximum_filter(stack, size=(2 * cut.hold + 1, 1, 1), mode="nearest")


@app.function
def smooth_masks(stack, cut=None):
    """Many frames. stack: float array (T, H, W). A running median over
    `cut.window` frames removes one-frame flickers of the edge without
    blurring motion (a median keeps edges, a mean would smear them)."""
    cut = cut or Cutout()
    stack = np.asarray(stack, dtype=np.float32)
    if cut.window <= 1 or len(stack) < 2:
        return stack
    return ndimage.median_filter(stack, size=(cut.window, 1, 1), mode="nearest")


@app.function
def load_alpha(path):
    """An 8-bit mask PNG as float alpha in [0, 1]."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


@app.function
def save_alpha(alpha, path):
    """Float alpha in [0, 1] to an 8-bit mask PNG."""
    a = np.clip(np.rint(np.asarray(alpha) * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(a).save(path)


@app.class_definition
@dataclass(frozen=True)
class Page:
    """The newspaper page. All layout happens in these units, which are the
    pixels of the 1080x1920 draft; the 4K master draws the same layout at
    twice the scale, so draft and master never differ."""

    width: int = 1080
    height: int = 1920
    margin_x: float = 40.0  # left and right
    margin_top: float = 56.0
    margin_bottom: float = 56.0
    columns: int = 2
    gutter: float = 30.0  # between columns


@app.class_definition
@dataclass(frozen=True)
class Place:
    """Where the clip sits on the page. The whole clip frame is scaled and
    centred; the page shows what falls inside it."""

    scale: float = 1.25  # clip width as a share of the page width
    cx: float = 0.5  # centre of the clip, as a share of the page width
    bottom: float = 0.80  # bottom edge of the clip, as a share of the page height


@app.class_definition
@dataclass(frozen=True)
class Runaround:
    """How the text keeps away from the figure. Lengths in ems (one em = the
    body font size), so they follow the type."""

    margin: float = 0.6  # empty space kept around the figure
    release: float = 0.5  # extra distance before a freed pixel is given back to text
    min_span: float = 2.0  # narrower gaps are left empty
    snap: float = 0.25  # span edges move in steps of this size, not by single pixels


@app.class_definition
@dataclass(frozen=True)
class Type:
    """The body text: font, size, how lines break, the drop cap."""

    font: str = "InstrumentSerif-Regular.ttf"
    size: float = 30.0  # in page units
    leading: float = 1.15  # line height as a multiple of the size
    mode: str = "char"  # "char": break anywhere, like the reference; "word": whole words
    justify: bool = False  # stretch each span to fill it exactly
    max_stretch: float = 0.6  # most extra space per gap when justifying, in ems
    dropcap_lines: int = 3  # 0 = no drop cap
    paragraph_gap: int = 1  # empty lines between paragraphs


# ----------------------------------------------------------------- the page


@app.function
def line_grid(page, typ):
    """The lines of every column. Returns (bands, baselines, columns):
    bands[i] = (top, bottom) of line i, baselines[i] its baseline, and
    columns[k] = (left, right) of column k."""
    lh = typ.size * typ.leading
    n = int((page.height - page.margin_top - page.margin_bottom) // lh)
    tops = page.margin_top + lh * np.arange(n)
    ascent, descent = font_for(typ.font, typ.size).getmetrics()
    baselines = tops + (lh + ascent - descent) / 2
    bands = np.stack([tops, tops + lh], axis=1)
    col_w = (page.width - 2 * page.margin_x - page.gutter * (page.columns - 1)) / page.columns
    lefts = page.margin_x + (col_w + page.gutter) * np.arange(page.columns)
    columns = np.stack([lefts, lefts + col_w], axis=1)
    return bands, baselines, columns


@app.function
def place_image(img, page, place):
    """A clip frame (PIL image: RGBA cut-out, or 'F' alpha) moved onto the
    page as `place` says. Returns an image of the page's size."""
    s = place.scale * page.width / img.width
    w, h = max(1, round(img.width * s)), max(1, round(img.height * s))
    left = round(place.cx * page.width - w / 2)
    top = round(place.bottom * page.height - h)
    small = img.resize((w, h), Image.BILINEAR)
    canvas = Image.new(img.mode, (page.width, page.height), 0)
    canvas.paste(small, (left, top))
    return canvas


@app.function
def resize_rgba(img, size):
    """Resize an RGBA image without losing the colour under transparent
    pixels: Pillow resizes RGBA premultiplied and turns that colour black,
    which "ink" drawing then prints as a dark halo. Colour and alpha apart."""
    rgb = img.convert("RGB").resize(size, Image.LANCZOS)
    rgb.putalpha(img.getchannel("A").resize(size, Image.LANCZOS))
    return rgb


@app.function
def place_figure(path, page, place):
    """An RGBA cut-out from masks/<piece>/rgba/ moved onto the page. Colour and
    alpha are placed apart: resizing RGBA in one go premultiplies it, which
    turns the colour of every transparent pixel black (and "ink" drawing,
    which reads the clip's colours inside the solid mask, would print it)."""
    img = Image.open(path).convert("RGBA")
    rgb = place_image(img.convert("RGB"), page, place)
    a = place_image(img.getchannel("A"), page, place)
    rgb.putalpha(a)
    return rgb


@app.function
def place_alpha(alpha, page, place):
    """A clip mask (float array) on the page, as a float array."""
    img = Image.fromarray(np.asarray(alpha, dtype=np.float32))  # mode "F"
    return np.asarray(place_image(img, page, place), dtype=np.float32)


@app.function
def blob(page, t):
    """A synthetic figure for trying the layout without any clip: a body that
    drifts and a smaller head that circles it. t in [0, 1)."""
    yy, xx = np.mgrid[0 : page.height, 0 : page.width].astype(np.float32)
    a = 2 * np.pi * t
    cx = page.width * (0.45 + 0.12 * np.sin(a))
    cy = page.height * 0.55
    body = ((xx - cx) / 210) ** 2 + ((yy - cy) / 330) ** 2 <= 1
    hx = cx + 260 * np.cos(2 * a)
    hy = cy - 330 + 140 * np.sin(2 * a)
    head = (xx - hx) ** 2 + (yy - hy) ** 2 <= 110**2
    return (body | head).astype(np.float32)


# ------------------------------------------------------------ the obstacle


@app.function
def obstacle(alpha_page, typ, ra, previous=None):
    """Where text may not go: the figure grown by `ra.margin` ems.

    In time it behaves like a latch: a pixel the figure comes near is blocked
    at once, but a blocked pixel is only freed when the figure has moved
    `ra.release` ems further away. A silhouette that trembles by a pixel
    then never moves a line of text."""
    figure = np.asarray(alpha_page) >= 0.5
    if not figure.any():
        return np.zeros_like(figure)
    dist = ndimage.distance_transform_edt(~figure)
    blocked = dist <= ra.margin * typ.size
    if previous is not None:
        blocked |= previous & (dist <= (ra.margin + ra.release) * typ.size)
    return blocked


@app.function
def free_spans(blocked, page, typ, ra, extra=()):
    """The gaps each line keeps beside the obstacle, in reading order:
    column by column, line by line, left to right.

    A line is blocked across its whole height, so no glyph can touch the
    figure. Edges snap to steps of `ra.snap` ems, measured from the column's
    left side, and gaps narrower than `ra.min_span` ems are dropped.
    `extra` adds fixed rectangles (x0, y0, x1, y1), e.g. the drop cap.
    Returns a list of (column, line, x0, x1, baseline)."""
    bands, baselines, columns = line_grid(page, typ)
    snap = ra.snap * typ.size
    min_w = ra.min_span * typ.size
    spans = []
    for k, (c0, c1) in enumerate(columns):
        for i, (y0, y1) in enumerate(bands):
            row = blocked[int(y0) : int(np.ceil(y1)), int(c0) : int(np.ceil(c1))].any(axis=0)
            row = row.copy()
            for bx0, by0, bx1, by1 in extra:
                if by0 < y1 and by1 > y0:
                    row[max(0, int(bx0 - c0)) : max(0, int(np.ceil(bx1 - c0)))] = True
            # free runs start where the padded row goes blocked -> free
            d = np.diff(np.concatenate([[False], ~row, [False]]).astype(np.int8))
            starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
            for s, e in zip(starts, ends):
                x0 = np.ceil(s / snap) * snap
                x1 = np.floor(e / snap) * snap
                if x1 - x0 >= min_w:
                    spans.append((k, i, c0 + x0, c0 + x1, float(baselines[i])))
    return spans


# ------------------------------------------------------------ the text


@app.function
@functools.cache
def font_for(name, size):
    """A font from fonts/ at a size in page units (PIL)."""
    return ImageFont.truetype(io.BytesIO(asset_bytes(f"fonts/{name}")), size=size)


@app.function
@functools.cache
def advance(name, size, ch):
    """How far one character moves the pen, in page units."""
    return font_for(name, size).getlength(ch)


@app.function
def paragraphs(text):
    """The text as a list of paragraphs (blank lines separate them), each
    with its whitespace folded into single spaces."""
    return [" ".join(p.split()) for p in text.split("\n\n") if p.strip()]


@app.class_definition
@dataclass
class Layout:
    """One frame of text. Every placed character has an x, a baseline y, its
    column and line, and `pos`: its index in the fixed text stream, which
    never changes from frame to frame: the text never jumps."""

    chars: list
    x: np.ndarray
    y: np.ndarray
    col: np.ndarray
    line: np.ndarray
    pos: np.ndarray
    dropcap: tuple | None  # (char, x, baseline, size) or None
    spans: list
    exhausted: bool  # the text ran out before the page did


@app.function
def dropcap_box(page, typ, paras):
    """The drop cap: the first letter of the text, as tall as
    `typ.dropcap_lines` lines, top-left of the first column. Returns
    ((char, x, baseline, size), rectangle it keeps free) or (None, None)."""
    if not typ.dropcap_lines or not paras:
        return None, None
    bands, baselines, columns = line_grid(page, typ)
    n = typ.dropcap_lines
    ch = paras[0][0]
    ref = font_for(typ.font, 100).getbbox("H")
    cap_ratio = (ref[3] - ref[1]) / 100  # cap height per unit of size
    body = font_for(typ.font, typ.size).getbbox("H")
    cap_top = baselines[0] - (body[3] - body[1])
    size = (baselines[n - 1] - cap_top) / cap_ratio
    x = columns[0][0]
    box = (x, bands[0][0], x + advance(typ.font, size, ch) + 0.3 * typ.size, bands[n - 1][1])
    return (ch, float(x), float(baselines[n - 1]), float(size)), box


@app.function
def fill(text, spans, typ, col_width=np.inf):
    """Pour the text into the spans.

    "char" mode puts one character after another and moves to the next span
    when the next one does not fit, so words break anywhere (the reference
    reel). "word" mode moves whole words; a narrow span that cannot hold the
    next word stays empty, and only a word wider than a whole column
    (`col_width`) is broken. A paragraph ends its line and skips
    `paragraph_gap` lines.
    Spaces at the start of a span are dropped."""
    paras = paragraphs(text)
    w = lambda c: advance(typ.font, typ.size, c)
    out = dict(chars=[], x=[], y=[], col=[], line=[], pos=[])
    offsets = np.cumsum([0] + [len(p) + 1 for p in paras])  # stream index of each paragraph
    p, j = 0, (1 if typ.dropcap_lines and paras else 0)
    skip_to = None
    for col, line, x0, x1, base in spans:
        if p >= len(paras):
            break
        if skip_to is not None and (col, line) < skip_to:
            continue
        skip_to = None
        para = paras[p]
        while j < len(para) and para[j] == " ":
            j += 1
        x = x0
        first = len(out["chars"])

        def put(c, xc, jj, base=base, col=col, line=line, p=p):
            out["chars"].append(c)
            out["x"].append(xc)
            out["y"].append(base)
            out["col"].append(col)
            out["line"].append(line)
            out["pos"].append(offsets[p] + jj)

        if typ.mode == "char":
            while j < len(para) and x + w(para[j]) <= x1 + 1e-6:
                put(para[j], x, j)
                x += w(para[j])
                j += 1
        else:
            while j < len(para):
                k = para.find(" ", j)
                k = len(para) if k < 0 else k
                ww = sum(w(c) for c in para[j:k])
                if x + ww <= x1 + 1e-6:
                    for c in para[j:k]:
                        put(c, x, j)
                        x += w(c)
                        j += 1
                    if j < len(para):  # the space after the word
                        if x + w(" ") <= x1 + 1e-6:
                            put(" ", x, j)
                            x += w(" ")
                        j += 1
                elif x == x0 and ww > col_width:  # wider than a column: break it
                    while j < k and x + w(para[j]) <= x1 + 1e-6:
                        put(para[j], x, j)
                        x += w(para[j])
                        j += 1
                    break
                else:
                    break

        ended = j >= len(para)
        if typ.justify and not ended:
            justify_span(out, first, x1, typ)
        if ended:
            p, j = p + 1, 0
            skip_to = (col, line + 1 + typ.paragraph_gap)

    return Layout(
        chars=out["chars"],
        x=np.asarray(out["x"], np.float32),
        y=np.asarray(out["y"], np.float32),
        col=np.asarray(out["col"], np.int16),
        line=np.asarray(out["line"], np.int16),
        pos=np.asarray(out["pos"], np.int32),
        dropcap=None,
        spans=spans,
        exhausted=p >= len(paras),
    )


@app.function
def justify_span(out, first, x1, typ):
    """Stretch the glyphs placed since index `first` so the span ends at x1:
    the spare room goes into the spaces ("word") or between letters
    ("char"), at most `typ.max_stretch` ems per gap; otherwise left ragged."""
    xs = out["x"]
    chars = out["chars"]
    last = len(chars)
    while last > first and chars[last - 1] == " ":
        last -= 1
    if last - first < 2:
        return
    end = xs[last - 1] + advance(typ.font, typ.size, chars[last - 1])
    spare = x1 - end
    if typ.mode == "word":
        gaps = [i for i in range(first + 1, last) if chars[i - 1] == " "]
    else:
        gaps = list(range(first + 1, last))
    if not gaps or spare <= 0 or spare / len(gaps) > typ.max_stretch * typ.size:
        return
    step = spare / len(gaps)
    shift, g = 0.0, 0
    for i in range(first + 1, last):
        if g < len(gaps) and gaps[g] == i:
            shift += step
            g += 1
        xs[i] += shift


@app.function
def layout(text, alpha_page, page, typ, ra, previous=None, extra=()):
    """One frame, start to end: obstacle, drop cap, spans, text. Returns the
    Layout and the obstacle (pass it back as `previous` for the next frame).
    `extra` adds rectangles the text keeps out of (a subtitle)."""
    blocked = obstacle(alpha_page, typ, ra, previous)
    paras = paragraphs(text)
    cap, box = dropcap_box(page, typ, paras)
    spans = free_spans(blocked, page, typ, ra, extra=([box] if box else []) + list(extra))
    _, _, columns = line_grid(page, typ)
    lay = fill(text, spans, typ, col_width=float(columns[0][1] - columns[0][0]))
    lay.dropcap = cap
    return lay, blocked


@app.function
def layouts(text, alphas, page, typ, ra):
    """Many frames in order, each obstacle latched on the one before."""
    previous, out = None, []
    for a in alphas:
        lay, previous = layout(text, a, page, typ, ra, previous)
        out.append(lay)
    return out


@app.function
def moved(a, b):
    """Share of the characters placed in both layouts that sit on a different
    line or column in b than in a: how much the text reflows between two
    frames."""
    common, ia, ib = np.intersect1d(a.pos, b.pos, return_indices=True)
    if not len(common):
        return 0.0
    return float(np.mean((a.line[ia] != b.line[ib]) | (a.col[ia] != b.col[ib])))


# ------------------------------------------------------------ preview


@app.function
def preview(
    lay, page, typ, figure=None, blocked=None, palette="paper", scale=1.0, marks=(), mark="#f2d94e"
):
    """A quick PIL drawing of one layout, for contact sheets and the
    notebook: paper, text, the cut-out on top. `blocked` tints the obstacle.
    `marks` are ranges (a, b) of the text stream drawn over a highlighter
    stroke of colour `mark`."""
    pal = PALETTES[palette]
    W, H = round(page.width * scale), round(page.height * scale)
    img = Image.new("RGB", (W, H), pal["bg"])
    if blocked is not None:
        tint = Image.fromarray((blocked * 40).astype(np.uint8)).resize((W, H))
        img.paste(Image.new("RGB", (W, H), pal["ink"]), (0, 0), tint)
    d = ImageDraw.Draw(img)
    f = font_for(typ.font, typ.size * scale)
    for a, b in marks:  # the highlighter goes under the ink
        for c, x, y, pos in zip(lay.chars, lay.x, lay.y, lay.pos):
            if a <= pos < b:
                w = advance(typ.font, typ.size, c)
                top, bot = y - 0.72 * typ.size, y + 0.18 * typ.size
                d.rectangle([x * scale, top * scale, (x + w) * scale, bot * scale], fill=mark)
    for c, x, y in zip(lay.chars, lay.x, lay.y):
        if c != " ":
            d.text((x * scale, y * scale), c, font=f, fill=pal["ink"], anchor="ls")
    if lay.dropcap:
        c, x, y, s = lay.dropcap
        d.text(
            (x * scale, y * scale),
            c,
            font=font_for(typ.font, s * scale),
            fill=pal["ink"],
            anchor="ls",
        )
    if figure is not None:
        fig = resize_rgba(figure, (W, H)) if figure.size != (W, H) else figure
        img.paste(fig, (0, 0), fig)
    return img


@app.class_definition
@dataclass(frozen=True)
class Look:
    """Palette and the touches of cheap print.
    All off by default; a piece turns them on in pieces.toml. Lengths in
    page units (draft pixels)."""

    palette: str = "paper"
    grain: float = 0.0  # film grain over the whole frame (std, share of full white)
    grain_figure: float = 0.0  # extra grain on the figure only
    grain_size: float = 0.7  # grain clump size (blur of the noise)
    misregister: float = 0.0  # red and blue plates of the figure pushed apart
    soften: float = 0.0  # slight blur of the whole page, like ink on paper
    # How the figure is drawn. "colour": the cut-out as it is. "ink" (for
    # black-and-white line art): inside the solid mask, dark becomes the
    # page's ink and light becomes paper, so the figure is printed like the text.
    figure: str = "colour"
    highlight: str = "#f2d94e"  # the one accent of visual-style.md: highlighter yellow
    ink_white: float = 0.80  # this grey and lighter is paper (0..1)
    ink_black: float = 0.12  # this grey and darker is full ink


@app.class_definition
@dataclass(frozen=True)
class Reel:
    """Video settings: frame rate, and an x264 quality good enough for phones."""

    fps: int = 24  # the clip's own rate, no invented frames
    crf: int = 14
    preset: str = "slow"
    # "On twos": each picture of the clip held for `step` frames (2 = 12
    # drawings a second, like hand animation). The text moves half as often;
    # the grain still changes every frame. Never holds across a cut.
    step: int = 1


@app.function
def segments(piece):
    """The parts of the clip a piece uses, in order, as (start, end) seconds:
    `segments = [[s, e], ...]` in pieces.toml, or a single `start`/`end`.
    A segment may name its own file in clips/ as a third item,
    `[s, e, "other.mp4"]` (see segment_clips). Between two segments there is
    a cut: the text reflows at once."""
    if "segments" in piece:
        return [(float(x[0]), float(x[1])) for x in piece["segments"]]
    return [(float(piece["start"]), float(piece["end"]))]


@app.function
def segment_clips(piece):
    """The file in clips/ each segment comes from: its own third item, or the
    piece's `clip`."""
    if "segments" in piece:
        return [x[2] if len(x) > 2 else piece["clip"] for x in piece["segments"]]
    return [piece["clip"]]


@app.class_definition
@dataclass(frozen=True)
class Music:
    """The soundtrack: a file in music/ (never committed), the moment of it
    the reel starts on, fades, and loudness (-14 LUFS with one fixed gain,
    peaks limited, 48 kHz AAC). No file, no sound."""

    file: str = ""
    start: float = 0.0  # seconds into the track
    fade_in: float = 0.4
    fade_out: float = 2.0
    lufs: float = -14.0
    ceiling_db: float = -3.5  # limiter ceiling; AAC adds 1-2 dB on top
    bitrate: str = "256k"
    # The music may stop at `cut_at` seconds of the reel (0 = it plays to the
    # end). A voice from a film can be laid over or instead of it: `voice` is
    # a file in clips/, `voice_stream` its audio stream, and `voices` a list
    # of [from, to, at]: the stretch from..to of the film placed at `at`
    # seconds of the reel. The voice is levelled once, on its words only
    # (voice_level_from..voice_level_to of the film), to `lufs`.
    cut_at: float = 0.0
    cut_fade: float = 0.3
    voice: str = ""
    voice_stream: int = 0
    voices: tuple = ()
    voice_level_from: float = 0.0
    voice_level_to: float = 0.0
    # With voices, the music can stay underneath at `bed_db` (relative to its
    # level alone) and rise back over `bed_rise` seconds when the voice ends.
    bed_db: float = 0.0
    bed_rise: float = 1.0


@app.function
def held(i, spans, step):
    """The clip frame shown at reel frame i when every picture is held for
    `step` frames, counted from the start of i's segment (so a held picture
    never comes from the shot before a cut)."""
    for first, count in spans:
        if first <= i < first + count:
            return first + (i - first) // step * step
    return i


@app.function
def piece_params(piece):
    """A piece from pieces.toml as resolved dataclasses: each [piece.<name>]
    table overrides the defaults of the dataclass of that name."""
    kinds = dict(
        cutout=Cutout,
        page=Page,
        place=Place,
        runaround=Runaround,
        type=Type,
        look=Look,
        reel=Reel,
        music=Music,
        subtitles=Subtitles,
    )
    out = {k: cls(**piece.get(k, {})) for k, cls in kinds.items()}
    if "palette" in piece and "palette" not in piece.get("look", {}):
        out["look"] = replace(out["look"], palette=piece["palette"])
    return out


@app.function
def shift(a, dx):
    """Shift a 2D array sideways by dx pixels (sub-pixel, edges repeat)."""
    return ndimage.shift(a, (0, dx), order=1, mode="nearest") if dx else a


@app.class_definition
@dataclass(frozen=True)
class Subtitles:
    """Subtitles under the figure: a block of type the page text runs around,
    like the figure. Each line of a piece's `captions` is [start, end, text]
    in seconds of the reel. Inside the safe zone (above the bottom 18%)."""

    font: str = "InstrumentSerif-Italic.ttf"
    size: float = 46.0  # page units
    leading: float = 1.15
    bottom: float = 0.80  # bottom edge of the block, share of the page height
    width: float = 0.82  # widest line, share of the page width
    pad: float = 0.5  # empty space around the block, in ems of the body type
    band: bool = True  # the text keeps out of the whole width beside it: a short line stays alone
    # "scene": the band stays for the whole of every scene that has a caption
    # (the text moves only at cuts, where it moves anyway); "always": for the
    # whole reel; "": only while a caption shows. Sized for the longest caption.
    reserve: str = "scene"


@app.function
def wrap_caption(text, sub, page):
    """The caption broken into lines no wider than `sub.width` of the page."""
    font = font_for(sub.font, sub.size)
    lines, line = [], ""
    for w in text.split():
        trial = f"{line} {w}".strip()
        if line and font.getlength(trial) > sub.width * page.width:
            lines.append(line)
            line = w
        else:
            line = trial
    return lines + ([line] if line else [])


@app.function
def caption_box(text, sub, page, typ):
    """The rectangle (x0, y0, x1, y1) the caption occupies, padded: what the
    page text keeps out of."""
    lines = wrap_caption(text, sub, page)
    font = font_for(sub.font, sub.size)
    w = max(font.getlength(x) for x in lines)
    h = sub.size * sub.leading * len(lines)
    pad = sub.pad * typ.size
    y1 = sub.bottom * page.height
    x0 = (page.width - w) / 2
    if sub.band:
        return (0.0, y1 - h - pad, float(page.width), y1 + pad)
    return (x0 - pad, y1 - h - pad, x0 + w + pad, y1 + pad)


@app.function
def caption_band(captions, sub, page, typ):
    """The fixed band reserved for all the captions of a piece: full width,
    as tall as the caption with the most lines, its bottom at `sub.bottom`."""
    if not captions:
        return None
    lines = max(len(wrap_caption(c[2], sub, page)) for c in captions)
    pad = sub.pad * typ.size
    y1 = sub.bottom * page.height
    h = sub.size * sub.leading * lines
    return (0.0, y1 - h - pad, float(page.width), y1 + pad)


@app.function
def scene_bands(captions, sub, page, typ, spans, fps):
    """The reserved band of every frame: one per scene (segment) that shows a
    caption at any moment, or for every frame with `reserve = "always"`.
    Returns a list, one entry per frame: the band or None."""
    band = caption_band(captions, sub, page, typ)
    total = sum(n for _, n in spans)
    if band is None or not sub.reserve:
        return [None] * total
    if sub.reserve == "always":
        return [band] * total
    out = []
    for first, n in spans:
        t0, t1 = first / fps, (first + n) / fps
        has = any(start < t1 and end > t0 for start, end, _ in captions)
        out += [band if has else None] * n
    return out


@app.function
def highlight_ranges(text, highlights):
    """Where each phrase of `highlights` ([start, end, phrase], in order) sits
    in the text stream: (start, end, a, b), the phrase found case-blind after
    the previous one. The stream is the paragraphs joined by one character,
    as `fill` counts it."""
    stream = " ".join(paragraphs(text)).lower()  # one character between paragraphs
    out, at = [], 0
    for start, end, phrase in highlights:
        a = stream.find(" ".join(phrase.split()).lower(), at)
        if a < 0:
            raise SystemExit(f"highlight not found in the text after char {at}: {phrase!r}")
        b = a + len(" ".join(phrase.split()))
        out.append((float(start), float(end), a, b))
        at = b
    return out


@app.function
def marks_at(ranges, t):
    """The highlighted stream ranges at time t: the phrase being spoken fills
    in from its first letter to its last between its start and end, and stays
    marked until the next phrase starts (the last one to the end)."""
    for k, (start, end, a, b) in enumerate(ranges):
        nxt = ranges[k + 1][0] if k + 1 < len(ranges) else float("inf")
        if start <= t < nxt:
            done = min(1.0, (t - start) / max(end - start, 1e-6))
            return [(a, a + int(np.ceil(done * (b - a))))]
    return []


@app.function
def caption_at(captions, t):
    """The caption text shown at time t (seconds), or None."""
    for start, end, text in captions:
        if start <= t < end:
            return text
    return None


@app.function
def draw_caption(img, text, sub, page, look, scale=1.0, band=None):
    """Draw the caption centred, in the page's ink, onto a PIL image: in the
    middle of `band` (x0, y0, x1, y1) if given, else ending at `sub.bottom`."""
    lines = wrap_caption(text, sub, page)
    font = font_for(sub.font, sub.size * scale)
    d = ImageDraw.Draw(img)
    lh = sub.size * sub.leading
    if band:
        y = (band[1] + band[3]) / 2 - lh * len(lines) / 2
    else:
        y = sub.bottom * page.height - lh * len(lines)
    for line in lines:
        w = font.getlength(line)
        d.text(
            ((page.width * scale - w) / 2, (y + lh * 0.8) * scale),
            line,
            font=font,
            fill=PALETTES[look.palette]["ink"],
            anchor="ls",
        )
        y += lh
    return img


@app.function
def render_frame(
    lay,
    page,
    typ,
    look,
    figure=None,
    rng=None,
    scale=1.0,
    region=None,
    caption=None,
    sub=None,
    band=None,
    marks=(),
):
    """One finished frame as an (H, W, 3) uint8 array: paper and text, the
    figure on top with its colour plates slightly out of register, film
    grain, a touch of softness. `figure` is the page-sized RGBA cut-out (PIL)
    in page units; `rng` drives the grain (np.random.Generator). In "ink"
    mode `region` (the page-sized solid mask, float) bounds the figure."""
    base = np.asarray(
        preview(
            lay, page, typ, palette=look.palette, scale=scale, marks=marks, mark=look.highlight
        ),
        np.float32,
    )
    W, H = base.shape[1], base.shape[0]
    alpha = np.zeros((H, W), np.float32)
    if figure is not None:
        fig = resize_rgba(figure, (W, H)) if figure.size != (W, H) else figure
        f = np.asarray(fig, np.float32)
        alpha = f[..., 3] / 255.0
        if look.figure == "ink":
            if region is not None:
                r = Image.fromarray(np.asarray(region, np.float32))
                alpha = np.asarray(r.resize((W, H), Image.BILINEAR), np.float32)
            lum = (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]) / 255.0
            k = np.clip((look.ink_white - lum) / (look.ink_white - look.ink_black), 0, 1)
            ink = np.asarray(Image.new("RGB", (1, 1), PALETTES[look.palette]["ink"]), np.float32)[
                0, 0
            ]
            alpha = alpha * k  # how much ink each pixel takes
            f = np.empty_like(f)
            f[..., :3] = ink
        d = look.misregister * scale
        out = np.empty_like(base)
        for c, dx in ((0, d), (1, 0.0), (2, -d)):  # red one way, blue the other
            a = shift(alpha, dx)
            out[..., c] = base[..., c] * (1 - a) + shift(f[..., c], dx) * a
        base = out
    if caption or band:  # on paper, above the figure: always readable
        sub = sub or Subtitles()
        img = Image.fromarray(np.clip(np.rint(base), 0, 255).astype(np.uint8))
        x0, y0, x1, y1 = band or caption_box(caption, sub, page, typ)
        ImageDraw.Draw(img).rectangle(
            [x0 * scale, y0 * scale, x1 * scale, y1 * scale], fill=PALETTES[look.palette]["bg"]
        )
        if caption:
            draw_caption(img, caption, sub, page, look, scale, band=band)
        base = np.asarray(img, np.float32)
        alpha = alpha.copy()
        alpha[int(y0 * scale) : int(y1 * scale), int(x0 * scale) : int(x1 * scale)] = 0
    if (look.grain or look.grain_figure) and rng is not None:
        n = rng.standard_normal((H, W)).astype(np.float32)
        if look.grain_size:
            n = ndimage.gaussian_filter(n, look.grain_size * scale)
            n /= n.std() + 1e-8
        base += ((look.grain + look.grain_figure * alpha) * 255.0 * n)[..., None]
    if look.soften:
        base = ndimage.gaussian_filter(base, (look.soften * scale, look.soften * scale, 0))
    return np.clip(np.rint(base), 0, 255).astype(np.uint8)


# ------------------------------------------------- the notebook's pictures


@app.class_definition
@dataclass(frozen=True)
class Notebook:
    """What the notebook shows: how big, how long, and how many frames the
    sections study. Lengths in page units (draft pixels)."""

    scale: float = 0.5  # the loop and the large stills, as a share of the page
    small: float = 0.35  # the side-by-side pictures
    strip_scale: float = 0.2  # the filmstrip of section 4
    window: int = 24  # frames the sections study, two seconds at 12 fps
    strip: int = 4  # pictures in the filmstrip
    strip_step: int = 2  # frames between two pictures of the filmstrip
    fps: float = 12.0  # rate of a figure that brings none (an upload, the shape)
    quality: int = 70  # loops made in the browser (animated WebP)
    crf: int = 28  # loops shipped with the demos (H.264, made by make_demo.py)
    crop: int = 420  # side of the print-look close-up
    chart: tuple = (600, 320)  # size of the chart of section 3, in screen pixels
    figure_width: int = 640  # figures drawn by the notebooks (demos, cut-outs), in px
    sheet: tuple = (6, 240)  # cutout.py: frames and width of the check sheet


@app.function
def place_at(places, k):
    """The place of frame k: one Place for the whole clip, or one per frame."""
    return places[k] if isinstance(places, list) else places


@app.function
def make_loop(text, figure, places, page, typ, ra, look, frames, scale, progress=iter):
    """The playground's loop: the frames `frames` of `figure` laid out in
    order, each obstacle latched on the one before as in a reel, drawn at
    `scale`. A list of (H, W, 3) uint8 arrays. `progress` wraps the frame
    numbers (the notebook passes a progress bar)."""
    previous, out = None, []
    for k in progress(list(frames)):
        alpha, fig = figure_on_page(figure[k], page, place_at(places, k))
        lay, previous = layout(text, alpha, page, typ, ra, previous)
        out.append(
            render_frame(
                lay,
                page,
                typ,
                look,
                figure=fig,
                rng=np.random.default_rng(k),
                scale=scale,
                region=alpha,
            )
        )
    return out


@app.function
def loop_webp(frames, fps, quality):
    """Frames as an animated WebP that plays in a loop: the browser plays it
    on its own, with no Python running between two frames."""
    imgs = [Image.fromarray(f) for f in frames]
    buf = io.BytesIO()
    imgs[0].save(
        buf,
        format="WEBP",
        save_all=True,
        append_images=imgs[1:],
        duration=round(1000 / fps),
        loop=0,
        quality=quality,
    )
    return buf.getvalue()


@app.function
def breathing(text, alphas, page, typ, ra):
    """A stretch of frames laid out twice: as the reels are (edges snapped,
    obstacle latched) and bare (edges to the pixel, nothing latched).
    Returns (smooth, bare), each a list of (Layout, obstacle)."""
    bare = replace(ra, snap=1.0 / typ.size, release=0.0)
    out = []
    for r in (ra, bare):
        previous, seq = None, []
        for a in alphas:
            lay, previous = layout(text, a, page, typ, r, previous)
            seq.append((lay, previous))
        out.append(seq)
    return out[0], out[1]


@app.function
def text_edge(blocked, page, typ, col, line, snap=0.0):
    """Where the text of one line stops before the obstacle: the first
    blocked x of that line in that column, in page units (the column's right
    side if nothing blocks it). With `snap` (page units) it moves in steps of
    that size, as free_spans does."""
    bands, _, columns = line_grid(page, typ)
    c0, c1 = columns[col]
    y0, y1 = bands[line]
    row = blocked[int(y0) : int(np.ceil(y1)), int(c0) : int(np.ceil(c1))].any(axis=0)
    hit = np.flatnonzero(row)
    e = float(hit[0]) if len(hit) else float(len(row))
    if snap:
        e = np.floor(e / snap) * snap
    return float(c0 + e)


@app.function
def guarded_line(bare, smooth, page, typ, snap):
    """The (column, line) where the guards of section 3 do the most: among
    the lines the figure blocks in every frame with some text before it, the
    one whose edge moves most often bare (`bare` obstacles, to the pixel) and
    least often guarded (`smooth` obstacles, snapped by `snap`)."""
    bands, _, columns = line_grid(page, typ)
    best, work = (0, 0), -1
    for col, (c0, c1) in enumerate(columns):
        for line in range(len(bands)):
            e = np.array([text_edge(b, page, typ, col, line) for b in bare])
            if not np.all((e > c0 + typ.size) & (e < c1)):
                continue
            g = np.array([text_edge(b, page, typ, col, line, snap) for b in smooth])
            w = int(np.count_nonzero(np.diff(e)) - np.count_nonzero(np.diff(g)))
            if w > work:
                best, work = (col, line), w
    return best


@app.function
def lines_changed(a, b):
    """How many lines have different free spans in Layout b than in a: the
    lines whose text has to be poured again."""

    def by_line(spans):
        out = {}
        for col, line, x0, x1, _ in spans:
            out.setdefault((col, line), []).append((x0, x1))
        return out

    sa, sb = by_line(a.spans), by_line(b.spans)
    return sum(sa.get(k) != sb.get(k) for k in set(sa) | set(sb))


@app.function
def moved_ranges(a, b):
    """The characters on a different line or column in b than in a, as
    ranges (start, end) of the text stream, for preview's `marks`."""
    common, ia, ib = np.intersect1d(a.pos, b.pos, return_indices=True)
    pos = common[(a.line[ia] != b.line[ib]) | (a.col[ia] != b.col[ib])]
    if not len(pos):
        return []
    cut = np.flatnonzero(np.diff(pos) > 1)
    starts = np.concatenate([[pos[0]], pos[cut + 1]])
    ends = np.concatenate([pos[cut], [pos[-1]]]) + 1
    return list(zip(starts.tolist(), ends.tolist()))


@app.function
def breath_chart(tracks, counts, size, palette="paper"):
    """Section 3's chart, in ink on paper, one row per entry, frames left to
    right. `tracks` ({label: xs}): where one line's text stops, drawn as
    steps on a scale shared by all of them, so a flat stretch is a line
    that holds still. `counts` ({label: ns}): a bar per frame, as tall as
    the number of lines poured again since the frame before."""
    pal = PALETTES[palette]
    W, H = size
    img = Image.new("RGB", (W, H), pal["bg"])
    d = ImageDraw.Draw(img)
    mono = font_for("IBMPlexMono-Regular.ttf", 13)
    ink = np.array(Image.new("RGB", (1, 1), pal["ink"]).getpixel((0, 0)), float)
    bg = np.array(Image.new("RGB", (1, 1), pal["bg"]).getpixel((0, 0)), float)

    def mix(f):  # the ink at coverage f of the paper
        return tuple(np.rint(bg + f * (ink - bg)).astype(int).tolist())

    label_w, pad = 150, 8
    rows = len(tracks) + len(counts)
    row_h = (H - pad) / rows
    n = len(next(iter(tracks.values())))
    xs = np.linspace(pad, W - label_w, n + 1)  # frame k spans xs[k]..xs[k+1]
    allv = np.concatenate([np.asarray(v, float) for v in tracks.values()])
    lo, hi = allv.min(), allv.max()
    hi = hi if hi > lo else lo + 1
    for r, (name, v) in enumerate(tracks.items()):
        top = pad + r * row_h
        ys = top + pad + (row_h - 3 * pad) * (1 - (np.asarray(v, float) - lo) / (hi - lo))
        pts = []
        for k, y in enumerate(ys.tolist()):
            pts += [(xs[k], y), (xs[k + 1], y)]
        d.line(pts, fill=mix(1.0), width=2)
        d.text((W - label_w + 12, top + row_h / 2), name, font=mono, fill=mix(0.8), anchor="lm")
    most = max(1, max(max(c, default=0) for c in counts.values()))
    bar_w = max(2.0, (xs[1] - xs[0]) * 0.6)
    for r, (name, ns) in enumerate(counts.items(), start=len(tracks)):
        base = pad + (r + 1) * row_h - pad
        for k, c in enumerate(ns, start=1):
            x = (xs[k] + xs[k + 1]) / 2
            h = max(1.0, c / most * (row_h - 2 * pad))
            d.rectangle([x - bar_w / 2, base - h, x + bar_w / 2, base], fill=mix(0.8))
        d.text((W - label_w + 12, base - row_h / 3), name, font=mono, fill=mix(0.8), anchor="lm")
    return img


@app.function
def filmstrip(lays, figures, page, typ, picks, step, scale, mark="#f2d94e"):
    """A row of small pages, frames `picks` of `lays`; in each, the letters
    that changed line since `step` frames before are marked."""
    tiles = []
    for k in picks:
        marks = moved_ranges(lays[k - step], lays[k]) if k >= step else []
        tiles.append(
            preview(lays[k], page, typ, figure=figures[k], scale=scale, marks=marks, mark=mark)
        )
    gap = 8
    out = Image.new(
        "RGB", (sum(t.width for t in tiles) + gap * (len(tiles) - 1), tiles[0].height), "white"
    )
    for i, t in enumerate(tiles):
        out.paste(t, (i * (t.width + gap), 0))
    return out


# ------------------------------------------------------------- figures


@app.function
def demo_names():
    """The demo figures shipped with the notebook, by folder name."""
    return json.loads(asset_text("demo/index.json"))  # written by tools/make_demo.py


@app.function
@functools.cache
def demo_frame(name, k):
    """Frame k of a demo figure: (cut-out RGBA, mask the text avoids as float
    array). Fetched only when it is shown: in the browser every file is a
    request, and the demos have hundreds."""
    rgba = Image.open(io.BytesIO(asset_bytes(f"demo/{name}/{k:04d}.webp"))).convert("RGBA")
    mask = Image.open(io.BytesIO(asset_bytes(f"demo/{name}/{k:04d}.png"))).convert("L")
    return rgba, np.asarray(mask, np.float32) / 255.0


@app.class_definition
class DemoFrames:
    """The frames of a demo as a sequence (len, [k]) that loads each frame on
    first use."""

    def __init__(self, name, count):
        self.name, self.count = name, count

    def __len__(self):
        return self.count

    def __getitem__(self, k):
        if not 0 <= k < self.count:
            raise IndexError(k)
        return demo_frame(self.name, k)


@app.function
@functools.cache
def load_demo(name):
    """A demo figure: (info, frames). Only demo.json is read here; the frames
    are fetched as they are shown (DemoFrames)."""
    info = json.loads(asset_text(f"demo/{name}/demo.json"))
    return info, DemoFrames(name, info["frames"])


@app.function
def local_names():
    """Pieces whose masks exist on this computer (masks/<slug>/, made with
    tools/masks.py) and are not shipped as demos: your own clips. Nothing of
    them is in the repository."""
    if not PIECES.exists() or not MASKS.exists():
        return []
    shipped = set(demo_names())
    return [
        slug
        for slug in load_pieces()
        if slug not in shipped
        and (MASKS / slug / "masks.json").exists()
        and (MASKS / slug / "rgba").exists()
    ]


@app.function
@functools.cache
def load_local(slug, step=2, width=640):
    """A local piece as a figure: (info, frames, places), every `step`-th
    frame at `width` px, with the piece's place for each frame (a scene may
    have its own)."""
    piece = load_pieces()[slug]
    p = piece_params(piece)
    spans = json.loads((MASKS / slug / "masks.json").read_text()).get("spans") or []
    masks = sorted((MASKS / slug).glob("[0-9]*.png"))
    frames, places = [], []
    for i in range(0, len(masks), step):
        rgba = Image.open(MASKS / slug / "rgba" / masks[i].name).convert("RGBA")
        h = round(rgba.height * width / rgba.width)
        mask = Image.open(masks[i]).convert("L").resize((width, h), Image.BILINEAR)
        frames.append((resize_rgba(rgba, (width, h)), np.asarray(mask, np.float32) / 255.0))
        places.append(segment_place(piece, segment_of(i, spans)) if spans else p["place"])
    from dataclasses import asdict

    info = dict(
        title=piece["title"], credit=f"{piece['title']} (your local masks)", look=asdict(p["look"])
    )
    return info, frames, places


@app.function
def frames_from_upload(data):
    """A figure from an uploaded image: a PNG or WebP with transparency (one
    frame) or an animated GIF/WebP (every frame). Its alpha is the mask."""
    import io

    img = Image.open(io.BytesIO(data))
    frames = []
    for k in range(getattr(img, "n_frames", 1)):
        img.seek(k)
        rgba = img.convert("RGBA")
        frames.append((rgba, np.asarray(rgba.getchannel("A"), np.float32) / 255.0))
    return frames


@app.function
def shape_frames(page, n=48):
    """A figure with no clip at all: the synthetic blob, `n` frames of one
    loop, drawn in the page's ink."""
    out = []
    for k in range(n):
        a = blob(page, k / n)
        rgba = Image.new("RGBA", (page.width, page.height), PAPER["ink"])
        rgba.putalpha(Image.fromarray((a * 255).astype(np.uint8)))
        out.append((rgba, a))
    return out


@app.function
def figure_on_page(frame, page, place):
    """One figure frame (cut-out, mask) moved onto the page as `place` says:
    (mask on the page as float array, cut-out on the page as RGBA image)."""
    rgba, mask = frame
    alpha = place_alpha(mask, page, place)
    rgb = place_image(rgba.convert("RGB"), page, place)
    rgb.putalpha(place_image(rgba.getchannel("A"), page, place))
    return alpha, rgb


@app.function
def grid_preview(page, typ, scale=0.5):
    """The empty page: the columns and the baselines the text sits on."""
    W, H = round(page.width * scale), round(page.height * scale)
    img = Image.new("RGB", (W, H), PAPER["bg"])
    d = ImageDraw.Draw(img)
    bands, baselines, columns = line_grid(page, typ)
    for c0, c1 in columns:
        box = [c0 * scale, bands[0][0] * scale, c1 * scale, bands[-1][1] * scale]
        d.rectangle(box, outline=PAPER["ink"])
        for y in baselines:
            d.line([c0 * scale, y * scale, c1 * scale, y * scale], fill="#c9c3b8")
    return img


if __name__ == "__main__":
    app.run()
