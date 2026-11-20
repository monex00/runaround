"""Phase 2: the runaround. Obstacle, spans and the text poured into them."""

from dataclasses import replace

import numpy as np
import pytest

from runaround import (
    TEXTS,
    Page,
    Runaround,
    Type,
    advance,
    blob,
    layout,
    layouts,
    line_grid,
    load_pieces,
    moved,
    obstacle,
    paragraphs,
)

PAGE, TYPE, RA = Page(), Type(), Runaround()
TEXT = (TEXTS / "spring.txt").read_text(encoding="utf-8")


def empty():
    return np.zeros((PAGE.height, PAGE.width), np.float32)


def test_identical_masks_identical_layout():
    """The text never jumps: the same mask always gives the same layout."""
    a, _ = layout(TEXT, blob(PAGE, 0.3), PAGE, TYPE, RA)
    b, _ = layout(TEXT, blob(PAGE, 0.3).copy(), PAGE, TYPE, RA)
    assert a.chars == b.chars
    for f in ("x", "y", "col", "line", "pos"):
        assert np.array_equal(getattr(a, f), getattr(b, f))


@pytest.mark.parametrize("mode", ["char", "word"])
@pytest.mark.parametrize("justify", [False, True])
def test_no_glyph_touches_the_figure(mode, justify):
    typ = replace(TYPE, mode=mode, justify=justify)
    for t in (0.0, 0.4, 0.8):
        lay, blocked = layout(TEXT, blob(PAGE, t), PAGE, typ, RA)
        bands, _, _ = line_grid(PAGE, typ)
        for c, x, line in zip(lay.chars, lay.x, lay.line):
            y0, y1 = bands[line]
            x1 = x + advance(typ.font, typ.size, c)
            box = blocked[int(y0) : int(np.ceil(y1)), int(x) : int(np.ceil(x1))]
            assert not box.any(), (c, x, line)


def test_glyphs_stay_inside_their_column():
    lay, _ = layout(TEXT, blob(PAGE, 0.2), PAGE, TYPE, RA)
    _, _, columns = line_grid(PAGE, TYPE)
    for c, x, col in zip(lay.chars, lay.x, lay.col):
        c0, c1 = columns[col]
        assert c0 - 1e-3 <= x and x + advance(TYPE.font, TYPE.size, c) <= c1 + 1e-3


def test_reading_order_follows_the_stream():
    lay, _ = layout(TEXT, blob(PAGE, 0.6), PAGE, TYPE, RA)
    assert np.all(np.diff(lay.pos) > 0)
    key = lay.col.astype(np.int64) * 10_000 + lay.line
    assert np.all(np.diff(key) >= 0)


def test_empty_page_uses_every_line():
    lay, _ = layout(TEXT, empty(), PAGE, TYPE, RA)
    bands, _, _ = line_grid(PAGE, TYPE)
    assert not lay.exhausted, "spring.txt is too short to fill an empty page"
    used = {(int(c), int(i)) for c, i in zip(lay.col, lay.line)}
    gaps = len(paragraphs(TEXT))  # at most one empty line per paragraph break
    assert len(used) >= PAGE.columns * len(bands) - gaps


def test_text_fills_page_around_figure_without_running_out():
    lay, _ = layout(TEXT, blob(PAGE, 0.5), PAGE, TYPE, RA)
    assert not lay.exhausted


def test_drop_cap_is_the_first_letter_and_text_starts_after_it():
    lay, _ = layout(TEXT, empty(), PAGE, TYPE, RA)
    assert lay.dropcap[0] == TEXT.strip()[0]
    assert lay.pos[0] == 1
    assert lay.x[0] > lay.dropcap[1] + advance(TYPE.font, lay.dropcap[3], lay.dropcap[0])


@pytest.mark.parametrize("justify", [False, True])
def test_word_mode_never_splits_a_word(justify):
    """No word in spring.txt is wider than a column, so two neighbouring
    letters of a word always sit side by side in the same span."""
    typ = replace(TYPE, mode="word", justify=justify)
    for t in (0.1, 0.5, 0.9):
        lay, _ = layout(TEXT, blob(PAGE, t), PAGE, typ, RA)
        for k in range(len(lay.chars) - 1):
            if lay.chars[k] == " " or lay.chars[k + 1] == " ":
                continue
            if lay.pos[k + 1] != lay.pos[k] + 1:
                continue
            same = lay.line[k] == lay.line[k + 1] and lay.col[k] == lay.col[k + 1]
            step = lay.x[k + 1] - lay.x[k]
            w = advance(typ.font, typ.size, lay.chars[k])
            assert same and abs(step - w) < 1e-3, (lay.chars[k], lay.chars[k + 1])


def test_obstacle_latches_in_time():
    """A figure that shrinks by a hair keeps its obstacle; one that grows
    takes the new pixels at once."""
    a = empty()
    a[800:1000, 400:600] = 1
    b0 = obstacle(a, TYPE, RA)
    smaller = empty()
    smaller[802:998, 402:598] = 1
    assert np.array_equal(obstacle(smaller, TYPE, RA, previous=b0), b0)
    bigger = empty()
    bigger[700:1000, 400:600] = 1
    b2 = obstacle(bigger, TYPE, RA, previous=b0)
    assert b2[700:800, 400:600].all()


def test_a_still_figure_does_not_move_the_text():
    frames = [blob(PAGE, 0.25)] * 4
    lays = layouts(TEXT, frames, PAGE, TYPE, RA)
    assert all(moved(lays[0], b) == 0.0 for b in lays[1:])


def test_spring_piece_has_its_text():
    piece = load_pieces()["spring"]
    assert (TEXTS / piece["text"]).exists()


def test_place_figure_keeps_colour_under_transparency(tmp_path):
    """Resizing RGBA premultiplies it; the wall behind a transparent pixel
    must keep its grey, or "ink" drawing prints it black."""
    from PIL import Image

    from runaround import Place, place_figure

    rgba = np.zeros((40, 60, 4), np.uint8)
    rgba[..., :3] = 200  # light wall everywhere
    rgba[10:30, 20:40, 3] = 255  # only the figure is opaque
    Image.fromarray(rgba).save(tmp_path / "f.png")
    fig = np.asarray(place_figure(tmp_path / "f.png", PAGE, Place(scale=1.0, bottom=0.5)))
    inside = fig[..., 3] == 0
    covered = fig[..., :3].sum(axis=2) > 0
    assert (inside & covered).any(), "transparent wall pixels lost their colour"
    assert fig[..., :3][inside & covered].min() >= 190


def test_held_frames_never_cross_a_cut():
    from runaround import held

    spans = [[0, 5], [5, 4]]
    assert [held(i, spans, 2) for i in range(9)] == [0, 0, 2, 2, 4, 5, 5, 7, 7]
    assert [held(i, spans, 1) for i in range(9)] == list(range(9))


def test_segment_place_overrides_one_segment():
    from runaround import segment_of, segment_place

    piece = dict(palette="paper", place=dict(scale=1.4), places={"1": dict(scale=0.9)})
    assert segment_place(piece, 0).scale == 1.4
    assert segment_place(piece, 1).scale == 0.9
    assert [segment_of(i, [[0, 3], [3, 2]]) for i in range(5)] == [0, 0, 0, 1, 1]


def test_scene_bands_reserve_whole_scenes_with_a_caption():
    from runaround import Subtitles, scene_bands

    spans = [[0, 24], [24, 24], [48, 24]]  # three 1 s scenes at 24 fps
    caps = [[0.5, 0.8, "So far, so good."]]  # only in the first scene
    bands = scene_bands(caps, Subtitles(reserve="scene"), PAGE, TYPE, spans, 24)
    assert all(b is not None for b in bands[:24]) and all(b is None for b in bands[24:])
    assert all(
        b is not None for b in scene_bands(caps, Subtitles(reserve="always"), PAGE, TYPE, spans, 24)
    )


def test_render_at_small_scale_keeps_colour_under_transparency():
    """Drawing the page at scale < 1 resizes the figure; it must not turn the
    wall under transparent pixels black, or "ink" figures get dark halos."""
    from PIL import Image

    from runaround import Look, render_frame

    rgba = np.zeros((PAGE.height, PAGE.width, 4), np.uint8)
    rgba[..., :3] = 230  # light wall everywhere
    rgba[900:1100, 400:700, 3] = 255  # the figure
    region = np.zeros((PAGE.height, PAGE.width), np.float32)
    region[860:1140, 360:740] = 1.0  # the solid mask reaches past the cut-out
    lay, _ = layout(TEXT, region, PAGE, TYPE, RA)
    look = Look(figure="ink", grain=0, grain_figure=0, misregister=0, soften=0)
    img = render_frame(
        lay, PAGE, TYPE, look, figure=Image.fromarray(rgba), scale=0.5, region=region
    )
    ring = img[870 // 2 : 890 // 2, 380 // 2 : 720 // 2]  # inside the region, outside the cut-out
    assert ring.mean() > 200


def test_the_loop_is_latched_and_reproducible():
    """make_loop draws the same pictures twice, and plays as an animated WebP
    with one picture per frame."""
    import io

    from PIL import Image

    from runaround import Look, Place, loop_webp, make_loop, shape_frames

    figure = shape_frames(PAGE, n=6)
    args = (TEXT, figure, Place(scale=1.0, bottom=1.0), PAGE, TYPE, RA, Look(), range(3), 0.2)
    a, b = make_loop(*args), make_loop(*args)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    webp = Image.open(io.BytesIO(loop_webp(a, 12, 70)))
    assert webp.n_frames == 3


def test_the_guards_pour_fewer_lines_again():
    """Section 3's claim: a figure that trembles by a pixel or two moves no
    line with both guards, while bare edges follow every pixel."""
    from runaround import breathing, lines_changed

    frames = []
    for k in range(6):
        a = empty()
        a[700 : 1100 + (k % 2) * 2, 380 : 700 + (k % 3)] = 1
        frames.append(a)
    smooth, bare = breathing(TEXT, frames, PAGE, TYPE, RA)
    pairs = list(zip(range(5), range(1, 6)))
    assert sum(lines_changed(smooth[i][0], smooth[j][0]) for i, j in pairs) == 0
    assert sum(lines_changed(bare[i][0], bare[j][0]) for i, j in pairs) > 0


def test_moved_ranges_cover_exactly_the_moved_letters():
    from runaround import moved_ranges

    a, _ = layout(TEXT, blob(PAGE, 0.1), PAGE, TYPE, RA)
    b, _ = layout(TEXT, blob(PAGE, 0.4), PAGE, TYPE, RA)
    marked = {p for s, e in moved_ranges(a, b) for p in range(s, e)}
    common, ia, ib = np.intersect1d(a.pos, b.pos, return_indices=True)
    changed = (a.line[ia] != b.line[ib]) | (a.col[ia] != b.col[ib])
    assert marked == set(common[changed].tolist())
