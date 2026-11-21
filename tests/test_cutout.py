"""Phase 1: cleaning raw masks from a segmentation model."""

import numpy as np
import pytest

from runaround import Cutout, clean_mask, load_alpha, save_alpha, smooth_masks


def two_figures():
    """A girl (big rectangle) with a pinhole and a big gap, a dog (small
    rectangle), and a speck of background noise."""
    a = np.zeros((200, 300), np.float32)
    a[20:180, 20:120] = 1.0  # girl
    a[60, 60] = 0.0  # pinhole: model noise
    a[100:140, 40:80] = 0.0  # big gap between arm and staff: real
    a[120:160, 200:250] = 0.9  # dog, a separate part
    a[10, 280] = 1.0  # speck
    return a


def test_keeps_every_part_drops_specks():
    out = clean_mask(two_figures(), Cutout(edge_px=0))
    assert out[100, 30] == 1.0  # girl
    assert out[140, 225] == 1.0  # dog survives: not only the largest part
    assert out[10, 280] == 0.0  # speck gone


def test_fills_small_holes_only():
    out = clean_mask(two_figures(), Cutout(edge_px=0))
    assert out[60, 60] == 1.0
    assert out[120, 60] == 0.0


def test_soft_rim_is_kept():
    a = np.zeros((50, 50), np.float32)
    a[10:40, 10:40] = 1.0
    a[10:40, 40] = 0.3  # anti-aliased edge
    out = clean_mask(a, Cutout(edge_px=2))
    assert np.isclose(out[20, 40], 0.3)
    out0 = clean_mask(a, Cutout(edge_px=0))
    assert out0[20, 40] == 0.0


def test_empty_mask():
    assert not clean_mask(np.zeros((8, 8))).any()


def test_same_mask_same_result():
    a = two_figures()
    assert np.array_equal(clean_mask(a), clean_mask(a.copy()))


def test_median_removes_one_frame_flicker():
    stack = np.zeros((5, 4, 4), np.float32)
    stack[:, 1:3, 1:3] = 1.0
    stack[2, 0, 0] = 1.0  # flicker in one frame
    out = smooth_masks(stack, Cutout(window=3))
    assert out[2, 0, 0] == 0.0
    assert np.array_equal(out[:, 1:3, 1:3], stack[:, 1:3, 1:3])


def test_alpha_roundtrip(tmp_path):
    a = np.linspace(0, 1, 256, dtype=np.float32).reshape(16, 16)
    save_alpha(a, tmp_path / "m.png")
    assert np.abs(load_alpha(tmp_path / "m.png") - a).max() <= 0.5 / 255 + 1e-6


def test_ink_fill_recovers_what_the_outline_encloses():
    """A white glove drawn as a black ring: the model saw only half of it."""
    from runaround import ink_fill

    gray = np.full((100, 100), 0.8, np.float32)  # light wall
    yy, xx = np.mgrid[0:100, 0:100]
    r = np.hypot(yy - 50, xx - 50)
    gray[(r > 18) & (r < 22)] = 0.1  # the ink outline
    gray[10, 10] = 0.1  # a crack far away
    model = np.zeros_like(gray)
    model[30:72, 30:50] = 1.0  # left half only
    out = ink_fill(model, gray, Cutout(ink=0.35, ink_reach=24, ink_close=2))
    assert out[50, 65] == 1.0  # the white inside, right half, is filled
    assert out[50, 71] == 1.0  # the outline on the far side joins
    assert out[10, 10] == 0.0  # ink far from the figure stays out
    assert np.array_equal(ink_fill(model, gray, Cutout()), model)  # off by default


def test_solidify_closes_a_broken_figure():
    from runaround import solidify

    a = np.zeros((60, 60), np.float32)
    a[10:50, 10:25] = 1.0
    a[10:50, 33:50] = 1.0  # same figure, split by an 8 px gap
    a[25:35, 40:45] = 0.0  # a hole
    out = solidify(a, Cutout(solidify=5))
    assert out[30, 29] == 1.0 and out[30, 42] == 1.0
    assert out[5, 5] == 0.0  # outside stays outside
    assert np.array_equal(solidify(a, Cutout()), a)  # off by default


def test_hold_bridges_one_bad_frame():
    from runaround import hold_masks

    stack = np.zeros((5, 4, 4), np.float32)
    stack[:, 1:3, 1:3] = 1.0
    stack[2, 1:3, 1:3] = 0.0  # the model lost the figure in one frame
    out = hold_masks(stack, Cutout(hold=1))
    assert out[2, 1:3, 1:3].all()
    assert np.array_equal(hold_masks(stack, Cutout()), stack)  # off by default


def test_glow_fill_takes_light_near_the_figure_only():
    from runaround import glow_fill

    rgb = np.zeros((40, 140, 3), np.uint8)
    rgb[18:22, 30:50] = (180, 200, 255)  # a spark between two heads
    rgb[2, 138] = 255  # a star far away
    a = np.zeros((40, 140), np.float32)
    a[10:30, 10:30] = 1.0
    a[10:30, 50:70] = 1.0
    out = glow_fill(a, rgb, Cutout(glow=0.7, ink_reach=24))
    assert out[20, 40] == 1.0
    assert out[2, 138] == 0.0
    assert np.array_equal(glow_fill(a, rgb, Cutout()), a)  # off by default


def test_ink_open_drops_a_thin_line_but_keeps_a_closed_shape():
    from runaround import ink_fill

    gray = np.full((80, 120), 0.8, np.float32)
    gray[60:63, :] = 0.1  # a floor line, 3 px, touching the figure
    gray[40:70, 50:70] = 0.1  # a dark leg the model missed
    model = np.zeros_like(gray)
    model[10:45, 45:75] = 1.0
    out = ink_fill(model, gray, Cutout(ink=0.35, ink_reach=30, ink_close=0, ink_open=3))
    assert out[55, 60] == 1.0  # the leg stays
    assert out[61, 5] == 0.0 and out[61, 115] == 0.0  # the floor line goes


def test_cutout_notebook_export_is_a_playground_upload():
    """cutout.py hands its cut-out to runaround.py as an animated WebP: the
    playground must read back every frame with its transparency as the mask."""
    pytest.importorskip("anywidget")  # cutout.py's group: uv sync --group masks
    from PIL import Image

    from cutout import cutout_webp, slug
    from runaround import frames_from_upload

    frames = []
    for k in range(3):
        a = np.zeros((40, 60, 4), np.uint8)
        a[..., :3] = 200
        a[10:30, 5 + 10 * k : 25 + 10 * k, 3] = 255
        frames.append(Image.fromarray(a))
    back = frames_from_upload(cutout_webp(frames, 12))
    assert len(back) == 3
    for (_, mask), f in zip(back, frames):
        assert np.array_equal(mask >= 0.5, np.asarray(f.getchannel("A")) >= 128)
    assert slug("Spring (2019) final.MKV") == "spring_2019_final_mkv"


def test_shot_picker_cuts_and_first_selection():
    """Cuts at the ends or a flash apart count once, and the first selection
    sits inside the longest shot, on tenths of a second."""
    pytest.importorskip("anywidget")
    from cutout import Scan, clean_cuts, default_shot

    assert clean_cuts([0.1, 5.0, 5.1, 9.0, 9.95], 10.0, 0.25) == [5.0, 9.0]
    scan = Scan(default=3.0)
    assert default_shot([2.0, 7.25], 10.0, scan) == (2.0, 5.0)
    assert default_shot([4.03], 5.0, scan) == (0.0, 3.0)
    assert default_shot([1.0, 2.84], 3.0, scan) == (1.0, 2.8)


def test_video_arrives_in_pieces(tmp_path):
    """The upload writes each piece to disk, refuses one out of order, and a
    different video with the same name does not take the first one's place."""
    pytest.importorskip("anywidget")
    from cutout import Scan, VideoDrop

    drop = VideoDrop(tmp_path, Scan())
    sent = []
    drop.send = lambda msg, buffers=None: sent.append(msg)
    drop._on_piece(drop, {"kind": "start", "name": "My Clip.MP4", "size": 6}, [])
    assert sent[-1] == {"kind": "next", "at": 0}
    drop._on_piece(drop, {"kind": "piece", "at": 3}, [b"xxx"])
    assert sent[-1]["kind"] == "error"
    drop._on_piece(drop, {"kind": "piece", "at": 0}, [b"abc"])
    drop._on_piece(drop, {"kind": "end"}, [])
    assert sent[-1]["kind"] == "error"  # only half of it has arrived
    drop._on_piece(drop, {"kind": "piece", "at": 3}, [b"def"])
    drop._on_piece(drop, {"kind": "end"}, [])
    first = tmp_path / "my_clip" / "source.mp4"
    assert sent[-1] == {"kind": "saved", "path": str(first)}
    assert first.read_bytes() == b"abcdef"
    drop._on_piece(drop, {"kind": "start", "name": "my clip.mp4", "size": 2}, [])
    drop._on_piece(drop, {"kind": "piece", "at": 0}, [b"zz"])
    drop._on_piece(drop, {"kind": "end"}, [])
    assert sent[-1]["path"] == str(tmp_path / "my_clip_2" / "source.mp4")
    assert first.read_bytes() == b"abcdef"
    drop._on_piece(drop, {"kind": "start", "name": "notes.txt", "size": 2}, [])
    assert sent[-1]["kind"] == "error"


def test_boxes_on_any_frame_are_followed_both_ways(tmp_path, monkeypatch):
    """A box on frame 2 of 5 is followed to the end and back to the start, and
    its masks land on the right frames, joined with those of a first-frame box."""
    pytest.importorskip("anywidget")
    import cutout as C

    prompts = C.as_prompts([[1, 2, 3, 4], {"frame": 2, "box": [5, 6, 7, 8]}])
    assert prompts[0] == {"frame": 0, "box": [1, 2, 3, 4]}
    assert C.prompt_groups(prompts) == {0: [[1, 2, 3, 4]], 2: [[5, 6, 7, 8]]}
    assert C.tracking_steps(5, prompts) == 5 + 6

    runs = {"on0": 5, "on2": 3, "back2": 3}
    monkeypatch.setattr(
        C,
        "clip_from",
        lambda frames, k, fps, work, backward=False: f"{'back' if backward else 'on'}{k}",
    )

    def fake(video, boxes):
        for i in range(runs[video]):
            m = np.zeros((1, 8), np.float32)
            # first-frame box lights column 0; the other marks its distance from frame 2
            m[0, 0 if video == "on0" else 1 + i + (3 if video == "back2" else 0)] = 1
            yield m

    monkeypatch.setattr(C, "sam2_masks", fake)
    ticks = []
    out = C.follow_boxes(list(range(5)), prompts, 12, tmp_path, lambda: ticks.append(1))
    assert len(out) == 5 and len(ticks) == 11
    assert all(m[0, 0] == 1 for m in out)
    assert out[2][0, 1] == 1 and out[3][0, 2] == 1 and out[4][0, 3] == 1  # forward
    assert out[2][0, 4] == 1 and out[1][0, 5] == 1 and out[0][0, 6] == 1  # backward


def test_clip_backward_begins_on_the_frame_of_the_box(tmp_path):
    """SAM 2 takes the box on the first frame of its video, so the video going
    backward must begin on that very frame and end on the first of the shot."""
    pytest.importorskip("anywidget")
    pytest.importorskip("imageio_ffmpeg")
    import subprocess

    from PIL import Image

    import cutout as C

    folder = tmp_path / "frames"
    folder.mkdir()
    levels = [20, 60, 100, 140, 180, 220]  # each frame has its own grey
    frames = []
    for i, v in enumerate(levels):
        frames.append(folder / f"{i:05d}.png")
        Image.new("RGB", (64, 64), (v, v, v)).save(frames[-1])
    video = C.clip_from(frames, 3, 12, tmp_path, backward=True)
    cmd = [C.ffmpeg(), "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    raw = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.uint8)
    seen = raw.reshape(-1, 64, 64).mean(axis=(1, 2))
    assert len(seen) == 4
    assert np.abs(seen - np.array([140, 100, 60, 20])).max() < 8
