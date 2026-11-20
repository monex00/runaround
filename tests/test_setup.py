"""Phase 0: the notebook imports and the catalogue of pieces is well formed."""

from runaround import FONTS, PALETTES, load_pieces, segments, use_fonts


def test_pieces_are_well_formed():
    pieces = load_pieces()
    assert pieces, "pieces.toml has no [[piece]]"
    seeds = set()
    for slug, p in pieces.items():
        assert p["palette"] in PALETTES, slug
        assert isinstance(p["seed"], int), slug
        segs = segments(p)
        assert segs, slug
        assert all(0 <= s < e for s, e in segs), slug
        seeds.add(p["seed"])
    assert len(seeds) == len(pieces), "two pieces share a seed"


def test_fonts_are_there():
    names = {f.stem for f in FONTS.glob("*.ttf")}
    assert {"InstrumentSerif-Regular", "IBMPlexMono-Regular"} <= names
    assert use_fonts() == dict(serif="Instrument Serif", mono="IBM Plex Mono")
