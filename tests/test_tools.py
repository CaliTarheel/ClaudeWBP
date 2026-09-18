"""The CAD-surgery tools: mirroring a body, and canting its panels."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from cant_panels import cant, census                                 # noqa: E402
from symmetrize import clip_half, mirror, symmetrize, symmetry_error  # noqa: E402

from echo1 import shapes                                             # noqa: E402
from echo1.geometry import Mesh                                      # noqa: E402
from echo1.solver import monostatic_rcs                              # noqa: E402


@pytest.fixture
def lopsided():
    """A symmetric body with a lump added on one side only."""
    return shapes.box(4.0, 2.0, 1.0) + shapes.box(1.4, 1.4, 1.4).translated(
        [1.0, 1.7, 0.4])


def test_clip_keeps_the_requested_side(lopsided):
    neg = clip_half(lopsided, axis=1, keep_negative=True)
    pos = clip_half(lopsided, axis=1, keep_negative=False)
    assert neg.vertices[:, 1].max() <= 1e-9
    assert pos.vertices[:, 1].min() >= -1e-9
    assert neg.total_area + pos.total_area == pytest.approx(lopsided.total_area, rel=1e-6)


def test_mirror_keeps_normals_outward():
    box = shapes.box(2.0, 1.0, 1.0)
    assert mirror(box, axis=1).volume == pytest.approx(box.volume, rel=1e-12)


def test_symmetrize_produces_a_symmetric_closed_body(lopsided):
    assert symmetry_error(lopsided, axis=1) < 0.95
    out = symmetrize(lopsided, axis=1, keep_negative=True)
    assert symmetry_error(out, axis=1) == pytest.approx(1.0)
    assert out.is_closed
    assert out.volume > 0


def test_symmetrize_makes_the_rcs_symmetric(lopsided):
    """The point of the exercise: mirror aspects must now agree."""
    out = symmetrize(lopsided, axis=1, keep_negative=True)
    az = np.array([13.0, 47.0, 95.0, 148.0])
    a = monostatic_rcs(out, 10e9, az, 0.0, pols=("VV",))
    b = monostatic_rcs(out, 10e9, (360 - az) % 360, 0.0, pols=("VV",))
    assert np.allclose(a.sigma["VV"], b.sigma["VV"], rtol=1e-6,
                       atol=1e-9 * a.sigma["VV"].max())


def test_symmetrize_keeps_the_kept_half(lopsided):
    """Mirroring the side *with* the lump must keep the lump, on both sides."""
    keep_lump = symmetrize(lopsided, axis=1, keep_negative=False)
    drop_lump = symmetrize(lopsided, axis=1, keep_negative=True)
    assert keep_lump.volume > lopsided.volume
    assert drop_lump.volume < lopsided.volume


def test_cant_tilts_vertical_panels_off_the_horizon():
    box = shapes.box(4.0, 2.0, 1.0)
    before = census(box)
    out, moved, n_moved = cant(box, threshold_deg=12.0, cant_deg=15.0)
    after = census(out)
    assert moved and n_moved > 0
    assert after[2.0][1] < 0.02 * before[2.0][1]
    assert out.n_faces == box.n_faces          # topology untouched
    assert out.is_closed and out.volume > 0


def test_cant_leaves_an_already_canted_body_alone():
    """Nothing within the threshold means nothing to do."""
    body = shapes.hopeless_diamond()            # every normal is well off the horizon
    out, moved, n_moved = cant(body, threshold_deg=12.0, cant_deg=15.0)
    assert not moved and n_moved == 0
    assert np.allclose(out.vertices, body.vertices)


def test_cant_reduces_the_horizon_flash():
    """A vertical slab is a mirror on the horizon; canting must quiet it."""
    box = shapes.box(3.0, 3.0, 1.2)
    canted, _, _ = cant(box, threshold_deg=12.0, cant_deg=20.0)
    az = np.arange(0.0, 360.0, 3.0)
    before = monostatic_rcs(box, 10e9, az, 0.0, pols=("VV",)).sigma["VV"]
    after = monostatic_rcs(canted, 10e9, az, 0.0, pols=("VV",)).sigma["VV"]
    assert after.max() < before.max() * 1e-2   # at least 20 dB off the peak


def test_panel_groups_require_connection():
    """Two coplanar faces that do not touch are two panels, not one.

    Keying on the plane alone would let a 'panel' span the whole body while
    carrying almost no area, and rotating it about its centroid would move
    metal by metres.
    """
    from cant_panels import panel_groups

    pair = shapes.box(1.0, 1.0, 1.0) + shapes.box(1.0, 1.0, 1.0).translated(
        [3.0, 0.0, 0.0])
    group = panel_groups(pair)
    top = np.flatnonzero(np.isclose(pair.face_normals[:, 2], 1.0))
    assert len(top) >= 2
    left = pair.face_centroids[top, 0] < 1.5
    assert len(np.unique(group[top[left]])) == 1
    assert len(np.unique(group[top[~left]])) == 1
    assert group[top[left]][0] != group[top[~left]][0]


def test_sweep_turns_forward_faces_to_the_planform_angle():
    from cant_panels import axis_census, sweep_to_planform

    box = shapes.box(3.0, 2.0, 1.0)
    before = axis_census(box)
    out, moved, n_moved = sweep_to_planform(box, threshold_deg=20.0,
                                            target_az_deg=36.5)
    after = axis_census(out)
    assert moved and n_moved > 0
    assert after[5.0][1] < 0.05 * before[5.0][1]
    assert out.n_faces == box.n_faces
    assert out.volume > 0


def test_sweep_leaves_an_aligned_body_alone():
    from cant_panels import sweep_to_planform

    body = shapes.faceted_delta()          # nothing faces straight down +x
    out, moved, n_moved = sweep_to_planform(body, threshold_deg=20.0)
    assert not moved and n_moved == 0
    assert np.allclose(out.vertices, body.vertices)


def test_sweep_moves_little_metal():
    """The point is a re-cut panel, not a redesigned airframe."""
    from cant_panels import sweep_to_planform

    box = shapes.box(3.0, 2.0, 1.0)
    out, _, _ = sweep_to_planform(box, threshold_deg=20.0, target_az_deg=36.5)
    moved = np.linalg.norm(out.vertices - box.vertices, axis=1)
    assert moved.max() < 0.5 * box.max_dimension
