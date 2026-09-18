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


def _slab(n=5, half=1.0, thick=0.25):
    """A closed slab whose top face is an n x n grid, so it can hold a pocket."""
    g = np.linspace(-half, half, n)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    top = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, thick)])
    bot = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, -thick)])
    verts = np.vstack([top, bot])
    off = len(top)

    def idx(i, j):
        return i * n + j

    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            faces += [[a, b, c], [a, c, d]]                     # top, +z outward
            faces += [[off + a, off + c, off + b], [off + a, off + d, off + c]]
    for i in range(n - 1):                                      # four side walls
        for a, b in ((idx(i, 0), idx(i + 1, 0)), (idx(n - 1, i), idx(n - 1, i + 1)),
                     (idx(n - 1 - i, n - 1), idx(n - 2 - i, n - 1)),
                     (idx(0, n - 1 - i), idx(0, n - 2 - i))):
            faces += [[a, off + a, off + b], [a, off + b, b]]
    m = Mesh(verts, np.array(faces))
    return m.flipped() if m.volume < 0 else m


def test_fill_recess_restores_a_pocket():
    from fill_recess import fill_recess

    flat = _slab()
    dented = flat.vertices.copy()
    pocket = (np.abs(dented[:, 0]) < 0.6) & (np.abs(dented[:, 1]) < 0.6) \
        & (dented[:, 2] > 0.0)
    assert pocket.sum() >= 4
    dented[pocket, 2] -= 0.18
    dented_mesh = Mesh(dented, flat.faces)
    assert dented_mesh.volume < flat.volume

    out, moved, _ = fill_recess(dented_mesh, (-0.7, 0.7, -0.7, 0.7, 0.0, 0.3),
                                axis=2, side=+1, margin=0.8)
    assert moved >= int(pocket.sum())
    assert out.volume == pytest.approx(flat.volume, rel=1e-6)
    top = out.vertices[out.vertices[:, 2] > 0.0]
    assert np.allclose(top[:, 2], 0.25, atol=1e-9)


def test_fill_recess_leaves_the_surrounding_skin_alone():
    from fill_recess import fill_recess

    flat = _slab()
    out, _, _ = fill_recess(flat, (-0.3, 0.3, -0.3, 0.3, 0.0, 0.3), axis=2, side=+1)
    assert out.volume == pytest.approx(flat.volume, rel=1e-9)
    assert out.n_faces == flat.n_faces


def test_fill_recess_mirrors():
    from fill_recess import fill_recess

    flat = _slab(n=7, half=3.0)
    dented = flat.vertices.copy()
    for sign in (+1, -1):
        sel = (np.abs(dented[:, 0]) < 1.0) & (np.abs(dented[:, 1] - sign * 1.5) < 0.6) \
            & (dented[:, 2] > 0.0)
        dented[sel, 2] -= 0.15
    out, _, _ = fill_recess(Mesh(dented, flat.faces),
                            (-1.1, 1.1, 0.8, 2.2, 0.0, 0.3),
                            axis=2, side=+1, margin=0.9, mirror_axis=1)
    assert out.volume == pytest.approx(flat.volume, rel=1e-6)


def test_pipeline_reorient_is_a_rotation():
    from pipeline import reorient

    body = shapes.faceted_delta()
    out = reorient(body, "zxy")
    assert out.volume == pytest.approx(body.volume, rel=1e-9)
    assert out.total_area == pytest.approx(body.total_area, rel=1e-9)


def test_check_mesh_passes_a_clean_body(tmp_path, capsys):
    from check_mesh import check
    from echo1.meshio import save_obj

    path = tmp_path / "clean.obj"
    save_obj(shapes.faceted_delta(), path)
    assert check(path, 10e9) is True
    out = capsys.readouterr().out
    assert "closed body" in out
    assert "FATAL" not in out


def test_check_mesh_rejects_non_manifold(tmp_path, capsys):
    from check_mesh import check

    # Three triangles sharing one edge.
    with open(tmp_path / "bad.obj", "w") as fh:
        for v in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)):
            fh.write(f"v {v[0]} {v[1]} {v[2]}\n")
        fh.write("f 1 2 3\nf 1 2 4\nf 1 2 5\n")
    assert check(tmp_path / "bad.obj", 10e9) is False
    assert "non-manifold" in capsys.readouterr().out


def test_check_mesh_rejects_zero_area(tmp_path, capsys):
    from check_mesh import check

    with open(tmp_path / "degen.obj", "w") as fh:
        for v in ((0, 0, 0), (1, 0, 0), (2, 0, 0)):
            fh.write(f"v {v[0]} {v[1]} {v[2]}\n")
        fh.write("f 1 2 3\n")
    assert check(tmp_path / "degen.obj", 10e9) is False
    assert "zero-area" in capsys.readouterr().out


def test_check_mesh_flags_inward_normals(tmp_path, capsys):
    from check_mesh import check
    from echo1.meshio import save_obj

    save_obj(shapes.box(2.0, 1.0, 1.0).flipped(), tmp_path / "inside-out.obj")
    check(tmp_path / "inside-out.obj", 10e9)
    assert "normals point inward" in capsys.readouterr().out


def test_check_mesh_flags_an_open_boundary(tmp_path, capsys):
    from check_mesh import check
    from echo1.meshio import save_obj

    save_obj(shapes.plate(1.0, 1.0), tmp_path / "sheet.obj")
    check(tmp_path / "sheet.obj", 10e9)
    assert "open boundary" in capsys.readouterr().out


def test_check_mesh_symmetry_metric():
    from check_mesh import _symmetry

    assert _symmetry(shapes.faceted_delta()) == pytest.approx(1.0)
    lop = shapes.box(4.0, 2.0, 1.0) + shapes.box(1.4, 1.4, 1.4).translated(
        [1.0, 1.7, 0.4])
    assert _symmetry(lop) < 0.95
