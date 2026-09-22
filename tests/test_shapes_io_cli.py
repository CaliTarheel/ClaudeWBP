"""Shape library, mesh files and the command line."""

import numpy as np
import pytest

from echo1 import shapes
from echo1.cli import main
from echo1.meshio import load_mesh, save_mesh


@pytest.mark.parametrize("name", sorted(shapes.BUILTIN))
def test_every_builtin_shape_is_sane(name):
    m = shapes.BUILTIN[name]()
    assert m.n_faces > 0
    assert len(m.edges) > 0
    assert np.all(np.isfinite(m.vertices))
    assert m.total_area > 0
    if m.is_closed:
        assert m.volume > 0, "closed bodies must be wound outward"


def test_aircraft_shapes_are_closed_solids():
    for fn in (shapes.hopeless_diamond, shapes.faceted_delta):
        m = fn()
        assert m.is_closed
        assert m.volume > 0


def test_faceted_delta_is_planform_aligned():
    """Every planform edge must lie along one of just two directions."""
    e = shapes.faceted_delta().edges
    flat = (np.abs(e.p0[:, 2]) < 1e-9) & (np.abs(e.p1[:, 2]) < 1e-9)
    d = e.e_hat[flat]
    angle = np.degrees(np.arctan2(np.abs(d[:, 1]), np.abs(d[:, 0])))
    assert len(np.unique(np.round(angle, 6))) == 1


def test_corner_reflectors_are_concave():
    for fn in (shapes.dihedral, shapes.trihedral):
        assert np.isclose(fn().edges.wedge_n.min(), 0.5)


@pytest.mark.parametrize("ext", [".obj", ".stl"])
def test_mesh_round_trip(tmp_path, ext):
    original = shapes.hopeless_diamond()
    path = tmp_path / f"model{ext}"
    save_mesh(original, path)
    back = load_mesh(path)
    assert back.n_faces == original.n_faces
    assert back.total_area == pytest.approx(original.total_area, rel=1e-6)
    assert back.volume == pytest.approx(original.volume, rel=1e-6)


def test_ascii_stl_round_trip(tmp_path):
    original = shapes.box(1.0, 2.0, 3.0)
    path = tmp_path / "model.stl"
    save_mesh(original, path, binary=False)
    back = load_mesh(path)
    assert back.volume == pytest.approx(original.volume, rel=1e-6)


def test_unsupported_format(tmp_path):
    with pytest.raises(ValueError, match="unsupported"):
        load_mesh(tmp_path / "model.ply")


def test_cli_shapes_and_info(capsys):
    assert main(["shapes"]) == 0
    assert "hopeless-diamond" in capsys.readouterr().out
    assert main(["info", "--shape", "box"]) == 0
    assert "facets" in capsys.readouterr().out


def test_cli_check(capsys):
    assert main(["check"]) == 0
    assert "flat plate" in capsys.readouterr().out


def test_cli_sweep_writes_csv(tmp_path, capsys):
    out = tmp_path / "rcs.csv"
    code = main(["sweep", "--shape", "hopeless-diamond", "--freq", "10GHz",
                 "--az", "0:360:5", "--pol", "VV,HH", "--csv", str(out)])
    assert code == 0
    assert out.exists()
    assert len(out.read_text().strip().splitlines()) == 2 + 72
    assert "peak" in capsys.readouterr().out


def test_cli_sweep_from_a_file(tmp_path):
    path = tmp_path / "m.obj"
    save_mesh(shapes.faceted_delta(), path)
    assert main(["sweep", "--mesh", str(path), "--az", "0:90:30",
                 "--pol", "VV", "--no-occlusion"]) == 0


def test_cli_rejects_unknown_shape():
    with pytest.raises(SystemExit):
        main(["info", "--shape", "flying-saucer"])


def test_polygon_plate_rim_and_interior():
    """Only the outline may diffract; the fan triangulation must be invisible."""
    m = shapes.polygon_plate([(0.8, 0), (0, 0.5), (-0.8, 0), (0, -0.5)])
    assert m.total_area == pytest.approx(0.8, rel=1e-12)
    assert np.sum(m.edges.n_adjacent == 1) == 4
    assert np.allclose(m.edges.wedge_n[m.edges.n_adjacent == 2], 1.0)


def test_polygon_plate_rejects_bad_outlines():
    with pytest.raises(ValueError, match="three"):
        shapes.polygon_plate([(0, 0), (1, 1)])


def test_delta_tip_fraction_controls_alignment():
    aligned = shapes.faceted_delta(tip_frac=0.5).edges
    skewed = shapes.faceted_delta(tip_frac=0.3).edges

    def planform_directions(e):
        flat = (np.abs(e.p0[:, 2]) < 1e-9) & (np.abs(e.p1[:, 2]) < 1e-9)
        d = e.e_hat[flat]
        return np.unique(np.round(
            np.degrees(np.arctan2(np.abs(d[:, 1]), np.abs(d[:, 0]))), 6))

    assert len(planform_directions(aligned)) == 1
    assert len(planform_directions(skewed)) == 2
    with pytest.raises(ValueError, match="tip_frac"):
        shapes.faceted_delta(tip_frac=1.5)


def test_transom_rake_removes_the_tail_flash():
    """The example's headline number, pinned as a test."""
    from echo1.solver import monostatic_rcs
    az = np.arange(0.0, 360.0, 0.5)
    flat = monostatic_rcs(shapes.hopeless_diamond(transom_rake=0.0), 10e9, az,
                          pols=("VV",))
    raked = monostatic_rcs(shapes.hopeless_diamond(), 10e9, az, pols=("VV",))
    gain_db = 10 * np.log10(flat.sigma["VV"].max() / raked.sigma["VV"].max())
    assert gain_db > 30.0
    # Every facet of the raked body points well out of the horizontal plane.
    n = shapes.hopeless_diamond().face_normals
    assert np.degrees(np.arcsin(np.abs(n[:, 2]))).min() > 30.0


# ---------------------------------------------------------------- the fighter

def test_faceted_fighter_is_a_clean_solid():
    m = shapes.faceted_fighter()
    assert m.is_closed
    assert m.volume > 0
    assert m.edges.wedge_n.min() > 0.6, "nothing should be too re-entrant to model"
    assert np.allclose(np.sort(m.vertices[:, 1]), -np.sort(-m.vertices[:, 1])[::-1])


def test_faceted_fighter_puts_every_planform_edge_on_two_directions():
    """Chine and sawtooth must share the same two plan directions."""
    e = shapes.faceted_fighter().edges
    flat = (np.abs(e.p0[:, 2]) < 1e-9) & (np.abs(e.p1[:, 2]) < 1e-9)
    d = e.e_hat[flat]
    angle = np.degrees(np.arctan2(np.abs(d[:, 1]), np.abs(d[:, 0])))
    assert len(np.unique(np.round(angle, 6))) == 1


def test_the_planform_azimuths_carry_more_edge_than_anything_else():
    """The rule's whole claim: four azimuths take the lot."""
    top = shapes.spike_azimuths(shapes.faceted_fighter())[:4]
    assert sorted(round(a, 1) for a, _ in top) == [54.9, 125.1, 234.9, 305.1]


def test_fin_alignment_moves_the_lobes_without_changing_the_aircraft():
    aligned = shapes.spike_azimuths(shapes.faceted_fighter())
    loose = shapes.spike_azimuths(shapes.faceted_fighter(fin_align=False))
    assert max(w for a, w in aligned if abs(a - 125.07) < 1.0) > \
        max(w for a, w in loose if abs(a - 125.07) < 1.0)
    assert max(w for a, w in loose if abs(a - 90.0) < 1.0) > \
        max(w for a, w in aligned if abs(a - 90.0) < 1.0)


def test_an_apex_outside_the_kernel_is_refused():
    """A fan from a point only tiles a polygon the point can see all of."""
    with pytest.raises(ValueError, match="kernel"):
        shapes.faceted_fighter(apex_frac=0.28)
    shapes.faceted_fighter(teeth=1, apex_frac=0.28)          # convex: fine


def test_fins_taper_to_a_sharp_rim():
    """No narrow flat strip may survive round a fin: strips mirror hard."""
    m = shapes.faceted_fighter()
    el = np.degrees(np.arcsin(np.abs(m.face_normals[:, 2])))
    assert el.min() > 20.0, "every facet must mirror well off the horizon"


def test_spike_azimuths_predicts_where_the_solver_finds_the_lobes():
    from echo1.plotting import to_dbsm
    from echo1.solver import monostatic_rcs
    m = shapes.faceted_fighter()
    az = np.arange(0.0, 360.0, 0.02)
    db = to_dbsm(monostatic_rcs(m, 10e9, az, pols=("VV",)).sigma["VV"])
    for target, _ in shapes.spike_azimuths(m)[:6]:
        near = np.abs((az - target + 180.0) % 360.0 - 180.0) <= 0.5
        assert db[near].max() > np.median(db) + 25.0, f"no lobe at {target}"


def test_spike_azimuths_weighting_ranks_a_crease_below_a_knife():
    m = shapes.faceted_fighter()
    by_length = dict(np.round(shapes.spike_azimuths(m, weight="length"), 3))
    by_fringe = dict(np.round(shapes.spike_azimuths(m), 3))
    beam = min(by_length, key=lambda a: abs(a - 90.0))
    planform = min(by_length, key=lambda a: abs(a - 125.07))
    assert by_length[beam] < by_length[planform]
    assert by_fringe[beam] / by_fringe[planform] < by_length[beam] / by_length[planform]


def test_spike_azimuths_merges_a_cluster_straddling_zero():
    """Two edges either side of due north are one lobe, not two."""
    m = shapes.plate(4.0, 4.0).rotated(np.array([0.0, 0.0, 1.0]),
                                       np.radians(0.5))
    out = shapes.spike_azimuths(m, tol_deg=2.0)
    assert len(out) == 4, out
    assert any(min(a, 360.0 - a) < 1.0 for a, _ in out), out


def test_specular_aspects_matches_the_flat_plate_formula():
    """Coplanar triangles must be gathered before the area is squared."""
    m = shapes.plate(2.0, 3.0)
    (az, el, area, peak), = shapes.specular_aspects(m)
    assert area == pytest.approx(6.0)
    assert el == pytest.approx(90.0)
    lam = 299792458.0 / 10e9
    assert peak == pytest.approx(10.0 * np.log10(4.0 * np.pi * 36.0 / lam ** 2))


def test_specular_bundling_adds_parallel_panels_that_do_not_touch():
    """Faces pointed the same way flash together whether or not they meet."""
    a = shapes.plate(2.0, 2.0)
    b = shapes.plate(2.0, 2.0).translated(np.array([0.0, 5.0, 0.0]))
    both = shapes.Mesh(np.vstack([a.vertices, b.vertices]),
                       np.vstack([a.faces, b.faces + len(a.vertices)]),
                       two_sided=True, name="two-plates")
    alone = shapes.specular_aspects(both)
    bundled = shapes.specular_aspects(both, bundle_deg=1.0)
    assert len(alone) == 2 and len(bundled) == 1
    assert bundled[0][2] == pytest.approx(8.0)
    assert bundled[0][3] == pytest.approx(alone[0][3] + 20.0 * np.log10(2.0))


def test_spike_azimuths_leaves_out_what_the_solver_cannot_carry():
    """The prediction must rank what will actually be solved, and no more."""
    m = shapes.dihedral()                       # its seam is a right-angle corner
    assert m.edges.wedge_n.min() == pytest.approx(0.5)
    seam = m.edges.length[m.edges.wedge_n < 0.6].sum()
    kept = sum(w for _, w in shapes.spike_azimuths(m, min_length=0.1))
    everything = sum(w for _, w in
                     shapes.spike_azimuths(m, min_length=0.1, min_wedge_n=0.0))
    # a right-angle wedge is as strong as a knife, and it flashes both ways
    assert everything - kept == pytest.approx(2.0 * seam, rel=1e-6)
