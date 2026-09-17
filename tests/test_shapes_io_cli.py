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
