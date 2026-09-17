import numpy as np
import pytest

from echo1.geometry import Mesh, normalize


def test_cube_facet_quantities(unit_cube):
    assert unit_cube.n_faces == 12
    assert unit_cube.total_area == pytest.approx(6.0)
    assert unit_cube.volume == pytest.approx(1.0)
    assert unit_cube.is_closed
    # Every outward normal must point away from the centre.
    n = unit_cube.face_normals
    c = unit_cube.face_centroids
    assert np.all(np.einsum("fk,fk->f", n, c) > 0)


def test_cube_wedge_angles(unit_cube):
    n = unit_cube.edges.wedge_n
    assert len(n) == 18
    # 12 true convex edges at 270 degrees exterior, 6 tessellation diagonals.
    assert np.sum(np.isclose(n, 1.5)) == 12
    assert np.sum(np.isclose(n, 1.0)) == 6


def test_coplanar_edges_are_exactly_flat():
    """A split flat panel must give n == 1 exactly, or it diffracts spuriously."""
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    m = Mesh(v, np.array([[0, 1, 2], [0, 2, 3]]), two_sided=True)
    interior = m.edges.wedge_n[m.edges.n_adjacent == 2]
    assert interior == pytest.approx(1.0, abs=0.0)


def test_open_sheet_rim_is_a_knife_edge():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    m = Mesh(v, np.array([[0, 1, 2], [0, 2, 3]]), two_sided=True)
    rim = m.edges.wedge_n[m.edges.n_adjacent == 1]
    assert len(rim) == 4
    assert np.allclose(rim, 2.0)


def test_concave_edge_is_detected():
    """Two plates meeting at a right angle from the inside give n = 1/2."""
    from echo1.shapes import dihedral
    n = dihedral().edges.wedge_n
    assert np.isclose(n.min(), 0.5)


def test_edge_frame_is_right_handed(unit_cube):
    e = unit_cube.edges
    assert np.allclose(np.cross(e.x_hat, e.y_hat), e.e_hat, atol=1e-12)
    assert np.allclose(np.einsum("ek,ek->e", e.x_hat, e.y_hat), 0.0, atol=1e-12)
    assert np.allclose(normalize(e.p1 - e.p0), e.e_hat, atol=1e-12)


def test_rejects_bad_meshes():
    v = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float)
    with pytest.raises(ValueError, match="zero area"):
        Mesh(v, np.array([[0, 1, 2]]))
    with pytest.raises(ValueError, match="shape"):
        Mesh(np.zeros((3, 3)), np.array([[0, 1, 2, 3]]))
    with pytest.raises(ValueError, match="outside"):
        Mesh(np.eye(3), np.array([[0, 1, 9]]))


def test_non_manifold_is_rejected():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], float)
    m = Mesh(v, np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]]))
    with pytest.raises(ValueError, match="non-manifold"):
        _ = m.edges


def test_transforms_preserve_structure(unit_cube):
    moved = unit_cube.translated([3, -2, 1]).rotated([1, 1, 0], 37.0).scaled(2.0)
    assert moved.total_area == pytest.approx(4.0 * unit_cube.total_area)
    assert np.allclose(np.sort(moved.edges.wedge_n), np.sort(unit_cube.edges.wedge_n))


def test_meshes_concatenate(unit_cube):
    both = unit_cube + unit_cube.translated([5, 0, 0])
    assert both.n_faces == 2 * unit_cube.n_faces
    assert both.volume == pytest.approx(2.0)
