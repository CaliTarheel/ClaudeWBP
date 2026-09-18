"""The Ufimtsev edge-wave term: its defining property, and its calibration."""

import numpy as np
import pytest

from conftest import FREQ, K, directions, pol_frame, rect_plate

from echo1.po import po_amplitude
from echo1.ptd import fringe_coefficients, po_edge_coefficients, ptd_amplitude
from echo1.solver import monostatic_rcs


@pytest.mark.parametrize("n", [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
def test_fringe_coefficients_are_finite_everywhere(n):
    """PTD's whole point: the fringe field stays finite at the reflection and
    shadow boundaries, where both the total and the PO coefficients blow up."""
    phi_p = np.linspace(1e-3, n * np.pi - 1e-3, 1501)
    for offset in np.linspace(-np.pi, np.pi, 151):
        phi = np.clip(phi_p + offset, 1e-6, n * np.pi - 1e-6)
        f, g = fringe_coefficients(phi, phi_p, np.full_like(phi, n))
        assert np.all(np.isfinite(f)) and np.all(np.isfinite(g))


def test_coplanar_wedge_does_not_diffract():
    """n == 1 is not a wedge at all, so a tessellated flat panel is safe."""
    phi = np.array([0.3, 1.0, 2.5, 3.0])
    phi_p = np.array([0.7, 2.0, 0.4, 1.1])
    f, g = fringe_coefficients(phi, phi_p, np.ones_like(phi))
    assert np.allclose(f, 0.0) and np.allclose(g, 0.0)


@pytest.mark.parametrize("a,b", [(0.20, 0.20), (0.35, 0.18), (0.41, 0.29)])
def test_edge_machinery_reproduces_exact_po_on_a_rectangle(a, b):
    """The calibration that fixes the equivalent-edge-current constant.

    Driving the edge kernel with the *PO* coefficients instead of the fringe
    ones must give back the exact Gordon integral over the plate -- amplitude
    and sign.  It is an identity for a rectangle, so any error in the constant,
    the sign, the obliquity power or the azimuth convention shows up at once.
    """
    plate = rect_plate(a, b)
    d = directions(80, seed=11)
    h, v = pol_frame(d)
    for e in (h, v):
        po = po_amplitude(plate, -d, d, e, e, K,
                          np.ones((len(d), plate.n_faces), bool)).sum(axis=1)
        edge = ptd_amplitude(plate.edges, -d, d, e, e, K,
                             np.ones((len(d), len(plate.edges)), bool),
                             po_only=True).sum(axis=1)
        assert np.allclose(edge, po, rtol=1e-9, atol=1e-12 * np.abs(po).max())


def test_rotating_the_body_rotates_the_answer():
    """Nothing in the edge kernel may depend on the global axes."""
    plate = rect_plate(0.33, 0.21)
    az = np.arange(0.0, 360.0, 11.0)
    base = monostatic_rcs(plate, FREQ, az, 25.0, pols=("VV", "HH"))
    spun = monostatic_rcs(plate.rotated([0, 0, 1], 40.0), FREQ, az + 40.0, 25.0,
                          pols=("VV", "HH"))
    for p in ("VV", "HH"):
        assert np.allclose(base.sigma[p], spun.sigma[p], rtol=1e-7,
                           atol=1e-9 * base.sigma[p].max())


def test_ptd_fills_in_the_physical_optics_nulls():
    """Away from specular, facet-only physical optics collapses into deep
    nulls that are not real; the edge waves are what fill them.

    The effect grows with the angle off the specular lobe, so this looks at
    50-85 degrees off the plate normal, where physical optics is worst.
    """
    plate = rect_plate(0.30, 0.30)
    el = np.arange(5.0, 40.0, 0.25)          # 50 to 85 degrees off normal
    got = monostatic_rcs(plate, FREQ, 0.0, el, pols=("HH",))
    po_only, total = got.sigma_po["HH"], got.sigma["HH"]
    assert total.mean() > 3.0 * po_only.mean()
    assert np.median(total) > 5.0 * np.median(po_only)
    assert total.min() > 100.0 * po_only.min()


def test_ptd_matters_less_near_specular():
    """Near the specular flash the facet term dominates and PTD is a small
    correction -- the complement of the test above, and a check that the edge
    term is not simply swamping everything."""
    plate = rect_plate(0.30, 0.30)
    el = np.arange(85.0, 90.0, 0.1)          # within 5 degrees of broadside
    got = monostatic_rcs(plate, FREQ, 0.0, el, pols=("HH",))
    ratio = got.sigma["HH"].mean() / got.sigma_po["HH"].mean()
    assert 0.8 < ratio < 1.25


def test_monostatic_cross_polarisation_is_reciprocal():
    body = rect_plate(0.4, 0.25).rotated([1, 2, 0.5], 31.0)
    got = monostatic_rcs(body, FREQ, np.arange(0, 360, 9.0), 27.0,
                         pols=("HV", "VH"))
    assert np.allclose(got.sigma["HV"], got.sigma["VH"], rtol=1e-9,
                       atol=1e-12 * got.sigma["HV"].max())


def test_po_edge_coefficients_lit_face_selection():
    """The o face is lit from above, the n face from below, and a knife edge
    is lit from exactly one side at a time."""
    phi = np.array([0.5, 0.5])
    f_top, _ = po_edge_coefficients(phi, np.array([1.0, 1.0]), np.array([2.0, 2.0]))
    f_bot, _ = po_edge_coefficients(phi, np.array([4.0, 4.0]), np.array([2.0, 2.0]))
    assert np.all(np.isfinite(f_top)) and np.all(np.isfinite(f_bot))
    assert not np.allclose(f_top, f_bot)


def test_reentrant_wedges_are_excluded():
    """A strongly re-entrant corner is outside single-bounce PTD, and the
    Keller coefficient grows without physical meaning there, so such edges
    must not be allowed into the sum."""
    from echo1.geometry import Mesh
    from echo1.ptd import diffracting

    # A narrow V: two plates meeting at a small exterior angle.
    half = np.radians(20.0)
    v = np.array([
        [0.0, -0.5, 0.0], [0.0, 0.5, 0.0],
        [np.cos(half), -0.5, np.sin(half)], [np.cos(half), 0.5, np.sin(half)],
        [np.cos(half), -0.5, -np.sin(half)], [np.cos(half), 0.5, -np.sin(half)],
    ])
    m = Mesh(v, np.array([[0, 1, 3], [0, 3, 2], [0, 4, 5], [0, 5, 1]]),
             two_sided=True)
    shared = m.edges.wedge_n < 0.5
    assert shared.any(), "expected a re-entrant edge in this geometry"
    assert not diffracting(m.edges)[shared].any()


def test_excluded_edges_are_reported():
    """Dropping geometry silently would be worse than not dropping it."""
    from echo1.geometry import Mesh

    half = np.radians(20.0)
    v = np.array([
        [0.0, -0.5, 0.0], [0.0, 0.5, 0.0],
        [np.cos(half), -0.5, np.sin(half)], [np.cos(half), 0.5, np.sin(half)],
        [np.cos(half), -0.5, -np.sin(half)], [np.cos(half), 0.5, -np.sin(half)],
    ])
    m = Mesh(v, np.array([[0, 1, 3], [0, 3, 2], [0, 4, 5], [0, 5, 1]]),
             two_sided=True)
    got = monostatic_rcs(m, FREQ, np.arange(0, 360, 45.0), pols=("VV",))
    count, length = got.excluded_edges
    assert count >= 1 and length > 0.0


def test_min_wedge_n_is_tunable():
    from echo1.geometry import Mesh
    from echo1.ptd import diffracting

    m = rect_plate(0.3, 0.2)
    assert diffracting(m.edges, min_wedge_n=0.5).sum() == 4      # the four rim edges
    assert diffracting(m.edges, min_wedge_n=2.5).sum() == 0


@pytest.mark.parametrize("n", [0.5, 0.501, 0.55])
def test_right_angle_reentrant_corners_are_excluded(n):
    """A right-angle re-entrant corner is a dihedral retroreflector: the
    backscatter coefficient has a pole in the retroreflection direction, where
    the real mechanism is the double bounce this model does not carry."""
    from echo1.geometry import EdgeSet
    from echo1.ptd import diffracting

    e = EdgeSet(
        p0=np.zeros((1, 3)), p1=np.array([[0.0, 0.0, 1.0]]),
        e_hat=np.array([[0.0, 0.0, 1.0]]), length=np.array([1.0]),
        x_hat=np.array([[1.0, 0.0, 0.0]]), y_hat=np.array([[0.0, 1.0, 0.0]]),
        wedge_n=np.array([n]), faces=np.array([[0, 1]]), n_adjacent=np.array([2]),
    )
    assert not diffracting(e).any()


def test_convex_right_angle_still_diffracts():
    """The guard must not take the ordinary convex edges with it."""
    from echo1.ptd import diffracting
    from echo1.shapes import box
    n = box().edges.wedge_n
    assert diffracting(box().edges)[np.isclose(n, 1.5)].all()


def test_dihedral_corners_are_reported():
    from echo1.shapes import dihedral
    from echo1.solver import dihedral_corners
    count, length = dihedral_corners(dihedral())
    assert count >= 1 and length > 0.0
