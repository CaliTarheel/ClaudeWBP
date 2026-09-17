"""Physical optics: the closed-form facet integral and its known limits."""

import numpy as np
import pytest

from conftest import FREQ, K, LAMBDA, directions, pol_frame, rect_plate

from echo1 import analytic, shapes
from echo1.po import gordon_integral, gordon_integral_quadrature, po_amplitude
from echo1.solver import monostatic_rcs


@pytest.mark.parametrize("scale, order", [(1e-3, 64), (1.0, 64), (10.0, 96), (200.0, 512)])
def test_gordon_matches_quadrature(scale, order):
    """The closed form must equal a brute-force integration of the same integral."""
    rng = np.random.default_rng(7)
    tri = rng.normal(size=(5, 3, 3))
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    w = rng.normal(size=(4, 3)) * scale
    a = gordon_integral(tri, n, w)
    b = gordon_integral_quadrature(tri, w, order=order)
    assert np.abs(a - b).max() / np.abs(b).max() < 1e-9


def test_gordon_specular_limit_is_the_area():
    """At w = 0 the integral is just the facet area."""
    tri = np.array([[[0., 0, 0], [2, 0, 0], [0, 3, 0]]])
    n = np.array([[0., 0, 1]])
    got = gordon_integral(tri, n, np.zeros((1, 3)))
    assert got[0, 0] == pytest.approx(3.0 + 0j)


def test_flat_plate_broadside_is_four_pi_a_squared():
    a = 0.30
    got = monostatic_rcs(shapes.plate(a, a), FREQ, [0.0], [90.0],
                         pols=("VV", "HH"), use_ptd=False)
    want = analytic.flat_plate_broadside(a * a, LAMBDA)
    for p in ("VV", "HH"):
        assert got.sigma[p][0] == pytest.approx(want, rel=1e-9)


def test_flat_plate_pattern_matches_the_analytic_sinc():
    """Off broadside, PO must follow cos^2 * sinc^2 in the principal plane."""
    a, b = 0.40, 0.25
    theta = np.arange(2.0, 60.0, 0.5)
    plate = rect_plate(a, b)
    got = monostatic_rcs(plate, FREQ, az_deg=0.0, el_deg=90.0 - theta,
                         pols=("HH",), use_ptd=False)
    want = analytic.flat_plate_po(a, b, theta, LAMBDA)
    # Compare where the pattern is not in a null.
    keep = want > want.max() * 1e-6
    assert np.allclose(got.sigma["HH"][keep], want[keep], rtol=1e-6)


def test_monostatic_copol_projection_identity():
    """For backscatter, e_r . (n x (i x e_i)) is exactly n . r -- so PO
    backscatter vanishes smoothly as a facet turns edge-on."""
    d = directions(60, seed=3)
    h, v = pol_frame(d)
    n = np.array([[0.3, -0.5, 0.81]])
    n = n / np.linalg.norm(n)
    for e in (h, v):
        p = np.cross(n[None, :, :], np.cross(-d, e)[:, None, :])
        proj = np.einsum("ak,afk->af", e, p)[:, 0]
        assert np.allclose(proj, d @ n[0], atol=1e-12)


def test_po_has_no_monostatic_cross_polarisation():
    """A known property of physical optics, and a check on the vector algebra."""
    plate = rect_plate(0.4, 0.3)
    got = monostatic_rcs(plate, FREQ, np.arange(0, 180, 7.0), 30.0,
                         pols=("HV", "VH"), use_ptd=False)
    assert np.all(got.sigma["HV"] < 1e-24)
    assert np.all(got.sigma["VH"] < 1e-24)


def test_sphere_approaches_the_optical_limit():
    r = 0.5
    got = monostatic_rcs(shapes.sphere(r, subdivisions=4), FREQ, [0.0], [0.0],
                         pols=("VV",), use_ptd=False, occlusion=False)
    assert analytic.to_dbsm(got.sigma["VV"][0]) == pytest.approx(
        analytic.to_dbsm(analytic.sphere_optical(r)), abs=1.0)


def test_splitting_a_facet_changes_nothing():
    """Physical optics is an integral over the surface: how it is diced is
    irrelevant, and the extra coplanar edges must not radiate."""
    a = 0.5
    coarse = rect_plate(a, a)
    v = np.array([[-a/2, -a/2, 0], [a/2, -a/2, 0], [a/2, a/2, 0],
                  [-a/2, a/2, 0], [0, 0, 0]], float)
    from echo1.geometry import Mesh
    fine = Mesh(v, np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]),
                two_sided=True)
    az = np.arange(0, 90, 3.0)
    args = dict(pols=("VV", "HH"), use_ptd=True)
    a1 = monostatic_rcs(coarse, FREQ, az, 35.0, **args)
    a2 = monostatic_rcs(fine, FREQ, az, 35.0, **args)
    for p in ("VV", "HH"):
        assert np.allclose(a1.sigma[p], a2.sigma[p], rtol=1e-8, atol=1e-18)
