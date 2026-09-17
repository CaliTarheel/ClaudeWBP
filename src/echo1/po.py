"""Physical optics: the specular half of the ECHO calculation.

Physical optics (PO) replaces the true induced surface current with the
tangent-plane approximation ``J = 2 n x H_i`` on the lit part of the body and
zero elsewhere, then radiates it.  On a *flat* facet that radiation integral

.. math::  I = \\int_S e^{j \\mathbf{w}\\cdot\\mathbf{r}}\\,dS,
           \\qquad \\mathbf{w} = k(\\hat{s} - \\hat{\\imath})

has a closed form (Gordon 1975), so no surface meshing or quadrature is
needed: each flat panel costs one contour sum over its three edges.  This is
precisely why ECHO 1 required shapes "made from flat panels" -- and why the
aircraft that came out of it was faceted.

Derivation of the closed form
-----------------------------
Split ``w`` into the facet normal component and the in-plane remainder
``w_t``.  Only ``w_t`` varies over the facet, so with the 2-D divergence
theorem applied to ``div(u e^{j q u.rho} / (j q)) = e^{j q u.rho}``:

.. math::
   I = e^{j\\mathbf{w}\\cdot\\mathbf{r}_0}\\,\\frac{1}{j|\\mathbf{w}_t|^2}
       \\sum_m \\bigl[\\mathbf{w}_t\\cdot(\\boldsymbol{\\ell}_m\\times\\hat{n})\\bigr]\\,
       e^{j\\mathbf{w}_t\\cdot\\boldsymbol{\\rho}_m}\\,
       \\mathrm{sinc}\\!\\left(\\tfrac{1}{2}\\mathbf{w}_t\\cdot\\boldsymbol{\\ell}_m\\right)

summed over the edges ``l_m`` in counter-clockwise order about ``n``, with
``rho_m`` the edge midpoint relative to ``r_0``.  As ``|w_t| -> 0`` (the
specular direction) this tends to the facet area.

Radar cross section
-------------------
With ``P = n x (i x e_p)`` and receive polarisation ``e_r``,

.. math::  \\sigma_{rp} = \\frac{k^2}{\\pi}\\,|\\hat{e}_r\\cdot P|^2\\,|I|^2 .

For a flat plate at normal incidence this reduces to the textbook
``sigma = 4 pi A^2 / lambda^2``; :mod:`echo1.analytic` and the test suite
check it.
"""

from __future__ import annotations

import numpy as np

from .geometry import Mesh

__all__ = ["gordon_integral", "gordon_integral_quadrature", "po_amplitude"]

# |w_t| * (facet size) below this is treated as the specular limit.  Balances
# series truncation against cancellation between the edge terms.
_WT_TOL = 1e-7


def _sinc(x: np.ndarray) -> np.ndarray:
    """sin(x)/x, continuous at 0."""
    return np.sinc(x / np.pi)


def gordon_integral(
    tri: np.ndarray,
    normals: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """Closed-form ``int_S exp(j w.r) dS`` over flat triangles.

    Parameters
    ----------
    tri : (F, 3, 3) array
        Triangle vertices, wound counter-clockwise about ``normals``.
    normals : (F, 3) array
        Unit facet normals.
    w : (A, 3) array
        Spatial frequency vectors ``k (s_hat - i_hat)``, one per look angle.

    Returns
    -------
    (A, F) complex array
    """
    tri = np.asarray(tri, dtype=float)
    normals = np.asarray(normals, dtype=float)
    w = np.atleast_2d(np.asarray(w, dtype=float))

    r0 = tri[:, 0, :]                                    # (F, 3) reference vertex
    wn = w @ normals.T                                   # (A, F)
    wt = w[:, None, :] - wn[..., None] * normals[None]   # (A, F, 3)
    wt2 = np.einsum("afk,afk->af", wt, wt)               # (A, F)

    phase0 = np.exp(1j * (w @ r0.T))                     # (A, F)
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )                                                    # (F,)

    # Facet size, used to make the specular cut-off scale-free.
    size = np.max(np.linalg.norm(tri - r0[:, None, :], axis=2), axis=1)  # (F,)
    specular = np.sqrt(wt2) * size[None, :] < _WT_TOL

    acc = np.zeros(wt2.shape, dtype=complex)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        edge = tri[:, b, :] - tri[:, a, :]                       # (F, 3)
        rho = 0.5 * (tri[:, b, :] + tri[:, a, :]) - r0           # (F, 3) midpoint
        weight = np.einsum("afk,fk->af", wt, np.cross(edge, normals))
        wt_dot_edge = np.einsum("afk,fk->af", wt, edge)
        wt_dot_rho = np.einsum("afk,fk->af", wt, rho)
        acc += weight * np.exp(1j * wt_dot_rho) * _sinc(0.5 * wt_dot_edge)

    safe = np.where(specular, 1.0, wt2)
    result = phase0 * acc / (1j * safe)
    return np.where(specular, phase0 * area[None, :], result)


def gordon_integral_quadrature(
    tri: np.ndarray,
    w: np.ndarray,
    order: int = 64,
) -> np.ndarray:
    """Brute-force quadrature reference for :func:`gordon_integral`.

    Used only by the test suite to confirm the closed form.  Maps a tensor
    Gauss-Legendre rule onto the triangle by the Duffy transform.
    """
    tri = np.asarray(tri, dtype=float)
    w = np.atleast_2d(np.asarray(w, dtype=float))

    x, wx = np.polynomial.legendre.leggauss(order)
    x = 0.5 * (x + 1.0)
    wx = 0.5 * wx
    u, v = np.meshgrid(x, x, indexing="ij")
    wu, wv = np.meshgrid(wx, wx, indexing="ij")
    # Duffy: (u, v) in unit square -> barycentric (u, v(1-u)) with Jacobian (1-u).
    l1 = u.ravel()
    l2 = (v * (1.0 - u)).ravel()
    l0 = 1.0 - l1 - l2
    jac = (1.0 - u).ravel() * (wu * wv).ravel()

    pts = (
        l0[None, :, None] * tri[:, None, 0, :]
        + l1[None, :, None] * tri[:, None, 1, :]
        + l2[None, :, None] * tri[:, None, 2, :]
    )                                                    # (F, Q, 3)
    area2 = np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )                                                    # (F,) = 2 * area
    phase = np.exp(1j * np.einsum("ak,fqk->afq", w, pts))
    return np.einsum("afq,q->af", phase, jac) * area2[None, :]


def po_amplitude(
    mesh: Mesh,
    i_hat: np.ndarray,
    s_hat: np.ndarray,
    e_inc: np.ndarray,
    e_rec: np.ndarray,
    k: float,
    lit: np.ndarray,
) -> np.ndarray:
    """Per-facet physical-optics scattering amplitude ``S``, with ``sigma = |sum S|^2``.

    Parameters
    ----------
    mesh : Mesh
    i_hat : (A, 3) array
        Incident propagation direction (from radar towards target).
    s_hat : (A, 3) array
        Scattering direction (towards receiver); ``-i_hat`` for monostatic.
    e_inc, e_rec : (A, 3) arrays
        Transmit and receive polarisation unit vectors.
    k : float
        Wavenumber ``2 pi / lambda``.
    lit : (A, F) bool array
        Facet illumination mask.

    Returns
    -------
    (A, F) complex array
        Amplitudes in units of sqrt(m^2); zero where ``lit`` is False.
    """
    tri = mesh.triangles
    normals = mesh.face_normals

    w = k * (s_hat - i_hat)                              # (A, 3)
    integral = gordon_integral(tri, normals, w)          # (A, F)

    # P = n x (i x e_inc): direction of the PO surface current's radiation.
    i_cross_e = np.cross(i_hat, e_inc)                   # (A, 3)
    p_vec = np.cross(normals[None, :, :], i_cross_e[:, None, :])   # (A, F, 3)
    proj = np.einsum("ak,afk->af", e_rec, p_vec)         # (A, F)

    # The PO current sits on whichever face the radar lights, so for a
    # two-sided sheet lit from behind the stored normal both P and the winding
    # of the Gordon contour flip.  Only P's flip survives in the product.
    side = np.sign(-np.einsum("ak,fk->af", i_hat, normals))
    side = np.where(side == 0.0, 1.0, side)

    amp = (k / np.sqrt(np.pi)) * side * proj * integral
    return np.where(lit, amp, 0.0)
