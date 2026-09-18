"""Ufimtsev's physical theory of diffraction: the edge waves ECHO was built on.

Physical optics alone predicts big specular flashes and essentially nothing
between them, which is wrong -- a faceted body tilted away from the radar
still scatters, from its *edges*.  Pyotr Ufimtsev's 1962 monograph *Method of
Edge Waves in the Physical Theory of Diffraction* (translated by the USAF
Foreign Technology Division in 1971) supplied the missing term, and Denys
Overholser's ECHO 1 was the code that put it to work on aircraft shapes.

PTD writes the true edge current as the PO current plus a *fringe* current
concentrated at the edge, and radiates the difference:

.. math::  f = f^{\\text{total}} - f^{\\text{PO}}, \\qquad
           g = g^{\\text{total}} - g^{\\text{PO}}

for the soft (E parallel to the edge) and hard (H parallel to the edge)
polarisations.  For a wedge of exterior angle ``n * pi``, with ``mu = pi / n``,
the exact (Keller) coefficients are

.. math::
   f,g = \\frac{\\sin\\mu}{n}\\left[
       \\frac{1}{\\cos\\mu - \\cos\\frac{\\phi-\\phi'}{n}}
       \\mp \\frac{1}{\\cos\\mu - \\cos\\frac{\\phi+\\phi'}{n}}\\right]

and the PO part contributed by a single illuminated face is obtained from the
endpoint of the PO surface integral over that face,

.. math::  f^{\\text{PO}} = \\frac{\\sin\\phi'}{\\cos\\phi + \\cos\\phi'},
           \\qquad
           g^{\\text{PO}} = \\frac{-\\sin\\phi}{\\cos\\phi + \\cos\\phi'},

summed over whichever faces the radar actually lights (the *n* face uses the
mirrored angles ``n*pi - phi``, ``n*pi - phi'``).  The normalisation above is
fixed, with no free constant, by requiring the poles of the two parts to
cancel at the reflection and shadow boundaries -- which is exactly PTD's
defining property, that the fringe field is finite everywhere.  Two coplanar
facets give ``n = 1`` and ``f = g = 0``, so tessellating a flat panel into
triangles introduces no spurious edges.

The fringe coefficients are turned into equivalent electric and magnetic edge
currents and radiated along each straight edge.  For *monostatic* scattering
the backscatter direction always lies on the Keller cone of every edge, so the
on-cone coefficients used here are the appropriate ones; for bistatic
geometries well off the cone they degrade to an approximation, which
:mod:`echo1.solver` warns about.
"""

from __future__ import annotations

import numpy as np

from .geometry import EdgeSet, normalize

__all__ = ["fringe_coefficients", "po_edge_coefficients", "ptd_amplitude", "diffracting"]

# Denominator magnitude below which a reflection/shadow boundary is assumed and
# the (finite) fringe coefficient is recovered by a symmetric average.
_SING_TOL = 1e-6
_SING_DELTA = 1e-4
# Looking along an edge is a caustic of this theory.  The equivalent-edge-
# current amplitude carries 1 / sin^2(beta0), where beta0 is the angle between
# the edge and the line of sight, so as the line of sight swings onto the edge
# the formulation does not merely lose accuracy, it diverges -- a 17 mm edge on
# a real airframe was found radiating like two square metres.  Near end-on
# there is no edge wave to speak of anyway: the edge presents almost no
# projected length, and what is left is a tip effect this model does not carry.
# So the contribution is tapered smoothly to zero below _SIN_BETA_TAPER, which
# also caps the amplification at 1 / _SIN_BETA_FULL^2.
_SIN_BETA_FLOOR = 1e-3
_SIN_BETA_TAPER = 0.10
_SIN_BETA_FULL = 0.25
# How far inside the wedge exterior an azimuth must sit to be used at all.
_DOMAIN_MARGIN = 1e-6
# Smallest exterior wedge angle (in units of pi) this model will diffract from.
# A re-entrant corner is a multiple-bounce geometry, and single diffraction has
# less and less to say about it as the corner sharpens.  The right-angle
# re-entrant corner (n = 1/2) is the extreme case: it is a dihedral
# retroreflector, and at backscatter the Keller coefficient has a pole exactly
# in the retroreflection direction, where the true return is the double bounce
# this code does not carry.  Measured worst-case |f| at backscatter:
#
#     n     0.50    0.501   0.51    0.55    0.60    |  1.05..2.0
#     |f|   6667    473     48      9.2     4.2     |  <= 1.0
#
# so edges sharper than n = 0.6 (a 108 degree exterior angle) are dropped
# rather than allowed to dominate the sum with a number that is not physics.
# CAD models also produce degenerate n -> 0 edges wherever two faces end up
# coincident.  :func:`echo1.solver.monostatic_rcs` reports what this removed.
_MIN_WEDGE_N = 0.6
_TWO_PI = 2.0 * np.pi


def _keller(phi: np.ndarray, phi_p: np.ndarray, n: np.ndarray):
    """Total (exact wedge) diffraction coefficients ``f_total, g_total``."""
    mu = np.pi / n
    cos_mu = np.cos(mu)
    term_m = (cos_mu - np.cos((phi - phi_p) / n))
    term_p = (cos_mu - np.cos((phi + phi_p) / n))
    pre = np.sin(mu) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        a = pre / term_m
        b = pre / term_p
    return a - b, a + b


def po_edge_coefficients(phi: np.ndarray, phi_p: np.ndarray, n: np.ndarray):
    """PO part of the edge diffraction, summed over illuminated faces.

    Returns ``(f_po, g_po)``.  Faces are lit according to ``phi'``: the *o*
    face (at ``phi = 0``) for ``0 < phi' < pi``, the *n* face (at
    ``phi = n*pi``) for ``(n-1)*pi < phi' < n*pi``.
    """
    n_pi = n * np.pi
    f = np.zeros(np.broadcast(phi, phi_p, n).shape, dtype=float)
    g = np.zeros_like(f)

    with np.errstate(all="ignore"):
        lit_o = (phi_p > 0.0) & (phi_p < np.pi)
        denom_o = np.cos(phi) + np.cos(phi_p)
        f = np.where(lit_o, np.sin(phi_p) / denom_o, 0.0)
        g = np.where(lit_o, -np.sin(phi) / denom_o, 0.0)

        lit_n = (phi_p > n_pi - np.pi) & (phi_p < n_pi)
        phi_n = n_pi - phi
        phi_pn = n_pi - phi_p
        denom_n = np.cos(phi_n) + np.cos(phi_pn)
        f = f + np.where(lit_n, np.sin(phi_pn) / denom_n, 0.0)
        g = g + np.where(lit_n, -np.sin(phi_n) / denom_n, 0.0)
    return f, g


def _raw_fringe(phi, phi_p, n):
    """``f_total - f_PO``; may be non-finite exactly on a GO boundary."""
    with np.errstate(all="ignore"):
        f_tot, g_tot = _keller(phi, phi_p, n)
        f_po, g_po = po_edge_coefficients(phi, phi_p, n)
        return f_tot - f_po, g_tot - g_po


def _near_singular(phi, phi_p, n) -> np.ndarray:
    """True where any diffraction denominator is close to zero."""
    mu = np.pi / n
    cos_mu = np.cos(mu)
    n_pi = n * np.pi
    d1 = np.abs(cos_mu - np.cos((phi - phi_p) / n))
    d2 = np.abs(cos_mu - np.cos((phi + phi_p) / n))
    d3 = np.where(
        (phi_p > 0.0) & (phi_p < np.pi),
        np.abs(np.cos(phi) + np.cos(phi_p)),
        np.inf,
    )
    d4 = np.where(
        (phi_p > n_pi - np.pi) & (phi_p < n_pi),
        np.abs(np.cos(n_pi - phi) + np.cos(n_pi - phi_p)),
        np.inf,
    )
    return np.minimum(np.minimum(d1, d2), np.minimum(d3, d4)) < _SING_TOL


def fringe_coefficients(phi, phi_p, n):
    """Ufimtsev fringe coefficients ``(f, g)``, finite at the GO boundaries.

    ``phi`` and ``phi'`` are the observation and incidence azimuths in the
    plane perpendicular to the edge, measured from the *o* face; ``n * pi`` is
    the exterior wedge angle.  All arguments broadcast together.
    """
    phi, phi_p, n = np.broadcast_arrays(
        np.asarray(phi, float), np.asarray(phi_p, float), np.asarray(n, float)
    )
    # Coplanar facets are not a wedge: they diffract nothing, exactly.
    flat = np.abs(n - 1.0) < 1e-9

    with np.errstate(all="ignore"):
        f, g = _raw_fringe(phi, phi_p, n)
        bad = _near_singular(phi, phi_p, n) | ~np.isfinite(f) | ~np.isfinite(g)
        bad &= ~flat
    # The fringe coefficient is analytic across a GO boundary even though both
    # of its parts blow up there, so recover it by averaging across the
    # boundary.  A sample point can itself land on another exact degeneracy, so
    # step out by an incommensurate factor until the average is finite.
        delta = _SING_DELTA
        for _ in range(4):
            if not np.any(bad):
                break
            f_lo, g_lo = _raw_fringe(phi - delta, phi_p, n)
            f_hi, g_hi = _raw_fringe(phi + delta, phi_p, n)
            f_av = 0.5 * (f_lo + f_hi)
            g_av = 0.5 * (g_lo + g_hi)
            ok = bad & np.isfinite(f_av) & np.isfinite(g_av)
            f = np.where(ok, f_av, f)
            g = np.where(ok, g_av, g)
            bad = bad & ~ok
            delta *= 3.7
        f = np.where(bad | flat, 0.0, f)
        g = np.where(bad | flat, 0.0, g)
    return f, g


def _azimuths(edges: EdgeSet, i_hat: np.ndarray, s_hat: np.ndarray):
    """Local wedge azimuths and cone angle for every (angle, edge) pair."""
    e = edges.e_hat                                          # (E, 3)
    x = edges.x_hat
    y = edges.y_hat

    from_source = -i_hat                                     # (A, 3)
    # Components perpendicular to each edge.
    d_p = from_source[:, None, :] - np.einsum("ak,ek->ae", from_source, e)[..., None] * e[None]
    d_s = s_hat[:, None, :] - np.einsum("ak,ek->ae", s_hat, e)[..., None] * e[None]

    sin_beta = np.linalg.norm(d_p, axis=2)                   # (A, E)
    phi_p = np.mod(
        np.arctan2(np.einsum("aek,ek->ae", d_p, y), np.einsum("aek,ek->ae", d_p, x)), _TWO_PI
    )
    phi = np.mod(
        np.arctan2(np.einsum("aek,ek->ae", d_s, y), np.einsum("aek,ek->ae", d_s, x)), _TWO_PI
    )
    return phi, phi_p, np.maximum(sin_beta, _SIN_BETA_FLOOR)


def diffracting(edges: EdgeSet, min_wedge_n: float = _MIN_WEDGE_N) -> np.ndarray:
    """(E,) bool: edges whose wedge angle this model can honestly handle.

    Coplanar facets (``n == 1``) are excluded because they diffract nothing,
    and strongly re-entrant corners because single-bounce PTD does not apply
    there -- see :data:`_MIN_WEDGE_N`.
    """
    n = edges.wedge_n
    return (n >= min_wedge_n) & (np.abs(n - 1.0) > 1e-9)


def ptd_amplitude(
    edges: EdgeSet,
    i_hat: np.ndarray,
    s_hat: np.ndarray,
    e_inc: np.ndarray,
    e_rec: np.ndarray,
    k: float,
    active: np.ndarray,
    po_only: bool = False,
    min_wedge_n: float = _MIN_WEDGE_N,
) -> np.ndarray:
    """Per-edge diffraction amplitude ``S``, with ``sigma = |sum S|^2``.

    Parameters mirror :func:`echo1.po.po_amplitude`.  ``active`` is an
    ``(A, E)`` mask of edges the radar can see.  With ``po_only=True`` the
    *PO* edge coefficients are used instead of the fringe ones, which
    reproduces the asymptotic edge behaviour of physical optics itself -- the
    test suite uses that to check this module against :mod:`echo1.po`.
    """
    active = np.asarray(active, bool) & diffracting(edges, min_wedge_n)[None, :]
    phi, phi_p, sin_beta = _azimuths(edges, i_hat, s_hat)
    n = np.broadcast_to(edges.wedge_n[None, :], phi.shape)

    # The wedge coefficients are defined only over the exterior of the wedge,
    # 0 < phi < n*pi.  A direction outside that lies inside the material: the
    # edge is neither lit nor visible from there, and the formulas do not
    # merely lose meaning, they land on poles.  On a real airframe an edge
    # reporting phi = 360 degrees on a wedge spanning 135 degrees produced
    # |f| = 3e4 and swamped everything else.
    span = n * np.pi
    inside = ((phi > _DOMAIN_MARGIN) & (phi < span - _DOMAIN_MARGIN)
              & (phi_p > _DOMAIN_MARGIN) & (phi_p < span - _DOMAIN_MARGIN))
    active = active & inside

    if po_only:
        f, g = po_edge_coefficients(phi, phi_p, n)
        f = np.where(np.isfinite(f), f, 0.0)
        g = np.where(np.isfinite(g), g, 0.0)
    else:
        f, g = fringe_coefficients(phi, phi_p, n)

    e_hat = edges.e_hat                                      # (E, 3)
    # Equivalent edge currents: eta*I_e = -(j/k) (e.E_i) f / sin^2(beta0),
    #                                I_m = -(j/k) (e.eta*H_i) g / sin^2(beta0).
    # The constant is not guessed.  With the f, g normalisation fixed above by
    # pole cancellation, it is the value for which feeding the *PO* edge
    # coefficients through this routine reproduces the exact Gordon integral
    # for a rectangular plate -- magnitude and sign, to machine precision, at
    # every aspect and polarisation (test_ptd.py).  That is what makes the
    # ``f_total - f_PO`` subtraction remove exactly the edge wave that
    # :mod:`echo1.po` already contains, instead of doubling it.
    e_dot_ei = np.einsum("ak,ek->ae", e_inc, e_hat)
    eta_h_inc = np.cross(i_hat, e_inc)                       # (A, 3) = eta * H_i
    e_dot_hi = np.einsum("ak,ek->ae", eta_h_inc, e_hat)

    e_dot_er = np.einsum("ak,ek->ae", e_rec, e_hat)
    s_cross_e = np.cross(s_hat[:, None, :], e_hat[None, :, :])   # (A, E, 3)
    er_dot_sxe = np.einsum("ak,aek->ae", e_rec, s_cross_e)

    # Fade out the end-on caustic (see _SIN_BETA_TAPER).
    ramp = np.clip((sin_beta - _SIN_BETA_TAPER) / (_SIN_BETA_FULL - _SIN_BETA_TAPER),
                   0.0, 1.0)
    taper = ramp * ramp * (3.0 - 2.0 * ramp)
    coeff = (-1.0j / k) * taper / sin_beta**2
    vec = coeff * (-e_dot_ei * e_dot_er * f + e_dot_hi * er_dot_sxe * g)

    # Line integral of exp(j w . r) along each straight edge.
    w = k * (s_hat - i_hat)                                  # (A, 3)
    seg = edges.p1 - edges.p0                                # (E, 3)
    mid = edges.midpoint
    phase = np.exp(1j * np.einsum("ak,ek->ae", w, mid))
    taper = np.sinc(np.einsum("ak,ek->ae", w, seg) / (2.0 * np.pi))
    span = edges.length[None, :] * phase * taper

    amp = (k / (2.0 * np.sqrt(np.pi))) * vec * span
    return np.where(active, amp, 0.0)
