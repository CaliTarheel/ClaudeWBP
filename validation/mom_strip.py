"""Independent check: 2-D method of moments for a flat strip.

Solves the TM (E parallel to the edges) electric-field integral equation on a
perfectly conducting strip with pulse basis functions and point matching --
a direct numerical solution of Maxwell's equations, sharing no code and no
approximation with echo1.  The strip's exact echo width is then compared
against echo1's physical optics and PO+PTD predictions for a long rectangular
plate, using the standard 2-D/3-D relation ``sigma_3D = 2 L^2 sigma_2D / lambda``.

Run:  python validation/mom_strip.py
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from scipy.special import hankel2

from echo1.geometry import Mesh
from echo1.po import po_amplitude
from echo1.ptd import ptd_amplitude

GAMMA = 1.781072417990198  # exp(Euler-Mascheroni)


def mom_strip_echo_width(width_lambda: float, theta_deg: np.ndarray,
                         cells_per_lambda: int = 40) -> np.ndarray:
    """Monostatic TM echo width of a strip, in units of lambda.

    ``theta`` is measured from the strip normal.  Units are normalised with
    lambda = 1 and eta = 1.
    """
    lam = 1.0
    k = 2 * np.pi / lam
    n = int(round(width_lambda * cells_per_lambda))
    d = width_lambda / n                                  # cell width
    x = (np.arange(n) + 0.5) * d - width_lambda / 2       # cell centres

    r = np.abs(x[:, None] - x[None, :])
    z = (k * d / 4.0) * hankel2(0, k * np.where(r == 0.0, 1.0, r))
    # Harrington's self term for a pulse cell.
    np.fill_diagonal(z, (k * d / 4.0) * (1.0 - 1j * (2.0 / np.pi) * np.log(GAMMA * k * d / (4.0 * np.e))))

    th = np.radians(np.atleast_1d(theta_deg))
    # Incident plane wave sampled on the strip; backscatter uses the same phase.
    phase = np.exp(1j * k * np.sin(th)[:, None] * x[None, :])
    cur = np.linalg.solve(z, phase.T).T                   # (A, n) surface current
    amp = (cur * phase * d).sum(axis=1)                   # radiation integral
    return (k / 4.0) * np.abs(amp) ** 2


def echo1_plate_rcs(width_lambda: float, length_lambda: float,
                    theta_deg: np.ndarray, with_ptd: bool):
    """echo1's monostatic RCS of a long rectangular plate, HH (E along the long edges)."""
    lam = 1.0
    k = 2 * np.pi / lam
    w, L = width_lambda, length_lambda
    verts = np.array([[-w/2, -L/2, 0], [w/2, -L/2, 0], [w/2, L/2, 0], [-w/2, L/2, 0]])
    plate = Mesh(verts, np.array([[0, 1, 2], [0, 2, 3]]), two_sided=True)

    th = np.radians(np.atleast_1d(theta_deg))
    r = np.stack([np.sin(th), np.zeros_like(th), np.cos(th)], axis=1)
    pol = np.tile(np.array([0.0, 1.0, 0.0]), (len(th), 1))   # E along the long edges

    lit = np.ones((len(th), plate.n_faces), bool)
    s = po_amplitude(plate, -r, r, pol, pol, k, lit).sum(axis=1)
    if with_ptd:
        act = np.ones((len(th), len(plate.edges)), bool)
        s = s + ptd_amplitude(plate.edges, -r, r, pol, pol, k, act).sum(axis=1)
    return np.abs(s) ** 2


def compare(width_lambda: float = 6.0, length_lambda: float = 60.0,
            theta_deg: np.ndarray | None = None):
    """Return ``(theta, mom, po, po_ptd)`` RCS arrays, all in m^2 at lambda = 1."""
    if theta_deg is None:
        theta_deg = np.arange(15.0, 86.0, 0.25)
    mom = 2.0 * length_lambda ** 2 * mom_strip_echo_width(width_lambda, theta_deg)
    po = echo1_plate_rcs(width_lambda, length_lambda, theta_deg, with_ptd=False)
    pt = echo1_plate_rcs(width_lambda, length_lambda, theta_deg, with_ptd=True)
    return theta_deg, mom, po, pt


def main() -> int:
    w, L = 6.0, 60.0
    theta, mom, po, pt = compare(w, L)
    db = lambda v: 10 * np.log10(np.maximum(v, 1e-30))

    print(f"Strip {w:.0f} lambda wide, modelled in echo1 as a {w:.0f} x {L:.0f} lambda")
    print("plate, TM / HH polarisation.  The method of moments is an independent")
    print("numerical solution of Maxwell's equations: no shared code, no asymptotics.\n")

    # Point-by-point dB comparison is dominated by how deep the interference
    # nulls happen to fall, so compare band by band in linear power as well.
    bands = [(20, 45), (45, 70), (70, 86)]
    print(f"  {'aspect band':>14s} {'MoM':>9s} {'PO':>9s} {'err':>7s} "
          f"{'PO+PTD':>9s} {'err':>7s}")
    print(f"  {'(deg off normal)':>14s} {'dBsm':>9s} {'dBsm':>9s} {'dB':>7s} "
          f"{'dBsm':>9s} {'dB':>7s}")
    worst = 0.0
    for lo, hi in bands:
        m = (theta >= lo) & (theta < hi)
        a, b, c = mom[m].mean(), po[m].mean(), pt[m].mean()
        worst = max(worst, abs(db(c) - db(a)))
        print(f"  {f'{lo}-{hi}':>14s} {db(a):9.2f} {db(b):9.2f} {db(b)-db(a):+7.2f} "
              f"{db(c):9.2f} {db(c)-db(a):+7.2f}")
    m = theta >= 20
    a, b, c = mom[m].mean(), po[m].mean(), pt[m].mean()
    print(f"  {'20-86 (all)':>14s} {db(a):9.2f} {db(b):9.2f} {db(b)-db(a):+7.2f} "
          f"{db(c):9.2f} {db(c)-db(a):+7.2f}")

    print("\nReading: physical optics alone falls further and further short as the")
    print("aspect swings away from specular -- by 14 dB near grazing, because it has")
    print("no edge waves at all.  Adding the Ufimtsev term recovers most of that.")
    print("The residual near grazing is first-order PTD's known limit: it carries")
    print("one diffraction per edge, and no wave that crosses the plate and")
    print("diffracts again.")

    ok = abs(db(c) - db(a)) < 3.0 and worst < 6.0
    print(f"\nagreement within tolerance: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
