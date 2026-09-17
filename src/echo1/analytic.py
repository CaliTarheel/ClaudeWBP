"""Closed-form radar cross sections, for validation and for sanity checks.

Every formula here is a textbook high-frequency result.  They are the yardstick
the test suite measures :mod:`echo1` against, and they are exposed publicly so
that a user can check a model before trusting a sweep of it.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "wavelength",
    "flat_plate_broadside",
    "flat_plate_po",
    "sphere_optical",
    "circular_disc_broadside",
    "dihedral_peak",
    "trihedral_peak",
    "cylinder_broadside",
    "to_dbsm",
    "from_dbsm",
]

C0 = 299_792_458.0  # m/s


def wavelength(freq_hz: float) -> float:
    return C0 / float(freq_hz)


def to_dbsm(sigma) -> np.ndarray:
    """m^2 -> dBsm, with a floor so that exact nulls stay plottable."""
    return 10.0 * np.log10(np.maximum(np.asarray(sigma, float), 1e-30))


def from_dbsm(dbsm) -> np.ndarray:
    return 10.0 ** (np.asarray(dbsm, float) / 10.0)


def flat_plate_broadside(area: float, lam: float) -> float:
    """``4 pi A^2 / lambda^2`` -- a flat plate seen along its normal."""
    return 4.0 * np.pi * area ** 2 / lam ** 2


def flat_plate_po(a: float, b: float, theta_deg, lam: float, plane: str = "E"):
    """Physical-optics RCS of an ``a`` by ``b`` rectangular plate.

    ``theta`` is measured from the plate normal, in the principal plane
    containing the side ``a``.  Reduces to :func:`flat_plate_broadside` at
    ``theta = 0``.
    """
    th = np.radians(np.asarray(theta_deg, float))
    k = 2.0 * np.pi / lam
    u = k * a * np.sin(th)
    return (
        4.0 * np.pi * (a * b) ** 2 / lam ** 2
        * np.cos(th) ** 2
        * np.sinc(u / np.pi) ** 2
    )


def sphere_optical(radius: float) -> float:
    """``pi a^2`` -- the optical-region RCS of a sphere."""
    return np.pi * radius ** 2


def circular_disc_broadside(radius: float, lam: float) -> float:
    return 4.0 * np.pi ** 3 * radius ** 4 / lam ** 2


def dihedral_peak(a: float, b: float, lam: float) -> float:
    """Peak RCS of a right-angle dihedral (a double-bounce mechanism)."""
    return 8.0 * np.pi * a ** 2 * b ** 2 / lam ** 2


def trihedral_peak(a: float, lam: float, kind: str = "triangular") -> float:
    """Peak RCS of a right-angle trihedral corner reflector."""
    if kind == "triangular":
        return 4.0 * np.pi * a ** 4 / (3.0 * lam ** 2)
    if kind == "square":
        return 12.0 * np.pi * a ** 4 / lam ** 2
    raise ValueError("kind must be 'triangular' or 'square'")


def cylinder_broadside(radius: float, length: float, lam: float) -> float:
    """Broadside RCS of a circular cylinder."""
    return 2.0 * np.pi * radius * length ** 2 / lam
