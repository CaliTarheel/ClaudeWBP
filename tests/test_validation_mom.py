"""Regression against the independent method-of-moments reference.

This is the check that the edge-wave term is not merely self-consistent but
*right*: a direct numerical solution of the integral equation, sharing no code
and no approximation with echo1, says how far off physical optics alone is and
how much of that gap PTD closes.
"""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "validation"))

from mom_strip import compare  # noqa: E402

db = lambda v: 10 * np.log10(np.maximum(v, 1e-30))


@pytest.fixture(scope="module")
def bands():
    theta, mom, po, pt = compare(width_lambda=6.0, length_lambda=60.0,
                                 theta_deg=np.arange(20.0, 86.0, 0.5))
    out = {}
    for lo, hi in ((20, 45), (45, 70), (70, 86)):
        m = (theta >= lo) & (theta < hi)
        out[(lo, hi)] = (mom[m].mean(), po[m].mean(), pt[m].mean())
    return out


def test_physical_optics_alone_is_badly_low_off_specular(bands):
    """Establishes that there is a real gap for PTD to close."""
    mom, po, _ = bands[(70, 86)]
    assert db(po) - db(mom) < -10.0


@pytest.mark.parametrize("band,tol", [((20, 45), 2.0), ((45, 70), 4.0), ((70, 86), 6.0)])
def test_ptd_agrees_with_moments(bands, band, tol):
    """Accuracy degrades towards grazing, as first-order PTD is expected to."""
    mom, _, pt = bands[band]
    assert abs(db(pt) - db(mom)) < tol


@pytest.mark.parametrize("band", [(20, 45), (45, 70), (70, 86)])
def test_ptd_always_improves_on_physical_optics(bands, band):
    mom, po, pt = bands[band]
    assert abs(db(pt) - db(mom)) < abs(db(po) - db(mom))
