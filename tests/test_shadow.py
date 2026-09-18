"""Occlusion: the grid-accelerated path must agree exactly with brute force."""

import numpy as np
import pytest

from echo1 import shadow, shapes
from echo1.geometry import Mesh
from echo1.solver import look_vectors, monostatic_rcs


def _directions(n=24, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


@pytest.fixture
def cluttered():
    """A body that genuinely hides parts of itself from some directions."""
    body = shapes.box(3.0, 1.0, 0.5)
    body = body + shapes.box(0.8, 0.8, 0.8).translated([0.0, 2.0, 0.0])
    body = body + shapes.box(0.8, 0.8, 0.8).translated([0.0, -2.0, 0.0])
    body = body + shapes.sphere(0.6, subdivisions=2).translated([-1.4, 0.0, 0.9])
    return body


@pytest.mark.parametrize("target", ["facets", "edges"])
def test_grid_matches_brute_force(cluttered, monkeypatch, target):
    r = _directions()
    fn = shadow.facet_illumination if target == "facets" else shadow.edge_illumination

    monkeypatch.setattr(shadow, "_BRUTE_FORCE_LIMIT", 10 ** 9)
    brute = fn(cluttered, r, occlusion=True)
    monkeypatch.setattr(shadow, "_BRUTE_FORCE_LIMIT", 0)
    gridded = fn(cluttered, r, occlusion=True)

    assert brute.shape == gridded.shape
    assert np.array_equal(brute, gridded)


def test_grid_matches_brute_force_on_rcs(cluttered, monkeypatch):
    az = np.arange(0.0, 360.0, 17.0)
    monkeypatch.setattr(shadow, "_BRUTE_FORCE_LIMIT", 10 ** 9)
    brute = monostatic_rcs(cluttered, 10e9, az, 15.0, pols=("VV",))
    monkeypatch.setattr(shadow, "_BRUTE_FORCE_LIMIT", 0)
    gridded = monostatic_rcs(cluttered, 10e9, az, 15.0, pols=("VV",))
    assert np.array_equal(brute.sigma["VV"], gridded.sigma["VV"])


def test_sprawling_facets_are_still_tested(monkeypatch):
    """One facet far larger than the rest must not fall out of the grid."""
    small = shapes.sphere(0.2, subdivisions=2).translated([0.0, 0.0, 1.0])
    wall = shapes.plate(8.0, 8.0).translated([0.0, 0.0, 2.0])
    body = small + wall
    r, _, _ = look_vectors([0.0], [90.0])          # straight up, through the wall
    monkeypatch.setattr(shadow, "_BRUTE_FORCE_LIMIT", 0)
    lit = shadow.facet_illumination(body, r, occlusion=True)
    # The sphere sits under the big plate and must be entirely hidden.
    assert not lit[0, : small.n_faces].any()


def test_occlusion_still_optional(cluttered):
    r = _directions(8, seed=3)
    on = shadow.facet_illumination(cluttered, r, occlusion=True)
    off = shadow.facet_illumination(cluttered, r, occlusion=False)
    assert on.sum() < off.sum()


def test_grid_handles_a_degenerate_projection(monkeypatch):
    """Looking straight down a flat plate gives a zero-thickness projection."""
    body = shapes.plate(2.0, 2.0)
    monkeypatch.setattr(shadow, "_BRUTE_FORCE_LIMIT", 0)
    r, _, _ = look_vectors([0.0], [0.0])           # edge-on
    lit = shadow.facet_illumination(body, r, occlusion=True)
    assert lit.shape == (1, body.n_faces)
