import numpy as np
import pytest

from echo1.geometry import Mesh

LAMBDA = 0.03
FREQ = 299_792_458.0 / LAMBDA
K = 2.0 * np.pi / LAMBDA


@pytest.fixture
def unit_cube():
    v = np.array([
        [-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
        [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5],
    ], float)
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    f = [t for a, b, c, d in quads for t in ([a, b, c], [a, c, d])]
    return Mesh(v, np.array(f), name="unit-cube")


def rect_plate(a, b):
    v = np.array([[-a/2, -b/2, 0], [a/2, -b/2, 0], [a/2, b/2, 0], [-a/2, b/2, 0]], float)
    return Mesh(v, np.array([[0, 1, 2], [0, 2, 3]]), two_sided=True, name="plate")


def directions(n, seed=0, normal=np.array([0., 0., 1.]), lo=0.2, hi=0.95):
    """Random unit look directions, kept away from grazing and from broadside."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n * 6, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    c = np.abs(d @ normal)
    return d[(c > lo) & (c < hi)][:n]


def pol_frame(d):
    h = np.cross(np.array([0., 0., 1.]), d)
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    return h, np.cross(d, h)
