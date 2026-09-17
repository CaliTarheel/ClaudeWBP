"""Which facets and edges the radar can actually see.

Two effects are separated:

* *orientation* -- a facet of a closed body carries PO current only if its
  outward normal has a component towards the radar.  An open, two-sided sheet
  is lit from whichever side the radar is on.
* *occlusion* -- a facet or edge that is oriented towards the radar may still
  be hidden behind another part of the body.  This is resolved by casting a
  ray from the facet centroid (or edge midpoint) towards the radar and
  checking it against every facet.

Occlusion is the expensive part and can be switched off for convex bodies,
where orientation alone is exact.
"""

from __future__ import annotations

import numpy as np

from .geometry import Mesh

__all__ = ["facet_illumination", "edge_illumination"]

_RAY_EPS = 1e-9


def _orientation(mesh: Mesh, r_hat: np.ndarray) -> np.ndarray:
    """(A, F) bool: facets oriented towards the radar."""
    cos = np.einsum("ak,fk->af", r_hat, mesh.face_normals)
    if mesh.two_sided:
        return np.abs(cos) > 0.0
    return cos > 0.0


def _ray_hits(
    origins: np.ndarray,
    r_hat: np.ndarray,
    tri: np.ndarray,
    skip: np.ndarray,
    scale: float,
) -> np.ndarray:
    """(Q,) bool: does the ray from each origin towards ``r_hat`` hit a facet?

    Moller-Trumbore, vectorised over (query, facet).  ``skip`` is a (Q, F)
    mask of facets each query must ignore (its own, and any it touches).
    """
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    e1 = v1 - v0
    e2 = v2 - v0
    pv = np.cross(r_hat, e2)                              # (F, 3)
    det = np.einsum("fk,fk->f", e1, pv)                   # (F,)
    parallel = np.abs(det) < 1e-14
    inv_det = np.where(parallel, 0.0, 1.0 / np.where(parallel, 1.0, det))

    tv = origins[:, None, :] - v0[None, :, :]             # (Q, F, 3)
    u = np.einsum("qfk,fk->qf", tv, pv) * inv_det[None, :]
    qv = np.cross(tv, e1[None, :, :])                     # (Q, F, 3)
    v = np.einsum("qfk,k->qf", qv, r_hat) * inv_det[None, :]
    t = np.einsum("qfk,fk->qf", qv, e2) * inv_det[None, :]

    hit = (
        (~parallel)[None, :]
        & (u >= -1e-9)
        & (v >= -1e-9)
        & (u + v <= 1.0 + 1e-9)
        & (t > _RAY_EPS * scale)
        & ~skip
    )
    return hit.any(axis=1)


def facet_illumination(mesh: Mesh, r_hat: np.ndarray, occlusion: bool = True) -> np.ndarray:
    """(A, F) bool mask of facets that scatter, for each look direction.

    ``r_hat`` points from the target towards the radar.
    """
    r_hat = np.atleast_2d(np.asarray(r_hat, float))
    lit = _orientation(mesh, r_hat)
    if not occlusion or mesh.n_faces < 2:
        return lit

    tri = mesh.triangles
    centroids = mesh.face_centroids
    scale = max(mesh.max_dimension, 1e-12)
    # A facet never occludes itself.
    skip = np.eye(mesh.n_faces, dtype=bool)
    for a in range(len(r_hat)):
        idx = np.flatnonzero(lit[a])
        if idx.size == 0:
            continue
        blocked = _ray_hits(centroids[idx], r_hat[a], tri, skip[idx], scale)
        lit[a, idx[blocked]] = False
    return lit


def edge_illumination(mesh: Mesh, r_hat: np.ndarray, occlusion: bool = True) -> np.ndarray:
    """(A, E) bool mask of edges that diffract, for each look direction.

    An edge radiates only if at least one of the facets forming it is oriented
    towards the radar, and the edge itself is not hidden behind the body.
    """
    r_hat = np.atleast_2d(np.asarray(r_hat, float))
    edges = mesh.edges
    oriented = _orientation(mesh, r_hat)

    face_a = edges.faces[:, 0]
    face_b = edges.faces[:, 1]
    active = oriented[:, face_a].copy()
    has_b = face_b >= 0
    if np.any(has_b):
        active[:, has_b] |= oriented[:, face_b[has_b]]

    if not occlusion or mesh.n_faces < 2:
        return active

    tri = mesh.triangles
    mid = edges.midpoint
    scale = max(mesh.max_dimension, 1e-12)
    # An edge is not occluded by the facets that form it.
    skip = np.zeros((len(edges), mesh.n_faces), dtype=bool)
    skip[np.arange(len(edges)), face_a] = True
    skip[np.flatnonzero(has_b), face_b[has_b]] = True

    for a in range(len(r_hat)):
        idx = np.flatnonzero(active[a])
        if idx.size == 0:
            continue
        blocked = _ray_hits(mid[idx], r_hat[a], tri, skip[idx], scale)
        active[a, idx[blocked]] = False
    return active
