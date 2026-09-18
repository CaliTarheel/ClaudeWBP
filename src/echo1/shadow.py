"""Which facets and edges the radar can actually see.

Two effects are separated:

* *orientation* -- a facet of a closed body carries PO current only if its
  outward normal has a component towards the radar.  An open, two-sided sheet
  is lit from whichever side the radar is on.
* *occlusion* -- a facet or edge that is oriented towards the radar may still
  be hidden behind another part of the body.  This is resolved by casting a
  ray from the facet centroid (or edge midpoint) towards the radar and
  checking it against the facets.

Occlusion is the expensive part.  Tested naively it is O(facets^2) per look
direction, which is fine for a test shape and hopeless for a CAD model.  For
anything but the smallest meshes the facets are bucketed into a uniform grid
on the plane perpendicular to the look direction, so each ray is tested only
against facets that project near it.  The two paths are required to agree
exactly (``tests/test_shadow.py``).

For a convex body, orientation alone is exact and the ray trace can be
switched off entirely.
"""

from __future__ import annotations

import numpy as np

from .geometry import Mesh

__all__ = ["facet_illumination", "edge_illumination"]

_RAY_EPS = 1e-9
# Below this many facets the direct test is faster than building a grid.
_BRUTE_FORCE_LIMIT = 512
# A facet whose projection spans more cells than this is tested against every
# ray rather than being written into all of them.
_MAX_CELLS_PER_FACET = 256
# Candidate pairs evaluated at a time, so peak memory does not track the model.
_PAIR_CHUNK = 400_000
# Cell sizes tried, as divisors of the projected model span.
_CELL_LADDER = (512, 362, 256, 181, 128, 91, 64, 32, 16)
# Refuse a cell size that would expand into more (cell, facet) entries than this.
_EXPANSION_LIMIT = 20_000_000


def _orientation(mesh: Mesh, r_hat: np.ndarray) -> np.ndarray:
    """(A, F) bool: facets oriented towards the radar."""
    cos = np.einsum("ak,fk->af", r_hat, mesh.face_normals)
    if mesh.two_sided:
        return np.abs(cos) > 0.0
    return cos > 0.0


def _basis(r_hat: np.ndarray):
    """Two unit vectors spanning the plane perpendicular to ``r_hat``."""
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(r_hat @ helper)) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    u = np.cross(r_hat, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(r_hat, u)


def _facet_terms(r_hat, tri):
    """Per-facet parts of the Moller-Trumbore test, which do not depend on the ray."""
    v0 = tri[:, 0]
    e1 = tri[:, 1] - v0
    e2 = tri[:, 2] - v0
    pv = np.cross(np.broadcast_to(r_hat, e2.shape), e2)
    det = np.einsum("fk,fk->f", e1, pv)
    usable = np.abs(det) > 1e-14
    inv_det = np.where(usable, 1.0 / np.where(usable, det, 1.0), 0.0)
    return v0, e1, e2, pv, inv_det, usable


def _hits_for_pairs(origins, skip_ids, r_hat, terms, scale, q_idx, f_idx):
    """Moller-Trumbore over an explicit list of (ray, facet) candidate pairs.

    Returns a boolean array over rays: True where some candidate facet lies in
    front of the ray's origin.  Evaluated in chunks so peak memory is set by
    ``_PAIR_CHUNK`` rather than by the size of the model.
    """
    v0, e1, e2, pv, inv_det, usable = terms
    out = np.zeros(len(origins), dtype=bool)
    total = len(q_idx)
    for start in range(0, total, _PAIR_CHUNK):
        sl = slice(start, min(start + _PAIR_CHUNK, total))
        q, f = q_idx[sl], f_idx[sl]

        inv = inv_det[f]
        tv = origins[q] - v0[f]
        u = np.einsum("pk,pk->p", tv, pv[f]) * inv
        qv = np.cross(tv, e1[f])
        v = (qv @ r_hat) * inv
        t = np.einsum("pk,pk->p", qv, e2[f]) * inv

        hit = (
            usable[f]
            & (u >= -1e-9)
            & (v >= -1e-9)
            & (u + v <= 1.0 + 1e-9)
            & (t > _RAY_EPS * scale)
            & ~(skip_ids[q] == f[:, None]).any(axis=1)
        )
        out[q[hit]] = True
    return out


def _grid_pairs(origins_u, origins_v, lo_u, hi_u, lo_v, hi_v, base_u, base_v,
                span, cell, n_rays):
    """Bucket facets into a uniform grid and return the candidate pairs.

    Returns ``(q_idx, f_idx, cost)``, or ``None`` if this cell size would
    expand into an unreasonable number of (cell, facet) entries.
    """
    nu = int(np.floor(span / cell)) + 1
    n_cells = nu * nu
    cu0 = np.floor((lo_u - base_u) / cell).astype(np.int64)
    cu1 = np.floor((hi_u - base_u) / cell).astype(np.int64)
    cv0 = np.floor((lo_v - base_v) / cell).astype(np.int64)
    cv1 = np.floor((hi_v - base_v) / cell).astype(np.int64)
    spans = (cu1 - cu0 + 1) * (cv1 - cv0 + 1)

    sprawling = spans > _MAX_CELLS_PER_FACET
    binned = np.flatnonzero(~sprawling)
    counts = spans[binned]
    if int(counts.sum()) > _EXPANSION_LIMIT:
        return None

    owner = np.repeat(binned, counts)
    within = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    width = np.repeat((cu1 - cu0 + 1)[binned], counts)
    cell_id = np.clip(
        (cv0[owner] + within // width) * nu + (cu0[owner] + within % width),
        0, n_cells - 1)

    order = np.argsort(cell_id, kind="stable")
    cell_sorted = cell_id[order]
    facet_sorted = owner[order]
    grid = np.arange(n_cells)
    starts = np.searchsorted(cell_sorted, grid, side="left")
    ends = np.searchsorted(cell_sorted, grid, side="right")

    ray_cell = np.clip(
        np.floor((origins_v - base_v) / cell).astype(np.int64) * nu
        + np.floor((origins_u - base_u) / cell).astype(np.int64),
        0, n_cells - 1)
    n_cand = ends[ray_cell] - starts[ray_cell]

    global_f = np.flatnonzero(sprawling)
    cost = int(n_cand.sum()) + int(global_f.size) * n_rays
    return ray_cell, starts, facet_sorted, n_cand, global_f, cost


def _expand(ray_cell, starts, facet_sorted, n_cand, global_f, n_rays):
    q_idx = np.repeat(np.arange(n_rays), n_cand)
    offset = np.arange(n_cand.sum()) - np.repeat(np.cumsum(n_cand) - n_cand, n_cand)
    f_idx = facet_sorted[np.repeat(starts[ray_cell], n_cand) + offset]
    if global_f.size:
        q_idx = np.concatenate([q_idx, np.repeat(np.arange(n_rays), global_f.size)])
        f_idx = np.concatenate([f_idx, np.tile(global_f, n_rays)])
    return q_idx, f_idx


def _ray_hits(origins, skip_ids, r_hat, tri, scale):
    """(Q,) bool: does the ray from each origin towards ``r_hat`` hit a facet?

    ``skip_ids`` is a (Q, k) array of facet indices each ray must ignore (its
    own facet, and for an edge the facets that form it), padded with -1.
    """
    n_faces = len(tri)
    n_rays = len(origins)
    if n_rays == 0:
        return np.zeros(0, dtype=bool)

    terms = _facet_terms(r_hat, tri)

    if n_faces <= _BRUTE_FORCE_LIMIT:
        q_idx = np.repeat(np.arange(n_rays), n_faces)
        f_idx = np.tile(np.arange(n_faces), n_rays)
        return _hits_for_pairs(origins, skip_ids, r_hat, terms, scale, q_idx, f_idx)

    u_hat, v_hat = _basis(r_hat)
    tri_u = tri @ u_hat
    tri_v = tri @ v_hat
    lo_u, hi_u = tri_u.min(axis=1), tri_u.max(axis=1)
    lo_v, hi_v = tri_v.min(axis=1), tri_v.max(axis=1)
    origins_u = origins @ u_hat
    origins_v = origins @ v_hat

    base_u = float(min(lo_u.min(), origins_u.min()))
    base_v = float(min(lo_v.min(), origins_v.min()))
    span = max(float(max(hi_u.max(), origins_u.max()) - base_u),
               float(max(hi_v.max(), origins_v.max()) - base_v), scale * 1e-9)

    # The cost is U-shaped in cell size: fine cells bury the large facets in
    # the every-ray fallback, coarse cells put every facet in every ray's
    # bucket.  A CAD model spans three decades of facet size, so the optimum
    # cannot be guessed from the typical facet -- search for it instead.
    best = None
    for divisor in _CELL_LADDER:
        got = _grid_pairs(origins_u, origins_v, lo_u, hi_u, lo_v, hi_v,
                          base_u, base_v, span, span / divisor, n_rays)
        if got is not None and (best is None or got[-1] < best[-1]):
            best = got
    if best is None:
        q_idx = np.repeat(np.arange(n_rays), n_faces)
        f_idx = np.tile(np.arange(n_faces), n_rays)
    else:
        q_idx, f_idx = _expand(*best[:-1], n_rays)

    return _hits_for_pairs(origins, skip_ids, r_hat, terms, scale, q_idx, f_idx)


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
    skip = np.arange(mesh.n_faces)[:, None]
    for a in range(len(r_hat)):
        idx = np.flatnonzero(lit[a])
        if idx.size == 0:
            continue
        blocked = _ray_hits(centroids[idx], skip[idx], r_hat[a], tri, scale)
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
    skip = np.stack([face_a, np.where(has_b, face_b, -1)], axis=1)

    for a in range(len(r_hat)):
        idx = np.flatnonzero(active[a])
        if idx.size == 0:
            continue
        blocked = _ray_hits(mid[idx], skip[idx], r_hat[a], tri, scale)
        active[a, idx[blocked]] = False
    return active
