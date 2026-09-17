"""Faceted geometry: triangle meshes and the wedge structure of their edges.

ECHO 1 could only analyse shapes "provided it could be made from flat panels".
This module is that restriction, made explicit: a body is a set of flat
triangular facets, plus the set of edges where those facets meet.

Two things are derived here that the scattering kernels need:

* per-facet quantities (outward normal, area, centroid, vertex winding), used
  by the physical-optics integral;
* per-edge wedge geometry (the local frame and the exterior wedge angle
  ``n * pi``), used by the Ufimtsev edge-wave correction.

Edge frame convention
---------------------
For an edge shared by facets *o* and *n*, we build a right-handed frame
``(x_hat, y_hat, e_hat)`` with

* ``x_hat`` in the *o* facet, perpendicular to the edge, pointing away from
  the edge into the facet;
* ``y_hat`` the outward normal of the *o* facet;
* ``e_hat = x_hat x y_hat`` the edge direction.

Azimuth ``phi`` is measured about ``e_hat`` from ``x_hat`` towards ``y_hat``.
The *o* facet then sits at ``phi = 0``, the exterior region is swept by
increasing ``phi``, and the *n* facet sits at ``phi = n * pi``.  For a cube
edge ``n = 3/2``; for two coplanar facets ``n = 1``; for the free rim of an
open sheet ``n = 2`` (a knife edge).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["Mesh", "EdgeSet", "normalize"]

_EPS = 1e-12


def normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return ``v`` scaled to unit length along ``axis`` (zero vectors stay zero)."""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return np.divide(v, norm, out=np.zeros_like(v), where=norm > _EPS)


@dataclass(frozen=True)
class EdgeSet:
    """Edges of a mesh, with the wedge geometry each one represents.

    All arrays are indexed by edge.  ``p0`` and ``p1`` are ordered so that
    ``p1 - p0`` points along ``e_hat``.
    """

    p0: np.ndarray            # (E, 3) edge start
    p1: np.ndarray            # (E, 3) edge end
    e_hat: np.ndarray         # (E, 3) unit edge direction
    length: np.ndarray        # (E,)   edge length
    x_hat: np.ndarray         # (E, 3) in-facet direction of the o facet
    y_hat: np.ndarray         # (E, 3) outward normal of the o facet
    wedge_n: np.ndarray       # (E,)   exterior wedge angle / pi
    faces: np.ndarray         # (E, 2) adjacent facet indices, -1 if none
    n_adjacent: np.ndarray    # (E,)   1 (knife edge) or 2

    def __len__(self) -> int:
        return int(self.p0.shape[0])

    @property
    def midpoint(self) -> np.ndarray:
        return 0.5 * (self.p0 + self.p1)

    def select(self, mask: np.ndarray) -> "EdgeSet":
        """Return the subset of edges selected by a boolean mask."""
        mask = np.asarray(mask, dtype=bool)
        return EdgeSet(
            p0=self.p0[mask],
            p1=self.p1[mask],
            e_hat=self.e_hat[mask],
            length=self.length[mask],
            x_hat=self.x_hat[mask],
            y_hat=self.y_hat[mask],
            wedge_n=self.wedge_n[mask],
            faces=self.faces[mask],
            n_adjacent=self.n_adjacent[mask],
        )


@dataclass
class Mesh:
    """A body approximated by flat triangular facets.

    Parameters
    ----------
    vertices : (V, 3) array
    faces : (F, 3) int array
        Vertex indices.  Winding is taken as authoritative: the outward normal
        of a facet is ``(v1 - v0) x (v2 - v0)``, normalised.
    two_sided : bool
        ``True`` for an open sheet with no interior (a plate, a fin), where a
        facet scatters from whichever side the radar illuminates.  ``False``
        for a closed body, where only facets whose outward normal faces the
        radar are illuminated.
    name : str
    """

    vertices: np.ndarray
    faces: np.ndarray
    two_sided: bool = False
    name: str = "mesh"
    _edges: Optional[EdgeSet] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.vertices = np.ascontiguousarray(self.vertices, dtype=float)
        self.faces = np.ascontiguousarray(self.faces, dtype=np.int64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (V, 3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must have shape (F, 3); quads must be split")
        if self.faces.size and (self.faces.min() < 0 or self.faces.max() >= len(self.vertices)):
            raise ValueError("face references a vertex index outside the vertex array")
        degenerate = self.face_areas <= _EPS
        if np.any(degenerate):
            raise ValueError(f"{int(degenerate.sum())} facet(s) have zero area")

    # ---------------------------------------------------------------- facets

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def triangles(self) -> np.ndarray:
        """(F, 3, 3) array of facet vertex coordinates, in winding order."""
        return self.vertices[self.faces]

    @property
    def face_normals(self) -> np.ndarray:
        """(F, 3) unit outward normals, from the vertex winding."""
        tri = self.triangles
        return normalize(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]))

    @property
    def face_areas(self) -> np.ndarray:
        tri = self.triangles
        return 0.5 * np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
        )

    @property
    def face_centroids(self) -> np.ndarray:
        return self.triangles.mean(axis=1)

    @property
    def total_area(self) -> float:
        return float(self.face_areas.sum())

    @property
    def volume(self) -> float:
        """Signed volume from the divergence theorem.

        Positive when the facet winding gives outward normals, so the sign is
        a cheap check that a closed body is wound the right way round.
        """
        tri = self.triangles
        return float(np.einsum("fk,fk->f",
                               tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)

    @property
    def is_closed(self) -> bool:
        """True when every edge is shared by exactly two facets."""
        return bool(np.all(self.edges.n_adjacent == 2))

    @property
    def extent(self) -> np.ndarray:
        """(2, 3) axis-aligned bounding box."""
        return np.vstack([self.vertices.min(axis=0), self.vertices.max(axis=0)])

    @property
    def max_dimension(self) -> float:
        lo, hi = self.extent
        return float(np.max(hi - lo))

    # ----------------------------------------------------------------- edges

    @property
    def edges(self) -> EdgeSet:
        """Unique edges with their wedge geometry (computed once, then cached)."""
        if self._edges is None:
            self._edges = self._build_edges()
        return self._edges

    def _build_edges(self) -> EdgeSet:
        f = self.faces
        # Directed edges of every facet, in winding order.
        directed = np.concatenate(
            [f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0
        )
        owner = np.tile(np.arange(self.n_faces), 3)

        key = np.sort(directed, axis=1)
        order = np.lexsort((key[:, 1], key[:, 0]))
        key_sorted = key[order]
        owner_sorted = owner[order]

        # Group identical undirected edges.
        new_group = np.ones(len(key_sorted), dtype=bool)
        if len(key_sorted) > 1:
            new_group[1:] = np.any(key_sorted[1:] != key_sorted[:-1], axis=1)
        group_id = np.cumsum(new_group) - 1
        n_edges = int(group_id[-1]) + 1 if len(group_id) else 0

        counts = np.bincount(group_id, minlength=n_edges)
        if np.any(counts > 2):
            raise ValueError(
                "non-manifold mesh: an edge is shared by more than two facets"
            )

        starts = np.flatnonzero(new_group)
        verts = key_sorted[starts]                       # (E, 2)
        face_a = owner_sorted[starts]                    # (E,)
        face_b = np.full(n_edges, -1, dtype=np.int64)
        has_second = counts == 2
        face_b[has_second] = owner_sorted[starts[has_second] + 1]

        pa = self.vertices[verts[:, 0]]
        pb = self.vertices[verts[:, 1]]
        mid = 0.5 * (pa + pb)
        raw_dir = normalize(pb - pa)

        normals = self.face_normals
        centroids = self.face_centroids

        n_o = normals[face_a]
        # In-facet direction of the o facet, perpendicular to the edge.
        to_centroid = centroids[face_a] - mid
        x_hat = normalize(to_centroid - (to_centroid * raw_dir).sum(1, keepdims=True) * raw_dir)
        e_hat = np.cross(x_hat, n_o)
        e_hat = normalize(e_hat)

        # Order the stored endpoints along e_hat.
        forward = (raw_dir * e_hat).sum(1) >= 0.0
        p0 = np.where(forward[:, None], pa, pb)
        p1 = np.where(forward[:, None], pb, pa)
        length = np.linalg.norm(p1 - p0, axis=1)

        # Exterior wedge angle: azimuth of the n facet in the (x_hat, n_o) frame.
        wedge_n = np.full(n_edges, 2.0)                  # knife edge by default
        idx = np.flatnonzero(has_second)
        if idx.size:
            fb = face_b[idx]
            to_centroid_b = centroids[fb] - mid[idx]
            d = raw_dir[idx]
            t_n = normalize(to_centroid_b - (to_centroid_b * d).sum(1, keepdims=True) * d)
            phi_n = np.arctan2(
                (t_n * n_o[idx]).sum(1), (t_n * x_hat[idx]).sum(1)
            )
            phi_n = np.mod(phi_n, 2.0 * np.pi)
            # Coplanar facets land on phi = pi within rounding; keep them exact so
            # that the diffraction coefficients cancel to zero.
            phi_n = np.where(np.abs(phi_n - np.pi) < 1e-9, np.pi, phi_n)
            wedge_n[idx] = phi_n / np.pi

        return EdgeSet(
            p0=p0,
            p1=p1,
            e_hat=e_hat,
            length=length,
            x_hat=x_hat,
            y_hat=n_o,
            wedge_n=wedge_n,
            faces=np.stack([face_a, face_b], axis=1),
            n_adjacent=np.where(has_second, 2, 1).astype(np.int64),
        )

    # ------------------------------------------------------------ transforms

    def translated(self, offset) -> "Mesh":
        return Mesh(self.vertices + np.asarray(offset, float), self.faces,
                    self.two_sided, self.name)

    def scaled(self, factor: float) -> "Mesh":
        return Mesh(self.vertices * float(factor), self.faces,
                    self.two_sided, self.name)

    def rotated(self, axis, angle_deg: float) -> "Mesh":
        """Rotate about ``axis`` through the origin (Rodrigues)."""
        k = normalize(np.asarray(axis, float))
        t = np.radians(angle_deg)
        kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        r = np.eye(3) + np.sin(t) * kx + (1 - np.cos(t)) * (kx @ kx)
        return Mesh(self.vertices @ r.T, self.faces, self.two_sided, self.name)

    def flipped(self) -> "Mesh":
        """Reverse facet winding (and therefore the outward normals)."""
        return Mesh(self.vertices, self.faces[:, [0, 2, 1]], self.two_sided, self.name)

    def __add__(self, other: "Mesh") -> "Mesh":
        """Concatenate two meshes into one body."""
        if not isinstance(other, Mesh):
            return NotImplemented
        return Mesh(
            np.vstack([self.vertices, other.vertices]),
            np.vstack([self.faces, other.faces + len(self.vertices)]),
            self.two_sided or other.two_sided,
            f"{self.name}+{other.name}",
        )

    def __repr__(self) -> str:
        return (
            f"Mesh(name={self.name!r}, vertices={len(self.vertices)}, "
            f"facets={self.n_faces}, edges={len(self.edges)}, "
            f"area={self.total_area:.4g} m^2, two_sided={self.two_sided})"
        )
