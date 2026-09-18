"""Cut a body on its centreline, keep one half, and mirror it.

A shape that is meant to be symmetric but is not pays for it in RCS: the two
sides no longer cancel or reinforce the way the design intended, and mirror
aspects can differ by tens of dB.  This clips the mesh on a plane, keeps one
side, mirrors it, and welds the seam so the result is a closed body that is
symmetric to machine precision.

    python tools/symmetrize.py model.obj --keep -y --out model-sym.obj
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from echo1.geometry import Mesh                                      # noqa: E402
from echo1.meshio import load_mesh, save_obj                         # noqa: E402


def clip_half(mesh: Mesh, axis: int = 1, keep_negative: bool = True,
              tol: float = 1e-12) -> Mesh:
    """Keep the part of ``mesh`` on one side of the plane ``axis = 0``.

    Triangles crossing the plane are cut, and the cut points are shared
    between neighbouring triangles so the result stays watertight along the
    seam.
    """
    # Snap vertices that already sit essentially on the plane exactly onto it,
    # so that a kept vertex and its mirror image coincide and the seam welds.
    snap = float(np.ptp(mesh.vertices, axis=0).max()) * 1e-9
    source = mesh.vertices.copy()
    source[np.abs(source[:, axis]) <= snap, axis] = 0.0

    verts = [np.asarray(v, float) for v in source]
    sign = 1.0 if keep_negative else -1.0
    dist = source[:, axis] * sign                   # keep where dist <= 0
    cache: dict = {}

    span = float(np.ptp(source, axis=0).max())

    def cut_point(a: int, b: int) -> int:
        key = (min(a, b), max(a, b))
        if key in cache:
            return cache[key]
        da, db = dist[key[0]], dist[key[1]]
        t = da / (da - db)
        edge = verts[key[1]] - verts[key[0]]
        # If the crossing lands on an endpoint, reuse it.  Manufacturing a
        # point a nanometre away instead makes a hair-thin sliver, which the
        # area filter then deletes -- tearing a hole whose edges are as long
        # as the original triangle.
        if t * np.linalg.norm(edge) < span * 1e-9:
            cache[key] = key[0]
        elif (1.0 - t) * np.linalg.norm(edge) < span * 1e-9:
            cache[key] = key[1]
        else:
            p = verts[key[0]] + t * edge
            p[axis] = 0.0                            # snap exactly onto the plane
            cache[key] = len(verts)
            verts.append(p)
        return cache[key]

    faces = []
    for tri in mesh.faces:
        d = dist[tri]
        if np.all(d > tol):
            continue
        if np.all(d <= tol):
            faces.append(list(tri))
            continue
        poly = []
        for i in range(3):
            j = (i + 1) % 3
            a, b = int(tri[i]), int(tri[j])
            if d[i] <= tol:
                poly.append(a)
            # Only a strict crossing makes a new point.  A vertex lying on the
            # plane is already in the polygon, and manufacturing a coincident
            # copy of it would stop neighbouring triangles sharing that vertex
            # and tear a hole along the cut.
            if d[i] * d[j] < 0.0:
                poly.append(cut_point(a, b))
        for i in range(1, len(poly) - 1):
            faces.append([poly[0], poly[i], poly[i + 1]])

    verts = np.array(verts)
    faces = np.array(faces, dtype=np.int64)
    # Drop slivers the cut may have produced, scaled to the model.
    tri = verts[faces]
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    span = float(np.ptp(source, axis=0).max())
    faces = faces[area > max(span ** 2 * 1e-12, 1e-12)]
    used = np.unique(faces)
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return Mesh(verts[used], remap[faces], mesh.two_sided, mesh.name)


def mirror(mesh: Mesh, axis: int = 1) -> Mesh:
    """Reflect through ``axis = 0``, reversing winding so normals stay outward."""
    scale = np.ones(3)
    scale[axis] = -1.0
    return Mesh(mesh.vertices * scale, mesh.faces[:, [0, 2, 1]],
                mesh.two_sided, mesh.name)


def _weld_seam(verts: np.ndarray, faces: np.ndarray, axis: int, tol: float):
    """Fuse only the vertices sitting on the cut plane.

    Welding everything would also fuse separate solids wherever they touch,
    which makes the mesh non-manifold.  The seam is the only place the two
    halves need to be joined.
    """
    on_seam = np.abs(verts[:, axis]) <= tol
    remap = np.arange(len(verts))
    table: dict = {}
    for i in np.flatnonzero(on_seam):
        key = tuple(np.round(verts[i] / tol).astype(np.int64))
        remap[i] = table.setdefault(key, i)
    faces = remap[faces]
    keep = ((faces[:, 0] != faces[:, 1])
            & (faces[:, 1] != faces[:, 2])
            & (faces[:, 2] != faces[:, 0]))
    faces = faces[keep]
    tri = verts[faces]
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    span = float(np.ptp(verts, axis=0).max())
    faces = faces[area > max(span ** 2 * 1e-12, 1e-12)]
    used = np.unique(faces)
    compact = np.full(len(verts), -1, dtype=np.int64)
    compact[used] = np.arange(len(used))
    return verts[used], compact[faces]


def symmetrize(mesh: Mesh, axis: int = 1, keep_negative: bool = True,
               weld_tol: float = 0.0) -> Mesh:
    """Keep one half of ``mesh`` and mirror it into a symmetric closed body."""
    half = clip_half(mesh, axis, keep_negative)
    both = half + mirror(half, axis)
    if weld_tol <= 0:
        weld_tol = float(np.ptp(mesh.vertices, axis=0).max()) * 1e-7
    v, f = _weld_seam(both.vertices, both.faces, axis, weld_tol)
    out = Mesh(v, f, mesh.two_sided, mesh.name + "-sym")
    return out.flipped() if out.is_closed and out.volume < 0 else out


def symmetry_error(mesh: Mesh, axis: int = 1, tol: int = 6):
    """Fraction of surface area whose plane has a mirror partner."""
    n = mesh.face_normals
    off = np.einsum("fk,fk->f", n, mesh.face_centroids)
    key = np.round(np.c_[n, off], tol)
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    area = np.bincount(inv, weights=mesh.face_areas)
    mir = uk.copy()
    mir[:, axis] *= -1
    have = {tuple(r) for r in uk}
    hit = np.array([tuple(np.round(r, tol)) in have for r in mir])
    return float(area[hit].sum() / area.sum())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh")
    ap.add_argument("--keep", default="-y", choices=["+y", "-y", "+x", "-x", "+z", "-z"],
                    help="which half to keep (default -y)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    axis = {"x": 0, "y": 1, "z": 2}[args.keep[1]]
    src = load_mesh(args.mesh)
    out = symmetrize(src, axis, keep_negative=args.keep[0] == "-")

    print(f"{'':22s} {'facets':>8s} {'closed':>7s} {'volume':>10s} {'area':>10s} "
          f"{'symmetric':>10s}")
    for label, m in (("source", src), ("symmetrized", out)):
        print(f"{label:22s} {m.n_faces:8d} {str(m.is_closed):>7s} {m.volume:10.3f} "
              f"{m.total_area:10.3f} {100*symmetry_error(m, axis):9.2f}%")
    save_obj(out, args.out)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
