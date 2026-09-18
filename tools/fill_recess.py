"""Fill a recess flush with the surrounding skin.

A pocket in the skin -- an inlet mouth, a scoop, a bay without its door -- is
a cavity, and a cavity is the one thing a single-bounce model cannot speak to
at all: its return builds over many reflections inside.  Sometimes the right
answer is to close it, either because the feature is not real or because the
design intent is a grid or a cover across the mouth, which electrically *is*
a flush surface.

This snaps everything inside a box onto the plane of the skin around it, then
cleans up the facets that collapse.  The body stays closed and the skin runs
straight across.

It suits a *pocket*: a recess in a reasonably flat patch of skin.  It does not
suit a louvre -- a grille of angled vanes with gaps between them -- because
there is no single floor to raise, and it does not suit a recess in a strongly
curved or canted patch, because one reference plane cannot represent the skin
there.  Both of those want the feature deleted and the opening re-lofted in
CAD, which is a modelling job rather than a mesh one.

    python tools/fill_recess.py model.obj --out filled.obj \\
        --box 2.6 3.9 1.9 2.9 -1.2 -0.1 --mirror-y
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from echo1.geometry import Mesh                                      # noqa: E402
from echo1.meshio import load_mesh, save_obj                         # noqa: E402


def _plane_fit(points, weights=None):
    """Area-weighted least-squares plane through ``points``; returns (normal, offset)."""
    w = np.ones(len(points)) if weights is None else np.asarray(weights, float)
    centre = (points * w[:, None]).sum(0) / w.sum()
    cov = ((points - centre) * w[:, None]).T @ (points - centre)
    normal = np.linalg.eigh(cov)[1][:, 0]
    return normal, float(normal @ centre)


def fill_recess(mesh: Mesh, box, axis: int = 2, side: int = +1,
                margin: float = 0.6, mirror_axis=None):
    """Flatten a recess onto the plane of the skin around it.

    ``box`` bounds the recess.  ``axis`` is the direction it opens along and
    ``side`` which way that is (+1 for a pocket in the upper surface, -1 for
    one in the belly).  The reference plane is fitted to the *outward-facing
    skin* in a ring of width ``margin`` around the box, so the fill matches
    the local surface rather than an average of the whole neighbourhood.

    With ``mirror_axis`` the mirrored recess is filled too, keeping a
    symmetric body symmetric.
    """
    boxes = [np.asarray(box, float).reshape(3, 2)]
    if mirror_axis is not None:
        m = boxes[0].copy()
        m[mirror_axis] = -m[mirror_axis][::-1]
        boxes.append(m)

    verts = mesh.vertices.copy()
    faces = mesh.faces
    filled = 0

    for bb in boxes:
        ring = bb.copy()
        ring[:, 0] -= margin
        ring[:, 1] += margin

        centroid = verts[faces].mean(axis=1)
        normals = Mesh(verts, faces).face_normals
        areas = Mesh(verts, faces).face_areas

        inside_vert = np.all((verts >= bb[:, 0]) & (verts <= bb[:, 1]), axis=1)
        in_ring = np.all((centroid >= ring[:, 0]) & (centroid <= ring[:, 1]), axis=1)
        in_box = np.all((centroid >= bb[:, 0]) & (centroid <= bb[:, 1]), axis=1)
        # Outward-facing skin around the recess.
        skin = in_ring & ~in_box & (normals[:, axis] * side > 0.5)
        if skin.sum() < 1 or not inside_vert.any():
            continue
        # Fit to the skin's vertices that lie OUTSIDE the recess.  A facet
        # straddling the rim is partly the pocket's own sloped wall, and
        # letting its depressed corners into the fit drags the reference plane
        # down into the hole.
        ref = np.unique(faces[skin])
        ref = ref[~inside_vert[ref]]
        if len(ref) < 3:
            continue
        pts = verts[ref]

        normal, offset = _plane_fit(pts)
        if normal[axis] * side < 0:
            normal, offset = -normal, -offset
        if abs(normal[axis]) < 1e-6:
            continue

        target = np.flatnonzero(inside_vert)
        if not len(target):
            continue
        # Push every vertex in the box out to the skin plane, but only inward
        # ones -- never pull the skin itself down into the hole.
        depth = (offset - verts[target] @ normal) / normal[axis]
        move = np.flatnonzero(depth * side > 0)
        verts[target[move], axis] += depth[move]
        filled += len(move)

    tri = verts[faces]
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    span = float(np.ptp(verts, axis=0).max())
    keep = area > max(span ** 2 * 1e-12, 1e-12)
    faces = faces[keep]

    used = np.unique(faces)
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    out = Mesh(verts[used], remap[faces], mesh.two_sided, mesh.name + "-filled")
    return out, filled, int((~keep).sum())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh")
    ap.add_argument("--out", required=True)
    ap.add_argument("--box", nargs=6, type=float, required=True,
                    metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"))
    ap.add_argument("--axis", type=int, default=2, choices=[0, 1, 2],
                    help="the direction the pocket opens along (default 2 = z)")
    ap.add_argument("--side", type=int, default=1, choices=[1, -1],
                    help="+1 for a recess in the upper surface, -1 for the belly")
    ap.add_argument("--margin", type=float, default=0.6,
                    help="width of the skin ring used to fit the reference plane")
    ap.add_argument("--mirror-y", action="store_true",
                    help="fill the mirrored pocket on the other side too")
    args = ap.parse_args(argv)

    src = load_mesh(args.mesh)
    out, moved, dropped = fill_recess(src, args.box, args.axis, args.side,
                                      args.margin,
                                      mirror_axis=1 if args.mirror_y else None)
    print(f"vertices pushed out to the skin plane: {moved}")
    print(f"facets that collapsed and were removed: {dropped}")
    print(f"\n{'':10s} {'facets':>7s} {'closed':>7s} {'volume':>9s} {'area':>9s}")
    for label, m in (("before", src), ("after", out)):
        print(f"{label:10s} {m.n_faces:7d} {str(m.is_closed):>7s} {m.volume:9.3f} "
              f"{m.total_area:9.3f}")
    save_obj(out, args.out)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
