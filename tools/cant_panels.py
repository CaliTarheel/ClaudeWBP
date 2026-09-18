"""Cant panels whose normal lies near the horizon.

A panel whose outward normal sits in the horizontal plane is a mirror pointed
at every radar on the horizon: at broadside it returns ``4 pi A^2 / lambda^2``,
which for a panel the size of a dinner tray at X-band is tens of dBsm.  The
standard shaping answer is to give every such panel a cant, so its specular
flash leaves the threat plane -- the aircraft still has the panel, but nobody
at your altitude is standing in its mirror.

This finds the coplanar panel groups whose normal elevation is below a
threshold and rotates each one about a horizontal hinge at its base, pulling
the top inward along the panel's own outward normal.  Vertices shared between
panels take the average displacement, so the body stays closed and its
topology is untouched.

    python tools/cant_panels.py model.obj --out model-canted.obj \\
        --threshold 12 --cant 15
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from echo1.geometry import Mesh, normalize                          # noqa: E402
from echo1.meshio import load_mesh, save_obj                        # noqa: E402


def panel_groups(mesh: Mesh, tol: int = 4):
    """Group facets into coplanar panels, keyed by (normal, plane offset)."""
    n = mesh.face_normals
    off = np.einsum("fk,fk->f", n, mesh.face_centroids)
    key = np.round(np.c_[n, off], tol)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    return inv


def cant(mesh: Mesh, threshold_deg: float = 12.0, cant_deg: float = 15.0,
         min_area: float = 0.0, up=(0.0, 0.0, 1.0)):
    """Tilt near-horizontal-normal panels out of the threat plane.

    Returns ``(new_mesh, report)`` where ``report`` lists the panels moved.
    """
    up = normalize(np.asarray(up, float))
    group = panel_groups(mesh)
    n_groups = int(group.max()) + 1
    normals = mesh.face_normals
    areas = mesh.face_areas

    shift = np.zeros_like(mesh.vertices)
    weight = np.zeros(len(mesh.vertices))
    moved = []

    sin_thr = np.sin(np.radians(threshold_deg))
    tan_cant = np.tan(np.radians(cant_deg))

    for g in range(n_groups):
        faces = np.flatnonzero(group == g)
        area = float(areas[faces].sum())
        if area <= min_area:
            continue
        nrm = normals[faces[0]]
        elev = float(nrm @ up)
        if abs(elev) >= sin_thr:
            continue                                   # already canted enough
        horiz = normalize(nrm - elev * up)
        if not np.any(horiz):
            continue

        verts = np.unique(mesh.faces[faces])
        height = mesh.vertices[verts] @ up
        base = height.min()
        if height.max() - base < 1e-9:
            continue                                   # a horizontal strip, no hinge
        # Pull the top of the panel inward: the panel rotates about its base.
        delta = -np.outer((height - base) * tan_cant, horiz)
        np.add.at(shift, verts, delta)
        np.add.at(weight, verts, 1.0)
        moved.append((g, area, np.degrees(np.arcsin(np.clip(elev, -1, 1))),
                      float(np.degrees(np.arctan2(nrm[1], nrm[0])) % 360),
                      float(height.max() - base)))

    active = weight > 0
    shift[active] /= weight[active, None]
    out = Mesh(mesh.vertices + shift, mesh.faces, mesh.two_sided,
               mesh.name + "-canted")
    return out, moved, int(active.sum())


def census(mesh: Mesh, up=(0.0, 0.0, 1.0)):
    """Area carried by panels within a few degrees of the horizon."""
    up = normalize(np.asarray(up, float))
    elev = np.degrees(np.arcsin(np.clip(mesh.face_normals @ up, -1, 1)))
    out = {}
    for lim in (2.0, 5.0, 10.0, 15.0):
        sel = np.abs(elev) < lim
        out[lim] = (int(sel.sum()), float(mesh.face_areas[sel].sum()))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=12.0,
                    help="cant any panel whose normal is within this many degrees "
                         "of the horizontal plane (default 12)")
    ap.add_argument("--cant", type=float, default=15.0,
                    help="how far to tilt it, in degrees (default 15)")
    ap.add_argument("--min-area", type=float, default=0.0,
                    help="ignore panels smaller than this, in m^2")
    args = ap.parse_args(argv)

    src = load_mesh(args.mesh)
    out, moved, n_verts = cant(src, args.threshold, args.cant, args.min_area)

    print(f"panels canted: {len(moved)}  (vertices moved: {n_verts})")
    if moved:
        moved.sort(key=lambda r: -r[1])
        print(f"  {'area m^2':>9s} {'normal az':>10s} {'normal el':>10s} {'height m':>9s}")
        for _, area, elev, azn, h in moved[:12]:
            print(f"  {area:9.4f} {azn:10.1f} {elev:+10.2f} {h:9.3f}")
        print(f"  total area canted: {sum(r[1] for r in moved):.3f} m^2")

    print(f"\n{'':10s} {'facets':>7s} {'closed':>7s} {'volume':>9s} {'area':>9s}")
    for label, m in (("before", src), ("after", out)):
        print(f"{label:10s} {m.n_faces:7d} {str(m.is_closed):>7s} {m.volume:9.3f} "
              f"{m.total_area:9.3f}")
    print(f"\narea with normal near the horizon:")
    b, a = census(src), census(out)
    for lim in sorted(b):
        print(f"  within {lim:4.0f} deg: {b[lim][1]:8.3f} m^2 -> {a[lim][1]:8.3f} m^2 "
              f"({b[lim][0]:5d} -> {a[lim][0]:5d} facets)")
    save_obj(out, args.out)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
