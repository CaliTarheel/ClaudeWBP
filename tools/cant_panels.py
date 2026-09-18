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
    """Group facets into *connected* coplanar panels.

    Keying on the plane alone is not enough: coplanar slivers anywhere on the
    body share a plane, so a "panel" could span the whole aircraft while
    carrying very little area.  Rotating such a group about its centroid moves
    metal by metres.  A panel is a coplanar patch whose facets actually touch.
    """
    n = mesh.face_normals
    off = np.einsum("fk,fk->f", n, mesh.face_centroids)
    _, plane = np.unique(np.round(np.c_[n, off], tol), axis=0, return_inverse=True)

    parent = np.arange(mesh.n_faces)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edges = mesh.edges
    shared = edges.faces[edges.n_adjacent == 2]
    same = plane[shared[:, 0]] == plane[shared[:, 1]]
    for a, b in shared[same]:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    roots = np.array([find(i) for i in range(mesh.n_faces)])
    _, inv = np.unique(roots, return_inverse=True)
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


def sweep_to_planform(mesh: Mesh, axis=(1.0, 0.0, 0.0), threshold_deg: float = 20.0,
                      target_az_deg: float = 36.5, up=(0.0, 0.0, 1.0),
                      min_area: float = 0.0):
    """Swing panels that face down a threat axis around to the planform angle.

    Canting a panel out of the horizontal plane does nothing for a panel that
    faces straight *along* the threat axis, and for a small panel it cannot:
    a facet of size ``D`` has a specular lobe about ``lambda / D`` wide, so a
    14 cm panel at X-band still answers a radar 12 degrees off its normal.

    The fix is the other half of the shaping rule.  Rotate the panel about the
    vertical until its normal lies at the planform angle -- parallel to the
    wing edges -- so its flash lands in the same few azimuths everything else
    already uses, and the head-on sector is left clean.  The panel turns about
    its own centroid, so a 14 cm face moves only a few centimetres of metal.

    Returns ``(new_mesh, report)``.
    """
    axis = normalize(np.asarray(axis, float))
    up = normalize(np.asarray(up, float))
    group = panel_groups(mesh)
    normals = mesh.face_normals
    areas = mesh.face_areas
    centroids = mesh.face_centroids

    shift = np.zeros_like(mesh.vertices)
    weight = np.zeros(len(mesh.vertices))
    moved = []
    cos_thr = np.cos(np.radians(threshold_deg))

    for g in range(int(group.max()) + 1):
        faces = np.flatnonzero(group == g)
        area = float(areas[faces].sum())
        if area <= min_area:
            continue
        nrm = normals[faces[0]]
        if float(nrm @ axis) < cos_thr:
            continue

        verts = np.unique(mesh.faces[faces])
        centre = (areas[faces, None] * centroids[faces]).sum(0) / area
        side = 1.0 if centre[1] >= 0.0 else -1.0
        now = np.arctan2(nrm[1], nrm[0])
        delta = side * np.radians(target_az_deg) - now

        c, s_ = np.cos(delta), np.sin(delta)
        rot = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
        rel = mesh.vertices[verts] - centre
        np.add.at(shift, verts, rel @ rot.T - rel)
        np.add.at(weight, verts, 1.0)
        moved.append((area, float(np.degrees(now) % 360), float(np.degrees(delta)),
                      float(np.abs(rel @ rot.T - rel).max())))

    active = weight > 0
    shift[active] /= weight[active, None]
    out = Mesh(mesh.vertices + shift, mesh.faces, mesh.two_sided,
               mesh.name + "-swept")
    return out, moved, int(active.sum())


def axis_census(mesh: Mesh, axis=(1.0, 0.0, 0.0)):
    """Area presented within a few degrees of a threat axis."""
    axis = normalize(np.asarray(axis, float))
    cos = mesh.face_normals @ axis
    return {lim: (int((cos > np.cos(np.radians(lim))).sum()),
                  float(mesh.face_areas[cos > np.cos(np.radians(lim))].sum()))
            for lim in (5.0, 10.0, 20.0, 30.0)}


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
    ap.add_argument("--mode", default="horizon", choices=["horizon", "axis"],
                    help="'horizon' cants panels off the horizontal plane; "
                         "'axis' swings panels that face down the threat axis "
                         "round to the planform angle")
    ap.add_argument("--target-az", type=float, default=36.5,
                    help="planform angle to swing them to, in degrees (axis mode)")
    args = ap.parse_args(argv)

    src = load_mesh(args.mesh)
    if args.mode == "axis":
        out, moved, n_verts = sweep_to_planform(
            src, threshold_deg=args.threshold, target_az_deg=args.target_az,
            min_area=args.min_area)
        print(f"panels swung to the planform angle: {len(moved)}  "
              f"(vertices moved: {n_verts})")
        if moved:
            moved.sort(key=lambda r: -r[0])
            print(f"  {'area m^2':>9s} {'was az':>8s} {'turned':>8s} {'metal moved':>12s}")
            for area, was, delta, disp in moved[:10]:
                print(f"  {area:9.4f} {was:8.1f} {delta:+8.1f} {disp*1000:9.1f} mm")
            print(f"  total area swung: {sum(r[0] for r in moved):.4f} m^2")
        b, a = axis_census(src), axis_census(out)
        print(f"\narea presented within a few degrees of the threat axis:")
        for lim in sorted(b):
            print(f"  within {lim:4.0f} deg: {b[lim][1]:8.4f} m^2 -> {a[lim][1]:8.4f} m^2 "
                  f"({b[lim][0]:5d} -> {a[lim][0]:5d} facets)")
        print(f"\n{'':10s} {'facets':>7s} {'closed':>7s} {'volume':>9s} {'area':>9s}")
        for label, m in (("before", src), ("after", out)):
            print(f"{label:10s} {m.n_faces:7d} {str(m.is_closed):>7s} {m.volume:9.3f} "
                  f"{m.total_area:9.3f}")
        save_obj(out, args.out)
        print(f"\nwrote {args.out}")
        return 0

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
