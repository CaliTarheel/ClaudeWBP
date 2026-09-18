"""STEP file in, radar signature out.

Chains the whole sequence this project ended up needing on a real CAD model:

    tessellate -> reorient -> symmetrize -> cant panels off the horizon
               -> swing forward faces to the planform -> sweep -> report

Each stage can be skipped, and every intermediate mesh is written out so you
can look at what changed.  Needs gmsh for the STEP stage (``pip install gmsh``).

    python tools/pipeline.py jet.step --out-dir work --scale 0.001 --axes zxy
    python tools/pipeline.py jet.obj  --out-dir work --skip-symmetrize
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from cant_panels import axis_census, cant, census, sweep_to_planform    # noqa: E402
from symmetrize import symmetrize, symmetry_error                       # noqa: E402

from echo1 import analytic, plotting                                    # noqa: E402
from echo1.cli import _frequency                                        # noqa: E402
from echo1.geometry import Mesh                                         # noqa: E402
from echo1.meshio import load_mesh, save_obj                            # noqa: E402
from echo1.solver import dihedral_corners, monostatic_rcs               # noqa: E402

_AXIS = {"x": 0, "y": 1, "z": 2}


def reorient(mesh: Mesh, axes: str) -> Mesh:
    rot = np.zeros((3, 3))
    for row, name in enumerate(axes):
        rot[row, _AXIS[name]] = 1.0
    if np.linalg.det(rot) < 0:
        rot[2] *= -1.0
    v = mesh.vertices @ rot.T
    return Mesh(v - (v.min(0) + v.max(0)) / 2.0, mesh.faces, mesh.two_sided, mesh.name)


def _report(tag: str, mesh: Mesh, lam: float) -> None:
    lo, hi = mesh.extent
    print(f"  {tag:22s} {mesh.n_faces:6d} facets  closed={str(mesh.is_closed):5s} "
          f"vol={mesh.volume:8.3f}  area={mesh.total_area:8.3f}  "
          f"{hi[0]-lo[0]:.2f} x {hi[1]-lo[1]:.2f} x {hi[2]-lo[2]:.2f} m")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help=".step/.stp, or an .obj/.stl already tessellated")
    ap.add_argument("--out-dir", default="work")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply coordinates (STEP mm -> m is 0.001)")
    ap.add_argument("--axes", default="xyz",
                    help="source axis order, so the result is nose +x, span +/-y, up +z")
    ap.add_argument("--freq", default="10GHz")
    ap.add_argument("--az-step", type=float, default=2.0)
    ap.add_argument("--elevations", default="0",
                    help="comma-separated elevations to sweep; negative = radar below")
    ap.add_argument("--keep", default="+y", help="half to keep when symmetrizing")
    ap.add_argument("--cant", type=float, default=15.0)
    ap.add_argument("--cant-threshold", type=float, default=12.0)
    ap.add_argument("--planform-az", type=float, default=0.0,
                    help="planform angle for the forward-face sweep; 0 measures it")
    ap.add_argument("--skip-symmetrize", action="store_true")
    ap.add_argument("--skip-cant", action="store_true")
    ap.add_argument("--skip-planform", action="store_true")
    ap.add_argument("--no-sweep", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    freq = _frequency(args.freq)
    lam = analytic.wavelength(freq)

    print(f"echo1 pipeline: {args.source}  @ {freq/1e9:.3f} GHz "
          f"(lambda = {lam*100:.2f} cm)\n")

    if os.path.splitext(args.source)[1].lower() in (".step", ".stp"):
        from step_to_mesh import drop_degenerate, tessellate, weld
        solids = tessellate(args.source, args.scale)
        span = max(float(np.ptp(np.vstack([v for _, v, _ in solids]), axis=0).max()), 1e-12)
        mesh = None
        for name, verts, faces in solids:
            verts, faces, _ = weld(verts, faces, span * 1e-6)
            faces, _ = drop_degenerate(verts, faces, span ** 2 * 1e-18)
            part = Mesh(verts, faces, name=name)
            if part.is_closed and part.volume < 0:
                part = part.flipped()
            mesh = part if mesh is None else mesh + part
        print(f"  tessellated {len(solids)} solids from STEP")
    else:
        mesh = load_mesh(args.source)
        if args.scale != 1.0:
            mesh = mesh.scaled(args.scale)

    mesh.name = os.path.splitext(os.path.basename(args.source))[0]
    mesh = reorient(mesh, args.axes)
    stages = [("0-imported", mesh)]
    _report("imported", mesh, lam)

    if not args.skip_symmetrize:
        axis = _AXIS[args.keep[1]]
        before = symmetry_error(mesh, axis)
        mesh = symmetrize(mesh, axis, keep_negative=args.keep[0] == "-")
        print(f"  symmetry: {100*before:.2f}% -> {100*symmetry_error(mesh, axis):.2f}% of area")
        stages.append(("1-symmetric", mesh))
        _report("symmetrized", mesh, lam)

    if not args.skip_cant:
        before = census(mesh)
        mesh, moved, _ = cant(mesh, args.cant_threshold, args.cant)
        print(f"  canted {len(moved)} panels; area within 2 deg of the horizon "
              f"{before[2.0][1]:.3f} -> {census(mesh)[2.0][1]:.3f} m^2")
        stages.append(("2-canted", mesh))
        _report("canted", mesh, lam)

    if not args.skip_planform:
        target = args.planform_az
        if target <= 0:
            e = mesh.edges
            long = e.length > 0.5
            ang = np.degrees(np.arctan2(np.abs(e.e_hat[long, 1]), e.e_hat[long, 0])) % 180
            h, edges = np.histogram(ang, bins=180, range=(0, 180), weights=e.length[long])
            swept = [i for i in np.argsort(-h) if 10 < edges[i] < 80]
            target = float(90.0 - edges[swept[0]]) if swept else 36.5
            print(f"  measured planform angle: edges at {edges[swept[0]]:.0f} deg "
                  f"-> faces swung to {target:.1f} deg")
        before = axis_census(mesh)
        mesh, moved, _ = sweep_to_planform(mesh, threshold_deg=20.0, target_az_deg=target)
        print(f"  swung {len(moved)} forward-facing panels; area within 5 deg of the "
              f"nose axis {before[5.0][1]:.4f} -> {axis_census(mesh)[5.0][1]:.4f} m^2")
        stages.append(("3-planform", mesh))
        _report("planform-swept", mesh, lam)

    for tag, m in stages:
        save_obj(m, os.path.join(args.out_dir, f"{tag}.obj"))
    print(f"\n  meshes written to {args.out_dir}/")

    n_corner, l_corner = dihedral_corners(mesh)
    if n_corner:
        print(f"  note: {n_corner} near-right-angle re-entrant edges ({l_corner:.1f} m). "
              f"Their double-bounce return is not modelled, so the real RCS is higher "
              f"than this in their retroreflection directions.")

    if args.no_sweep:
        return 0

    az = np.arange(0.0, 360.0, args.az_step)
    print(f"\n  sweeping {len(az)} aspects x {len(args.elevations.split(','))} elevations")
    for el in (float(e) for e in args.elevations.split(",")):
        rows = []
        for tag, m in stages:
            t0 = time.time()
            s = monostatic_rcs(m, freq, az, el, pols=("VV",)).sigma["VV"]
            rows.append((tag, s))
            print(f"    {tag:14s} el {el:+5.1f}  {time.time()-t0:5.0f}s", flush=True)
        print(f"\n  elevation {el:+.1f} deg (VV)")
        print(f"    {'stage':16s} {'peak':>9s} {'median':>9s} {'power-mean':>11s}")
        for tag, s in rows:
            db = analytic.to_dbsm(s)
            print(f"    {tag:16s} {db.max():7.1f}dB {np.median(db):7.1f}dB "
                  f"{analytic.to_dbsm(s.mean()):9.1f}dB")
        np.save(os.path.join(args.out_dir, f"sweep_el{el:+.0f}.npy"),
                {"az": az, **{t: s for t, s in rows}}, allow_pickle=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
