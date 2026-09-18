"""Run a CAD model through echo1 and report its signature.

    python tools/step_to_mesh.py jet.step --out-dir meshes --scale 0.001
    python examples/cad_signature.py meshes/jet.obj --freq 10GHz --az-step 0.25

Expects a mesh already in echo1's frame: nose along +x, wings along +/-y, up
along +z.  ``--axes`` permutes a model that came out of CAD in some other
orientation (``zxy`` means "the source z axis is my x axis", and so on).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from echo1 import analytic, plotting                                 # noqa: E402
from echo1.cli import _frequency                                     # noqa: E402
from echo1.geometry import Mesh                                      # noqa: E402
from echo1.meshio import load_mesh                                   # noqa: E402
from echo1.solver import monostatic_rcs                              # noqa: E402

_AXIS = {"x": 0, "y": 1, "z": 2}


def reorient(mesh: Mesh, axes: str, centre: bool = True) -> Mesh:
    """Permute axes so the model sits in echo1's frame, preserving handedness."""
    if len(axes) != 3 or set(axes) != set("xyz"):
        raise ValueError("--axes must be a permutation of 'xyz', e.g. zxy")
    rot = np.zeros((3, 3))
    for row, name in enumerate(axes):
        rot[row, _AXIS[name]] = 1.0
    if np.linalg.det(rot) < 0:                    # keep it a rotation, not a mirror
        rot[2] *= -1.0
    v = mesh.vertices @ rot.T
    if centre:
        v = v - (v.min(axis=0) + v.max(axis=0)) / 2.0
    return Mesh(v, mesh.faces, mesh.two_sided, mesh.name)


def describe(mesh: Mesh, lam: float) -> None:
    lo, hi = mesh.extent
    n = mesh.edges.wedge_n
    print(f"{mesh.name}")
    print(f"  {mesh.n_faces} facets, {len(mesh.edges)} edges, "
          f"{mesh.total_area:.1f} m^2 wetted area")
    print(f"  length {hi[0]-lo[0]:.3f} m  span {hi[1]-lo[1]:.3f} m  "
          f"height {hi[2]-lo[2]:.3f} m")
    print(f"  closed body: {mesh.is_closed}"
          + (f", volume {mesh.volume:.2f} m^3" if mesh.is_closed else ""))
    print(f"  electrical size {mesh.max_dimension/lam:.0f} wavelengths")
    print(f"  edges: {int(np.sum(np.abs(n-1) <= 1e-9))} coplanar (silent), "
          f"{int(np.sum(n > 1+1e-9))} convex, {int(np.sum(n < 1-1e-9))} re-entrant")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh")
    ap.add_argument("--freq", default="10GHz")
    ap.add_argument("--az-step", type=float, default=0.25)
    ap.add_argument("--el", type=float, default=0.0)
    ap.add_argument("--pol", default="VV,HH")
    ap.add_argument("--axes", default="xyz", help="source axis order, e.g. zxy")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out-dir", default="examples/output")
    ap.add_argument("--tag", default=None, help="prefix for the output files")
    ap.add_argument("--no-occlusion", action="store_true")
    args = ap.parse_args(argv)

    freq = _frequency(args.freq)
    lam = analytic.wavelength(freq)
    mesh = load_mesh(args.mesh)
    if args.scale != 1.0:
        mesh = mesh.scaled(args.scale)
    mesh = reorient(mesh, args.axes)
    describe(mesh, lam)

    az = np.arange(0.0, 360.0, args.az_step)
    pols = tuple(p.strip().upper() for p in args.pol.split(","))
    print(f"\nsweeping {len(az)} aspects at {freq/1e9:.3f} GHz "
          f"(lambda = {lam*100:.2f} cm), elevation {args.el:g} deg, {'+'.join(pols)}")
    t0 = time.time()
    result = monostatic_rcs(mesh, freq, az, args.el, pols=pols,
                            occlusion=not args.no_occlusion)
    print(f"  took {time.time()-t0:.1f} s "
          f"({(time.time()-t0)/len(az)*1000:.0f} ms per aspect)\n")

    dropped, length = result.excluded_edges
    if dropped:
        print(f"note: {dropped} edges ({length:.1f} m) were too re-entrant for "
              f"single-bounce PTD and were excluded\n")

    for p in pols:
        print(result.summary(p))

    tag = args.tag or os.path.splitext(os.path.basename(args.mesh))[0]
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, tag)
    result.to_csv(f"{base}-rcs.csv")
    plotting.save_figure(plotting.polar_rcs(result, floor_db=-40.0), f"{base}-polar.png")
    plotting.save_figure(
        plotting.cartesian_rcs(result, show_components=True, floor_db=-40.0),
        f"{base}-cut.png")
    plotting.save_figure(plotting.plot_mesh(mesh), f"{base}-geometry.png")
    print(f"\nwrote {base}-rcs.csv, -polar.png, -cut.png, -geometry.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
