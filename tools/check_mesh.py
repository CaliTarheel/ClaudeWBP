"""Check a mesh before spending an hour sweeping it.

Modelling packages are happy to hand you geometry that this solver cannot use,
or can use but will quietly answer wrongly about.  This reports the things that
matter and says which are fatal, which will bias the answer, and which are
merely worth knowing.

    python tools/check_mesh.py model.obj --freq 10GHz

What it looks for, in order of how much it will cost you:

  fatal      non-manifold edges, zero-area facets -- echo1 refuses these
  biasing    inward-facing normals, open boundaries, dihedral corners
  worth      asymmetry, facet size against the wavelength, wedge-angle spread
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from echo1 import analytic                                            # noqa: E402
from echo1.cli import _frequency                                      # noqa: E402
from echo1.geometry import Mesh                                       # noqa: E402
from echo1.meshio import load_mesh                                    # noqa: E402
from echo1.ptd import _MIN_WEDGE_N                                    # noqa: E402
from echo1.solver import dihedral_corners                             # noqa: E402


def raw_triangles(path):
    """Load without echo1's validation, so the fatal checks can run at all."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".obj":
        verts, faces = [], []
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                p = line.split()
                if not p:
                    continue
                if p[0] == "v":
                    verts.append([float(x) for x in p[1:4]])
                elif p[0] == "f":
                    idx = [int(t.split("/")[0]) - 1 for t in p[1:]]
                    faces += [[idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1)]
        return np.array(verts, float), np.array(faces, np.int64)
    m = load_mesh(path)
    return m.vertices, m.faces


def check(path, freq_hz: float):
    lam = analytic.wavelength(freq_hz)
    verts, faces = raw_triangles(path)
    fatal, biasing, notes = [], [], []

    print(f"{os.path.basename(str(path))}: {len(verts)} vertices, {len(faces)} triangles")

    tri = verts[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    span = float(np.ptp(verts, axis=0).max())
    # Match what echo1.geometry.Mesh actually refuses, so this does not cry wolf.
    tiny = area <= 1e-12
    if tiny.any():
        fatal.append(f"{int(tiny.sum())} zero-area facets "
                     f"(Blender: Mesh > Clean Up > Degenerate Dissolve)")
    sliver = (area > 1e-12) & (area <= span ** 2 * 1e-12)
    if sliver.any():
        notes.append(f"{int(sliver.sum())} sliver facets, under 1e-12 of the model's "
                     f"area. Harmless to the answer, but they are what tears holes "
                     f"when a mesh is cut or welded")

    edges = np.sort(np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    if (counts > 2).any():
        fatal.append(f"{int((counts > 2).sum())} non-manifold edges shared by 3+ faces "
                     f"(Blender: Select > All by Trait > Non Manifold)")
    open_edges = int((counts == 1).sum())

    dup = len(verts) - len(np.unique(np.round(verts / max(span, 1e-12) * 1e9), axis=0))
    if dup:
        notes.append(f"{dup} duplicate vertices at 1e-9 of the model size "
                     f"(Blender: M > By Distance)")

    if fatal:
        print("\n  FATAL -- echo1 will refuse this mesh:")
        for f in fatal:
            print(f"    - {f}")
        return False

    mesh = Mesh(verts, faces, name="checked")
    lo, hi = mesh.extent
    print(f"  {hi[0]-lo[0]:.3f} x {hi[1]-lo[1]:.3f} x {hi[2]-lo[2]:.3f} m, "
          f"area {mesh.total_area:.3f} m^2, {mesh.max_dimension/lam:.0f} wavelengths across")

    e = mesh.edges
    boundary = e.n_adjacent == 1
    if open_edges:
        biasing.append(f"{open_edges} open boundary edges ({e.length[boundary].sum():.3f} m). "
                       f"Every one becomes a knife edge that diffracts as if the skin "
                       f"ended there")
    else:
        print(f"  closed body, volume {mesh.volume:.3f} m^3")
        if mesh.volume < 0:
            biasing.append("normals point inward (negative volume). Illumination and the "
                           "PO sign will both be wrong "
                           "(Blender: Shift+N, Recalculate Outside)")

    n_corner, l_corner = dihedral_corners(mesh)
    if n_corner:
        biasing.append(f"{n_corner} near-right-angle re-entrant edges ({l_corner:.1f} m): "
                       f"dihedral corners. Their double-bounce return is not modelled, so "
                       f"the true RCS is HIGHER than predicted in their retroreflection "
                       f"directions")
    dropped = e.wedge_n < _MIN_WEDGE_N
    if dropped.any():
        notes.append(f"{int(dropped.sum())} edges ({e.length[dropped].sum():.1f} m) are too "
                     f"re-entrant to model and will be excluded")

    n = e.wedge_n
    flat = np.abs(n - 1.0) <= 1e-9
    print(f"  edges: {int(flat.sum())} exactly coplanar (silent), "
          f"{int((n > 1 + 1e-9).sum())} convex, {int((n < 1 - 1e-9).sum())} re-entrant")

    sym = _symmetry(mesh)
    print(f"  left-right symmetry: {100*sym:.2f}% of area")
    if sym < 0.999:
        notes.append(f"{100*(1-sym):.1f}% of area has no mirror partner. Mirror aspects can "
                     f"differ by tens of dB; build one side and mirror it in Blender")

    small = area < (lam / 4) ** 2
    print(f"  facets below (lambda/4)^2: {int(small.sum())} of {len(area)}, "
          f"carrying {100*area[small].sum()/area.sum():.1f}% of the area")
    if area[small].sum() / area.sum() < 0.05 and small.sum() > len(area) * 0.2:
        notes.append("most facets are sub-wavelength but carry almost no area: detail you "
                     "can delete freely, it is costing run time rather than signature")

    if biasing:
        print("\n  WILL BIAS THE ANSWER:")
        for b in biasing:
            print(f"    - {b}")
    if notes:
        print("\n  worth knowing:")
        for t in notes:
            print(f"    - {t}")
    if not biasing and not notes:
        print("\n  clean.")
    return True


def _symmetry(mesh: Mesh, axis: int = 1, tol: int = 4):
    n = mesh.face_normals
    off = np.einsum("fk,fk->f", n, mesh.face_centroids)
    key = np.round(np.c_[n, off], tol)
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    a = np.bincount(inv, weights=mesh.face_areas)
    mir = uk.copy()
    mir[:, axis] *= -1
    have = {tuple(r) for r in uk}
    hit = np.array([tuple(np.round(r, tol)) in have for r in mir])
    return float(a[hit].sum() / a.sum())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh", nargs="+")
    ap.add_argument("--freq", default="10GHz")
    args = ap.parse_args(argv)
    ok = True
    for i, path in enumerate(args.mesh):
        if i:
            print()
        ok &= check(path, _frequency(args.freq))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
