"""Convert a STEP assembly into echo1 facet meshes.

STEP is a boundary-representation format: faces are trimmed analytic surfaces,
not triangles.  This script tessellates them with gmsh's OpenCASCADE kernel and
writes one OBJ per solid, plus a combined OBJ.

A *planar* face tessellates exactly -- the triangles lie in the real surface,
so for a faceted airframe there is no geometric approximation at all.  Curved
faces are chorded, and the deviation matters: see docs/THEORY.md.

Needs gmsh (``pip install gmsh``); it is not a dependency of echo1 itself.

    python tools/step_to_mesh.py model.step --out-dir meshes --scale 0.001
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def tessellate(path, scale=1.0, mesh_size=0.0, min_size=0.0, verbose=False):
    """Tessellate a STEP file, returning ``[(name, vertices, triangles), ...]``.

    One entry per solid.  Vertices are multiplied by ``scale``.  ``mesh_size``
    of 0 asks for the coarsest mesh the topology allows, which for planar faces
    is a minimal triangulation of each polygon.
    """
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("step")
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()

        big = mesh_size if mesh_size > 0 else 1e22
        gmsh.option.setNumber("Mesh.MeshSizeMax", big)
        gmsh.option.setNumber("Mesh.MeshSizeMin", min_size)
        # Keep the mesh as coarse as the geometry allows: nodes only where the
        # CAD puts vertices, so planar polygons are not subdivided.
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 5)          # Delaunay
        gmsh.model.mesh.generate(2)

        volumes = gmsh.model.getEntities(3)
        groups = []
        if volumes:
            for dim, tag in volumes:
                name = gmsh.model.getEntityName(dim, tag) or f"solid{tag}"
                faces = [abs(t) for _, t in gmsh.model.getBoundary([(dim, tag)],
                                                                  oriented=False)]
                groups.append((name, faces))
        else:                                               # sheet bodies only
            groups.append(("surfaces", [t for _, t in gmsh.model.getEntities(2)]))

        out = []
        for name, faces in groups:
            tris, coords = [], {}
            for f in faces:
                types, _, node_tags = gmsh.model.mesh.getElements(2, f)
                for etype, nodes in zip(types, node_tags):
                    if etype != 2:                          # 3-node triangle
                        continue
                    tris.extend(np.asarray(nodes, dtype=np.int64).reshape(-1, 3))
            if not tris:
                continue
            tris = np.array(tris)
            uniq = np.unique(tris)
            for t in uniq:
                c, _, _, _ = gmsh.model.mesh.getNode(int(t))
                coords[int(t)] = c
            remap = {t: i for i, t in enumerate(uniq)}
            verts = np.array([coords[int(t)] for t in uniq], float) * scale
            faces_idx = np.vectorize(remap.get)(tris)
            out.append((_clean_name(name), verts, faces_idx))
        return out
    finally:
        gmsh.finalize()


def weld(verts, faces, tol):
    """Merge vertices closer together than ``tol`` and drop collapsed triangles.

    CAD tessellation leaves slivers -- triangles with a micron-scale edge that
    carry no area but, if simply deleted, would leave the mesh open and turn a
    seam into a spurious knife edge.  Welding the coincident vertices instead
    collapses the sliver to a repeated index and heals the topology.
    """
    from scipy.spatial import cKDTree

    parent = np.arange(len(verts))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in cKDTree(verts).query_pairs(tol, output_type="ndarray"):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    roots = np.array([find(i) for i in range(len(verts))])
    uniq, inv = np.unique(roots, return_inverse=True)
    merged = np.zeros((len(uniq), 3))
    counts = np.bincount(inv, minlength=len(uniq)).astype(float)
    for axis in range(3):
        merged[:, axis] = np.bincount(inv, weights=verts[:, axis],
                                      minlength=len(uniq)) / counts

    faces = inv[faces]
    keep = ((faces[:, 0] != faces[:, 1])
            & (faces[:, 1] != faces[:, 2])
            & (faces[:, 2] != faces[:, 0]))
    return merged, faces[keep], int((~keep).sum())


def drop_degenerate(verts, faces, min_area):
    tri = verts[faces]
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    keep = area > min_area
    return faces[keep], int((~keep).sum())


def _clean_name(name):
    keep = "".join(c if (c.isalnum() or c in "-_") else "-" for c in str(name))
    return keep.strip("-").lower() or "solid"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", help="input .step / .stp file")
    ap.add_argument("--out-dir", default="meshes", help="where to write the OBJ files")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply coordinates by this (STEP mm -> metres is 0.001)")
    ap.add_argument("--mesh-size", type=float, default=0.0,
                    help="target facet size in output units; 0 means as coarse as possible")
    ap.add_argument("--weld-tol", type=float, default=0.0,
                    help="merge vertices closer than this; 0 uses 1e-6 of the model size")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    from echo1.geometry import Mesh
    from echo1.meshio import save_obj

    solids = tessellate(args.step, args.scale, args.mesh_size, verbose=args.verbose)
    if not solids:
        print("no solids found", file=sys.stderr)
        return 1

    span = max(float(np.ptp(np.vstack([v for _, v, _ in solids]), axis=0).max()), 1e-12)
    tol = args.weld_tol if args.weld_tol > 0 else span * 1e-6

    os.makedirs(args.out_dir, exist_ok=True)
    combined = None
    print(f"{'solid':22s} {'facets':>8s} {'welded':>7s} {'dropped':>8s} "
          f"{'closed':>7s} {'volume':>12s}")
    for name, verts, faces in solids:
        verts, faces, collapsed = weld(verts, faces, tol)
        faces, dropped = drop_degenerate(verts, faces, span ** 2 * 1e-18)
        mesh = Mesh(verts, faces, name=name)
        if mesh.is_closed and mesh.volume < 0:
            mesh = mesh.flipped()
        path = os.path.join(args.out_dir, f"{name}.obj")
        save_obj(mesh, path)
        print(f"{name[:22]:22s} {mesh.n_faces:8d} {collapsed:7d} {dropped:8d} "
              f"{str(mesh.is_closed):>7s} {mesh.volume:12.6g}")
        combined = mesh if combined is None else combined + mesh

    combined.name = _clean_name(os.path.splitext(os.path.basename(args.step))[0])
    out = os.path.join(args.out_dir, f"{combined.name}.obj")
    save_obj(combined, out)
    lo, hi = combined.extent
    print(f"\ncombined: {combined.n_faces} facets, {len(combined.edges)} edges, "
          f"{combined.total_area:.4g} area units^2")
    print(f"  bounding box  x {lo[0]:.4g}..{hi[0]:.4g}  "
          f"y {lo[1]:.4g}..{hi[1]:.4g}  z {lo[2]:.4g}..{hi[2]:.4g}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
