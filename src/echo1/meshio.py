"""Reading and writing faceted geometry: Wavefront OBJ and STL.

Only what a facet model needs: vertices and triangles.  Polygons with more
than three vertices are fanned into triangles, since the scattering kernels
assume triangles (and a planar polygon fans without any loss).
"""

from __future__ import annotations

import os
import struct

import numpy as np

from .geometry import Mesh

__all__ = ["load_mesh", "save_mesh", "load_obj", "save_obj", "load_stl", "save_stl"]


def _fan(poly):
    return [[poly[0], poly[i], poly[i + 1]] for i in range(1, len(poly) - 1)]


def load_obj(path, two_sided: bool = False) -> Mesh:
    verts, faces = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                verts.append([float(x) for x in parts[1:4]])
            elif parts[0] == "f":
                idx = []
                for token in parts[1:]:
                    n = int(token.split("/")[0])
                    idx.append(n - 1 if n > 0 else len(verts) + n)
                if len(idx) >= 3:
                    faces.extend(_fan(idx))
    if not faces:
        raise ValueError(f"no faces found in {path}")
    name = os.path.splitext(os.path.basename(str(path)))[0]
    return Mesh(np.array(verts, float), np.array(faces, np.int64), two_sided, name)


def save_obj(mesh: Mesh, path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# echo1 mesh: {mesh.name}\n")
        for v in mesh.vertices:
            fh.write(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
        for f in mesh.faces:
            fh.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")


def load_stl(path, two_sided: bool = False) -> Mesh:
    with open(path, "rb") as fh:
        head = fh.read(5)
        fh.seek(0)
        if head[:5] == b"solid":
            text = fh.read().decode("utf-8", errors="replace")
            if "facet" in text and "vertex" in text:
                pts = [
                    [float(x) for x in ln.split()[1:4]]
                    for ln in text.splitlines()
                    if ln.strip().startswith("vertex")
                ]
                tri = np.array(pts, float).reshape(-1, 3, 3)
                return _from_triangles(tri, path, two_sided)
        fh.seek(84)
        raw = fh.read()
    count = len(raw) // 50
    tri = np.zeros((count, 3, 3))
    for i in range(count):
        vals = struct.unpack_from("<12fH", raw, i * 50)
        tri[i] = np.array(vals[3:12]).reshape(3, 3)
    return _from_triangles(tri, path, two_sided)


def _from_triangles(tri: np.ndarray, path, two_sided: bool) -> Mesh:
    flat = tri.reshape(-1, 3)
    uniq, inv = np.unique(np.round(flat, 9), axis=0, return_inverse=True)
    name = os.path.splitext(os.path.basename(str(path)))[0]
    return Mesh(uniq, inv.reshape(-1, 3).astype(np.int64), two_sided, name)


def save_stl(mesh: Mesh, path, binary: bool = True) -> None:
    tri = mesh.triangles
    nrm = mesh.face_normals
    if binary:
        with open(path, "wb") as fh:
            fh.write(f"echo1 {mesh.name}".ljust(80)[:80].encode("ascii", "replace"))
            fh.write(struct.pack("<I", len(tri)))
            for n, t in zip(nrm, tri):
                fh.write(struct.pack("<12fH", *n, *t[0], *t[1], *t[2], 0))
        return
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"solid {mesh.name}\n")
        for n, t in zip(nrm, tri):
            fh.write(f" facet normal {n[0]:.9g} {n[1]:.9g} {n[2]:.9g}\n  outer loop\n")
            for p in t:
                fh.write(f"   vertex {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")
            fh.write("  endloop\n endfacet\n")
        fh.write(f"endsolid {mesh.name}\n")


def load_mesh(path, two_sided: bool = False) -> Mesh:
    """Load OBJ or STL, chosen by file extension."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".obj":
        return load_obj(path, two_sided)
    if ext == ".stl":
        return load_stl(path, two_sided)
    raise ValueError(f"unsupported mesh format {ext!r}; use .obj or .stl")


def save_mesh(mesh: Mesh, path, **kwargs) -> None:
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".obj":
        return save_obj(mesh, path)
    if ext == ".stl":
        return save_stl(mesh, path, **kwargs)
    raise ValueError(f"unsupported mesh format {ext!r}; use .obj or .stl")
