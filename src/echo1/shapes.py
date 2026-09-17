"""Built-in faceted bodies.

Canonical test shapes with known answers, plus the two that make ECHO's point:
the *hopeless diamond*, the faceted lifting body that fell out of a code which
could only handle flat panels, and a planform-aligned faceted delta, which
shows the other half of the trick -- once edges are unavoidable, make them
parallel so their spikes pile up in a few azimuths you can steer away from.

Everything returns a :class:`~echo1.geometry.Mesh` with outward-facing normals.
"""

from __future__ import annotations

import numpy as np

from .geometry import Mesh

__all__ = [
    "plate", "polygon_plate", "disc", "box", "sphere", "cylinder", "dihedral",
    "trihedral", "hopeless_diamond", "faceted_delta", "BUILTIN",
]


def _quad(a, b, c, d):
    return [[a, b, c], [a, c, d]]


def plate(width: float = 1.0, height: float = 1.0, name: str = "plate") -> Mesh:
    """Flat rectangular plate in the x-y plane, normal along ``+z``, two-sided."""
    w, h = width / 2.0, height / 2.0
    v = np.array([[-w, -h, 0], [w, -h, 0], [w, h, 0], [-w, h, 0]], float)
    return Mesh(v, np.array(_quad(0, 1, 2, 3)), two_sided=True, name=name)


def polygon_plate(points, name: str = "polygon-plate") -> Mesh:
    """Flat plate with an arbitrary polygon outline, in the x-y plane.

    ``points`` is a sequence of ``(x, y)`` vertices in order.  The outline is
    triangulated from its centroid, so the interior edges are coplanar and
    contribute no diffraction -- only the rim does.  Useful for isolating what
    an outline's edge directions do, without a body's other edges in the way.
    """
    pts = np.asarray(points, float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        raise ValueError("points must be a sequence of at least three (x, y) pairs")
    centre = pts.mean(axis=0)
    v = np.vstack([np.column_stack([pts, np.zeros(len(pts))]),
                   [[centre[0], centre[1], 0.0]]])
    c = len(pts)
    f = [[c, i, (i + 1) % len(pts)] for i in range(len(pts))]
    return Mesh(v, np.array(f), two_sided=True, name=name)


def disc(radius: float = 0.5, segments: int = 48, name: str = "disc") -> Mesh:
    """Polygonal approximation to a flat circular disc, normal along ``+z``."""
    if segments < 3:
        raise ValueError("segments must be at least 3")
    t = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    rim = np.stack([radius * np.cos(t), radius * np.sin(t), np.zeros_like(t)], axis=1)
    v = np.vstack([[[0.0, 0.0, 0.0]], rim])
    f = [[0, 1 + i, 1 + (i + 1) % segments] for i in range(segments)]
    return Mesh(v, np.array(f), two_sided=True, name=name)


def box(lx: float = 1.0, ly: float = 1.0, lz: float = 1.0, name: str = "box") -> Mesh:
    """Closed rectangular box centred on the origin."""
    x, y, z = lx / 2.0, ly / 2.0, lz / 2.0
    v = np.array([
        [-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
        [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z],
    ], float)
    f = (_quad(0, 3, 2, 1) + _quad(4, 5, 6, 7) + _quad(0, 1, 5, 4)
         + _quad(1, 2, 6, 5) + _quad(2, 3, 7, 6) + _quad(3, 0, 4, 7))
    return Mesh(v, np.array(f), name=name)


def sphere(radius: float = 0.5, subdivisions: int = 3, name: str = "sphere") -> Mesh:
    """Icosphere: a sphere tessellated into near-uniform triangles."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], float)
    f = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ])
    for _ in range(int(subdivisions)):
        cache, new_v, new_f = {}, list(v), []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                cache[key] = len(new_v)
                new_v.append((new_v[a] + new_v[b]) / 2.0)
            return cache[key]

        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new_f += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        v, f = np.array(new_v), np.array(new_f)

    v = v / np.linalg.norm(v, axis=1, keepdims=True) * radius
    return Mesh(v, f, name=name)


def cylinder(radius: float = 0.25, length: float = 1.0, segments: int = 36,
             capped: bool = True, name: str = "cylinder") -> Mesh:
    """Faceted circular cylinder with its axis along ``+z``."""
    t = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring = np.stack([radius * np.cos(t), radius * np.sin(t)], axis=1)
    lo = np.hstack([ring, np.full((segments, 1), -length / 2.0)])
    hi = np.hstack([ring, np.full((segments, 1), length / 2.0)])
    v = np.vstack([lo, hi])
    f = []
    for i in range(segments):
        j = (i + 1) % segments
        f += _quad(i, j, segments + j, segments + i)
    if capped:
        c_lo, c_hi = len(v), len(v) + 1
        v = np.vstack([v, [[0, 0, -length / 2.0], [0, 0, length / 2.0]]])
        for i in range(segments):
            j = (i + 1) % segments
            f.append([c_lo, j, i])
            f.append([c_hi, segments + i, segments + j])
    return Mesh(v, np.array(f), two_sided=not capped, name=name)


def dihedral(a: float = 0.3, b: float = 0.3, name: str = "dihedral") -> Mesh:
    """Right-angle dihedral: two plates meeting along the ``y`` axis.

    A double-bounce shape.  Single-bounce PO+PTD cannot reproduce its peak --
    it is here precisely as the counter-example.
    """
    v = np.array([
        [a, -b / 2, 0], [0, -b / 2, 0], [0, b / 2, 0], [a, b / 2, 0],
        [0, -b / 2, a], [0, b / 2, a],
    ], float)
    # Wound so that both normals point into the reflecting quadrant.
    f = np.array(_quad(3, 2, 1, 0) + _quad(1, 2, 5, 4))
    return Mesh(v, f, two_sided=True, name=name)


def trihedral(a: float = 0.3, name: str = "trihedral") -> Mesh:
    """Triangular right-angle trihedral (a corner reflector).

    Like :func:`dihedral`, a triple-bounce shape included as a known limit of
    the single-bounce model.
    """
    v = np.array([[0, 0, 0], [a, 0, 0], [0, a, 0], [0, 0, a]], float)
    # Wound so that all three normals point into the reflecting octant.
    f = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3]])
    return Mesh(v, f, two_sided=True, name=name)


def hopeless_diamond(length: float = 12.0, span: float = 9.0, height: float = 1.9,
                     depth: float = 1.3, tail_frac: float = 0.28,
                     tail_span_frac: float = 0.42, transom_rake: float = 0.12,
                     name: str = "hopeless-diamond") -> Mesh:
    """A faceted diamond lifting body, nose along ``+x``.

    The shape is a closed body built entirely from flat triangles: a diamond
    planform at mid-height (the chine), a ridge line above it and a keel line
    below, with every surface between them planar.  Nothing here is curved,
    because the code it is meant for could not have analysed a curve.

    Parameters
    ----------
    length, span : float
        Overall length and maximum width, in metres.
    height, depth : float
        Rise above and drop below the chine plane.
    tail_frac : float
        Position of the trailing edge behind the widest point, as a fraction
        of ``length``.
    tail_span_frac : float
        Trailing-edge width as a fraction of ``span``.
    transom_rake : float
        How far forward the aft *ridge* point sits, as a fraction of ``length``.
        Zero leaves a flat aft face: a specular mirror pointed straight down the
        tail, worth +52 dBsm at 10 GHz for the default body -- more than a
        truck.  Raking the ridge alone leans the two aft panels back as well as
        outward, so their normals tilt out of the horizontal plane entirely and
        the flash leaves the threat sector.  Doing exactly this, panel by panel,
        is what a code like ECHO is for.
    """
    nose = length * (1.0 - tail_frac)
    tail = -length * tail_frac
    hw = span / 2.0
    tw = span * tail_span_frac / 2.0

    chine = np.array([
        [nose, 0.0, 0.0],      # 0 nose
        [0.0, hw, 0.0],        # 1 widest, port
        [tail, tw, 0.0],       # 2 trailing edge, port
        [tail, -tw, 0.0],      # 3 trailing edge, starboard
        [0.0, -hw, 0.0],       # 4 widest, starboard
    ])
    apex_f = np.array([nose * 0.42, 0.0, height])       # forward ridge point
    rake = length * transom_rake
    apex_a = np.array([tail + rake, 0.0, height * 0.62])  # aft ridge point
    keel_f = np.array([nose * 0.48, 0.0, -depth])
    keel_a = np.array([tail, 0.0, -depth * 0.55])

    v = np.vstack([chine, apex_f, apex_a, keel_f, keel_a])
    N, W_P, T_P, T_S, W_S, AF, AA, KF, KA = range(9)

    f = [
        # upper surface
        [N, W_P, AF], [W_P, AA, AF], [W_P, T_P, AA],
        [N, AF, W_S], [W_S, AF, AA], [W_S, AA, T_S],
        # lower surface (opposite winding)
        [N, KF, W_P], [W_P, KF, KA], [W_P, KA, T_P],
        [N, W_S, KF], [W_S, KA, KF], [W_S, T_S, KA],
        # flat transom, in the x = tail plane
        [AA, T_P, KA], [AA, KA, T_S],
    ]
    return Mesh(v, np.array(f), name=name)


def faceted_delta(length: float = 20.0, span: float = 13.0, height: float = 2.4,
                  depth: float = 1.5, tip_frac: float = 0.5,
                  fin_height: float = 2.2, fin_cant_deg: float = 32.0,
                  fin_thickness: float = 0.10, name: str = "faceted-delta") -> Mesh:
    """A planform-aligned faceted delta with canted fins, nose along ``+x``.

    The planform is a symmetric diamond: the nose sits at ``+length/2``, the
    wing tips at mid-length, and the tail apex at ``-length/2``, so the two
    trailing edges are mirror images of the two leading edges.  All four
    therefore lie along just **two** directions, and the fins are canted rather
    than vertical so that they add no new ones.

    That is the second lesson a diffraction-aware RCS code teaches.  Edges
    cannot be removed, but each one throws its energy into a narrow fan
    perpendicular to itself; make all the edges parallel and those fans
    collapse into a few sharp azimuthal spikes with quiet sectors between
    them, instead of a smear of returns in every direction.  It is why the
    F-117 looks the way it does, and :func:`echo1.shapes.hopeless_diamond`
    against this shape shows the difference.

    ``tip_frac`` is where the wing tips sit along the length, measured back
    from the nose.  At the default ``0.5`` the leading and trailing edges are
    mirror images and the alignment is exact; any other value gives the
    trailing edges a different sweep from the leading edges and breaks it,
    which is how ``examples/signatures.py`` isolates what alignment is worth.

    The resulting leading-edge sweep, measured from the spanwise axis, is
    ``degrees(arctan(2 * tip_frac * length / span))``.
    """
    if not 0.05 < tip_frac < 0.95:
        raise ValueError("tip_frac must lie strictly between 0.05 and 0.95")
    hw, hl = span / 2.0, length / 2.0
    tip_x = hl - length * tip_frac
    verts = [
        np.array([hl, 0.0, 0.0]),                    # 0 nose
        np.array([tip_x, hw, 0.0]),                  # 1 port tip
        np.array([-hl, 0.0, 0.0]),                   # 2 tail apex
        np.array([tip_x, -hw, 0.0]),                 # 3 starboard tip
        np.array([hl - length * tip_frac * 0.40, 0.0, height]),     # 4 ridge
        np.array([hl - length * tip_frac * 0.32, 0.0, -depth]),     # 5 keel
    ]
    NOSE, TIP_P, TAIL, TIP_S, AP, KE = range(6)
    faces = [
        [NOSE, TIP_P, AP], [TIP_P, TAIL, AP],
        [NOSE, AP, TIP_S], [TIP_S, AP, TAIL],
        [NOSE, KE, TIP_P], [TIP_P, KE, TAIL],
        [NOSE, TIP_S, KE], [TIP_S, TAIL, KE],
    ]

    # Canted fins, as thin closed wedges so the body stays a closed solid.
    # Each fin's own leading edge is swept to match the wing's.
    cant = np.radians(fin_cant_deg)
    fin_sweep = np.arctan2(hl - tip_x, hw)
    for sign in (+1.0, -1.0):
        root_a = np.array([tip_x - (hl - tip_x) * 0.34, sign * hw * 0.17, height * 0.30])
        root_b = np.array([tip_x - (hl - tip_x) * 1.10, sign * hw * 0.06, height * 0.14])
        tip = root_a + np.array([
            -fin_height / np.tan(fin_sweep),
            sign * fin_height * np.sin(cant),
            fin_height * np.cos(cant),
        ])
        outline = [root_a, root_b, tip]
        nrm = np.cross(root_b - root_a, tip - root_a)
        nrm = nrm / np.linalg.norm(nrm) * (fin_thickness / 2.0)
        i = len(verts)
        verts += [q + nrm for q in outline] + [q - nrm for q in outline]
        faces += [[i, i + 1, i + 2], [i + 3, i + 5, i + 4]]
        for u, w in ((0, 1), (1, 2), (2, 0)):
            faces += _quad(i + u, i + 3 + u, i + 3 + w, i + w)

    mesh = Mesh(np.array(verts), np.array(faces), name=name)
    return mesh.flipped() if mesh.volume < 0 else mesh


#: Built-in shapes, by the name the command line uses.
BUILTIN = {
    "plate": plate,
    "disc": disc,
    "box": box,
    "sphere": sphere,
    "cylinder": cylinder,
    "dihedral": dihedral,
    "trihedral": trihedral,
    "hopeless-diamond": hopeless_diamond,
    "faceted-delta": faceted_delta,
}
