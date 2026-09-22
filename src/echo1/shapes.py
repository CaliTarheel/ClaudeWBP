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
from .ptd import fringe_coefficients

__all__ = [
    "plate", "polygon_plate", "disc", "box", "sphere", "cylinder", "dihedral",
    "trihedral", "hopeless_diamond", "faceted_delta", "faceted_fighter",
    "spike_azimuths", "specular_aspects", "panels",
    "BUILTIN",
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


def _wedge_plate(outline, thickness: float, inset: float = 0.25):
    """A four-sided plate that tapers to a sharp rim, as vertices and faces.

    Giving a fin thickness the obvious way -- offsetting the outline both ways
    and capping it -- leaves a narrow flat strip all round the rim, and a strip
    is a flat plate: a 12 cm one at 10 GHz peaks near 27 dBsm, which is louder
    than the 40 m^2 skin panel it is bolted to seen anywhere but its own
    specular.  So the rim is kept sharp and the thickness is carried on an
    interior ridge instead, parallel to the first pair of sides so it adds no
    new edge direction.
    """
    p0, p1, p2, p3 = (np.asarray(q, float) for q in outline)
    rise = p3 - p0
    a0 = p0 + inset * (p1 - p0) + 0.5 * rise
    a1 = p1 - inset * (p1 - p0) + 0.5 * rise
    nrm = np.cross(p1 - p0, rise)
    nrm = nrm / np.linalg.norm(nrm) * (thickness / 2.0)
    verts = [p0, p1, p2, p3, a0 + nrm, a1 + nrm, a0 - nrm, a1 - nrm]
    P0, P1, P2, P3, A0, A1, B0, B1 = range(8)
    faces = (_quad(P0, P1, A1, A0) + [[P1, P2, A1]]
             + _quad(P2, P3, A0, A1) + [[P3, P0, A0]]
             + _quad(P1, P0, B0, B1) + [[P2, P1, B1]]
             + _quad(P3, P2, B1, B0) + [[P0, P3, B0]])
    return verts, faces


def _kernel_clearance(outline, p):
    """How far inside every edge's supporting line a plan point sits.

    A fan of triangles from a single point tiles a polygon only if the point
    can see the whole boundary -- it must lie in the polygon's *kernel*.  A
    sawtooth planform is not convex, so this is a real constraint and not a
    formality: an apex outside the kernel makes the skin fold back over itself,
    which is still a closed mesh and still returns a volume, but is not a
    surface any aircraft has.  Negative means outside.
    """
    poly = np.asarray(outline, float)[:, :2]
    p = np.asarray(p, float)[:2]
    nxt = np.roll(poly, -1, axis=0)
    d = nxt - poly
    inward = np.column_stack([-d[:, 1], d[:, 0]])                  # left of the walk
    inward /= np.linalg.norm(inward, axis=1, keepdims=True)
    twice_area = np.sum(poly[:, 0] * nxt[:, 1] - nxt[:, 0] * poly[:, 1])
    inward *= np.sign(twice_area)                     # whichever way it was walked
    return float(np.min(np.einsum("ij,ij->i", p - poly, inward)))


def _skin_height(chine, apex, p):
    """Height of the upper skin above a plan point, on the fan from the apex."""
    a = apex[:2]
    for i in range(len(chine)):
        b, c = chine[i][:2], chine[(i + 1) % len(chine)][:2]
        d = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(d) < 1e-12:
            continue
        u = ((p[0] - a[0]) * (c[1] - a[1]) - (p[1] - a[1]) * (c[0] - a[0])) / d
        v = ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) / d
        if u >= -1e-9 and v >= -1e-9 and u + v <= 1.0 + 1e-9:
            return float((1.0 - u - v) * apex[2])   # chine sits at z = 0
    raise ValueError(f"plan point {p} is not over the upper skin")


def faceted_fighter(length: float = 20.0, span: float = 13.2, height: float = 2.6,
                    depth: float = 1.7, teeth: int = 2, tooth: float = 0.12,
                    apex_frac: float = 0.60, keel_frac: float = 0.64,
                    fin_height: float = 2.6, fin_cant_deg: float = 35.0,
                    fin_chord: float = 0.22, fin_x: float = 0.55,
                    fin_y: float = 0.42, fin_thickness: float = 0.12,
                    fin_align: bool = True, fin_sweep_deg: float = 70.0,
                    name: str = "faceted-fighter") -> Mesh:
    """A faceted airframe generated from the shaping rules, not traced.

    Built the way the rules say to build one, so that every feature is there
    for a reason rather than by imitation:

    * The planform is walked out along **two directions only**.  The leading
      edges set them; the sawtooth trailing edge is then made of segments
      parallel to the opposite leading edge, so notching the tail adds no new
      edge direction.
    * The body is a double pyramid over that planform -- an apex above, a
      keel below -- so every facet is a large flat panel tilted well off the
      horizontal, and none of them mirrors a radar at the aircraft's own
      altitude.
    * The fins are **parallelograms in plan**, their root and tip chords on one
      planform direction and their leading and trailing edges on the other, so
      canting them outboard for control authority costs no new edge direction.

    All edge diffraction therefore collapses into the four azimuths that
    :func:`spike_azimuths` predicts in closed form, plus the beam, where the
    fore-and-aft ridges of the pyramid throw.

    An aligned sawtooth necessarily makes the aircraft *longer* than the plain
    diamond it notches: zigzagging adds length.  ``tooth`` is the depth of the
    notch as a fraction of a tooth, and the leading-edge sweep is solved from
    it so that ``length`` and ``span`` come out as asked.

    Parameters
    ----------
    length, span : float
        Overall dimensions in metres.
    height, depth : float
        Apex above and keel below the chine plane.
    teeth : int
        Sawtooth teeth per side.  1 gives a plain diamond.
    tooth : float
        Notch depth as a fraction of a tooth; 0 gives a plain diamond.
    apex_frac, keel_frac : float
        Where the apex and keel sit along the length, from the nose.
    fin_chord, fin_x, fin_y : float
        Fin root chord as a fraction of ``length``; and where its forward end
        sits, aft of the nose as a fraction of ``length`` and outboard as a
        fraction of the half-span.
    fin_align : bool
        Keep the fins on the planform directions.  Setting it ``False`` builds
        the same aircraft with conventionally swept fins -- a fore-and-aft root
        and a leading edge at ``fin_sweep_deg`` from the lateral axis -- which
        is what the rule is there to stop.  Useful for showing the cost.
    """
    if teeth < 1:
        raise ValueError("teeth must be at least 1")
    if not 0.0 <= tooth < 1.0:
        raise ValueError("tooth must lie in [0, 1)")

    k = int(teeth)
    reach = k - tooth * (k - 1)
    # Solve the leading-edge angle so the walked planform closes on the
    # requested length and span.
    cot_phi = length * reach / (span * k)
    phi = float(np.arctan2(1.0, cot_phi))
    half = span / 2.0

    d_le = np.array([-np.cos(phi), np.sin(phi)])        # nose -> port tip
    d_in = np.array([-np.cos(phi), -np.sin(phi)])       # aft and inboard
    d_out = d_le                                        # aft and outboard
    step = half / (np.sin(phi) * reach)

    nose = np.array([length / 2.0, 0.0])
    port = [nose, nose + (half / np.sin(phi)) * d_le]   # nose, tip
    for i in range(k):
        port.append(port[-1] + step * d_in)
        if i < k - 1:
            port.append(port[-1] + tooth * step * d_out)
    port[-1][1] = 0.0                                   # land exactly on the centreline

    # Chine, counter-clockwise seen from above: nose, port side, tail, starboard.
    chine = [np.array([p[0], p[1], 0.0]) for p in port]
    chine += [np.array([p[0], -p[1], 0.0]) for p in port[-2:0:-1]]

    nose_x = chine[0][0]
    apex = np.array([nose_x - apex_frac * length, 0.0, height])
    keel = np.array([nose_x - keel_frac * length, 0.0, -depth])
    for who, pt, frac in (("apex", apex, apex_frac), ("keel", keel, keel_frac)):
        if _kernel_clearance(chine, pt) < 0.0:
            raise ValueError(
                f"{who}_frac={frac} puts the {who} outside the planform's kernel, "
                "so the skin fanned from it folds over itself.  A sawtooth "
                "planform is not convex; move the peak aft (larger *_frac) or "
                "set teeth=1.")

    verts = chine + [apex, keel]
    n_chine = len(chine)
    A, K = n_chine, n_chine + 1
    faces = []
    for i in range(n_chine):
        j = (i + 1) % n_chine
        faces.append([i, j, A])                          # upper skin
        faces.append([j, i, K])                          # lower skin

    # Fins: a parallelogram is planar whatever its corners, so the root chord
    # can follow the skin down while every edge stays on a planform direction.
    cant = np.radians(fin_cant_deg)
    chord = fin_chord * length
    lateral = fin_height * np.tan(cant)
    psi = np.radians(90.0 - fin_sweep_deg)
    for s in (1.0, -1.0):
        if fin_align:
            u_root = np.array([-np.cos(phi), -s * np.sin(phi), 0.0])   # aft, inboard
            u_cant = np.array([-np.cos(phi), s * np.sin(phi), 0.0])    # aft, outboard
        else:
            u_root = np.array([-1.0, 0.0, 0.0])                        # straight aft
            u_cant = np.array([-np.sin(psi), s * np.cos(psi), 0.0])
        p0 = np.array([nose_x - fin_x * length, s * fin_y * half, 0.0])
        p1 = p0 + chord * u_root
        root = []
        for p in (p0, p1):
            p = p.copy()
            p[2] = _skin_height(chine, apex, p) - 0.1 * fin_height
            root.append(p)
        rise = lateral * u_cant + np.array([0.0, 0.0, fin_height])
        outline = [root[0], root[1], root[1] + rise, root[0] + rise]
        fv, ff = _wedge_plate(outline, fin_thickness)
        i0 = len(verts)
        verts += fv
        faces += [[a + i0, b + i0, c + i0] for a, b, c in ff]

    mesh = Mesh(np.array(verts), np.array(faces), name=name)
    return mesh.flipped() if mesh.volume < 0 else mesh


def panels(mesh: Mesh, tol: float = 1e-6):
    """Index each facet by the flat panel it belongs to.

    Triangles are how a mesh stores a panel, not what a panel *is*: a 40 m^2
    plate split into eight triangles scatters as one 40 m^2 plate, not as eight
    5 m^2 ones, and since a specular return goes as area squared that is a 9 dB
    difference.  Facets joined across an exactly coplanar edge are one panel.
    """
    parent = list(range(mesh.n_faces))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    e = mesh.edges
    flat = (np.abs(e.wedge_n - 1.0) < tol) & (e.n_adjacent == 2)
    for f0, f1 in e.faces[flat][:, :2]:
        a, b = find(int(f0)), find(int(f1))
        if a != b:
            parent[a] = b
    return np.array([find(i) for i in range(mesh.n_faces)])


def specular_aspects(mesh: Mesh, min_area: float = 0.0, freq_hz: float = 10e9):
    """Aspects where each flat panel mirrors, and how loud it is when it does.

    The facet counterpart to :func:`spike_azimuths`.  A flat panel returns
    ``4 pi A^2 / lambda^2`` straight back along its own normal and very little
    anywhere else, so its normal *is* the aspect to keep away from.  Coplanar
    facets are gathered into panels first.  Returns ``(azimuth_deg,
    elevation_deg, area, peak_dbsm)`` tuples, loudest first.

    What the list is for is not the loudest row but the *elevation* column: a
    peak at 70 degrees below is a peak no radar will ever stand in, while a
    small facet whose normal lies near the horizon is a hole in the design
    however modest its area.
    """
    lam = 299792458.0 / freq_hz
    group = panels(mesh)
    rows = []
    for g in np.unique(group):
        m = group == g
        area = float(mesh.face_areas[m].sum())
        if area <= min_area:
            continue
        n = mesh.face_normals[m][int(np.argmax(mesh.face_areas[m]))]
        rows.append((float(np.degrees(np.arctan2(n[1], n[0])) % 360.0),
                     float(np.degrees(np.arcsin(np.clip(n[2], -1.0, 1.0)))),
                     area,
                     float(10.0 * np.log10(4.0 * np.pi * area ** 2 / lam ** 2))))
    return sorted(rows, key=lambda r: -r[3])


def spike_azimuths(mesh: Mesh, min_length: float = 1.0, tol_deg: float = 1.0,
                   weight: str = "fringe"):
    """Azimuths where a body's long edges will throw their diffraction lobes.

    A straight edge radiates into the plane perpendicular to itself, so at a
    given elevation its flash lands where the line of sight is broadside to it.
    Returns ``(azimuth_deg, strength)`` pairs, strongest first -- the prediction
    to check a sweep against, before spending anything on a solve.

    Length alone over-ranks a long shallow crease.  The default weighting
    therefore scales each edge by its fringe coefficient at the symmetric
    broadside geometry, which is what actually sets how hard it radiates, so
    the strength reads in *knife-edge-equivalent metres*: a metre of sharp
    edge counts one, a metre of a 20-degree crease counts about a fifth.  Pass
    ``weight="length"`` for the raw geometry.
    """
    e = mesh.edges
    keep = e.length > min_length
    if not np.any(keep):
        return []
    az = np.degrees(np.arctan2(e.e_hat[keep, 1], e.e_hat[keep, 0]))
    if weight == "length":
        w = e.length[keep]
    elif weight == "fringe":
        n = e.wedge_n[keep]
        f, g = fringe_coefficients(n * np.pi / 2.0, n * np.pi / 2.0, n)
        w = e.length[keep] * np.hypot(np.abs(f), np.abs(g)) / np.sqrt(2.0)
    else:
        raise ValueError("weight must be 'fringe' or 'length'")
    live = w > 1e-9                                   # a flat joint throws nothing
    if not np.any(live):
        return []
    az, w = az[live], w[live]
    flashes = np.concatenate([az + 90.0, az - 90.0]) % 360.0
    weight = np.concatenate([w] * 2)
    order = np.argsort(flashes)
    flashes, weight = flashes[order], weight[order]
    out = []
    i = 0
    while i < len(flashes):
        j = i
        while j + 1 < len(flashes) and flashes[j + 1] - flashes[i] < tol_deg:
            j += 1
        out.append((float(np.average(flashes[i:j + 1], weights=weight[i:j + 1])),
                    float(weight[i:j + 1].sum())))
        i = j + 1
    if len(out) > 1 and (out[0][0] + 360.0) - out[-1][0] < tol_deg:
        first, last = out.pop(0), out.pop()           # one cluster straddling 0 deg
        total = first[1] + last[1]
        az = (last[0] + first[1] / total * ((first[0] + 360.0) - last[0])) % 360.0
        out.append((float(az), float(total)))
    return sorted(out, key=lambda r: -r[1])


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
    "faceted-fighter": faceted_fighter,
}
