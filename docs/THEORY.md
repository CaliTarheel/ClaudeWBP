# Theory

Conventions: time dependence `exp(+j w t)`, so an outgoing wave carries
`exp(-j k R)`. The incident field is a unit-amplitude plane wave with
propagation direction `i` and polarisation `e_i`; the scattered direction is
`s`, and `s = -i` for backscatter. `k = 2 pi / lambda`, `eta` is the free-space
impedance, and `n` is a facet's outward unit normal.

Radar cross section is `sigma = |S|^2`, where `S` is the complex amplitude the
code accumulates over facets and edges. Facet and edge contributions are summed
**coherently** — they interfere, and most of the structure in a signature is
that interference.

---

## 1. Physical optics over a flat facet

Physical optics replaces the true surface current with the tangent-plane
approximation

```
J = 2 n x H_i      on the lit surface,      J = 0 elsewhere.
```

Radiating that current into the far field leaves one integral over each facet,

```
I = integral over S of exp(j w . r) dS,     w = k (s - i),
```

and, for a perfect conductor,

```
sigma_rt = (k^2 / pi) |e_r . P|^2 |I|^2,    P = n x (i x e_t)
```

for transmit polarisation `e_t` and receive polarisation `e_r`.

### Gordon's closed form

`I` has a closed form on a flat polygon, which is why a facet model needs no
surface meshing at all. Split `w` into the part along the normal and the
in-plane remainder `w_t`; only `w_t` varies over the facet. In the facet plane,

```
div( u exp(j q u . rho) / (j q) ) = exp(j q u . rho),     u = w_t/|w_t|, q = |w_t|,
```

so the 2-D divergence theorem turns the area integral into a contour integral,
and each straight edge integrates to a sinc:

```
                                  1
I = exp(j w . r_0) * ----------------------- * SUM over edges m of
                            j |w_t|^2

        [ w_t . (L_m x n) ] * exp(j w_t . rho_m) * sinc( (w_t . L_m) / 2 )
```

with `L_m` the edge vector in counter-clockwise order about `n`, `rho_m` the
edge midpoint relative to the reference vertex `r_0`, and `sinc(x) = sin(x)/x`.
As `|w_t| -> 0` — the specular direction — the sum tends to the facet area.
This is Gordon (1975); `echo1.po.gordon_integral` implements it and
`gordon_integral_quadrature` integrates the same thing numerically so the test
suite can confirm it (they agree to ~1e-12).

At broadside on a plate this reduces to `sigma = 4 pi A^2 / lambda^2`, as it
must.

### An identity worth knowing

For monostatic co-polarised backscatter, with `i = -r` and `e_r = e_t = e`
perpendicular to `r`,

```
e . P = e . [ i (n . e) - e (n . i) ] = n . r
```

exactly. So the facet term switches off *smoothly* as a facet rotates edge-on:
physical optics backscatter has no discontinuity at a shadow boundary, and the
illumination mask can be applied without smoothing. (`tests/test_po.py`.)

---

## 2. Ufimtsev's edge waves

Physical optics knows only about surfaces. The true current near an edge differs
from the PO current, and Ufimtsev's idea is to radiate that difference — the
*fringe* current — as a line source along the edge:

```
f = f_total - f_PO          (soft: E parallel to the edge)
g = g_total - g_PO          (hard: H parallel to the edge)
```

Both terms are taken from the canonical wedge problem, a wedge of exterior angle
`n * pi`, with `phi'` and `phi` the incidence and observation azimuths in the
plane perpendicular to the edge, measured from one face.

### The total coefficients

The exact (Keller) wedge coefficients, with `mu = pi / n`:

```
f, g = (sin mu / n) * [  1 / (cos mu - cos((phi - phi')/n))
                      -+ 1 / (cos mu - cos((phi + phi')/n)) ]
```

upper sign for `f`, lower for `g`. For a half-plane (`n = 2`) this collapses to
the familiar `sec((phi -+ phi')/2)` pair.

### The physical-optics part, derived

The PO part is the contribution physical optics *already* makes near the edge,
which must be removed so it is not counted twice. Take a half-plane at `phi = 0`
lit from above, with `i = -(cos phi', sin phi')`. For E parallel to the edge the
PO current is `J = (2/eta) sin(phi') z`, and for H parallel it is `J = 2 x`,
radiating with a factor `-sin(phi)`. Both give a surface integral
`integral from 0 to infinity of exp(j a x) dx` whose endpoint contribution goes
as `1 / a` with

```
a = k (s - i) . x = k (cos phi + cos phi').
```

So the two coefficients have the shape `sin(phi')/(cos phi + cos phi')` and
`-sin(phi)/(cos phi + cos phi')`, and their **normalisation is not free**: the
whole point of PTD is that the fringe coefficient is finite at the reflection
and shadow boundaries, where the total and PO parts each diverge. Requiring the
poles to cancel at `phi + phi' = pi` and `phi - phi' = pi` fixes the constants
to exactly one:

```
f_PO = sin(phi') / (cos phi + cos phi')
g_PO = -sin(phi) / (cos phi + cos phi')
```

summed over whichever faces are lit — the *o* face for `0 < phi' < pi`, the *n*
face for `(n-1) pi < phi' < n pi`, the latter using the mirrored angles
`n pi - phi` and `n pi - phi'`. The same pole-cancellation check confirms the
mirrored form needs no extra sign.

Two consequences the code depends on. Coplanar facets give `n = 1`, where
`sin(mu) = 0` and the two face contributions cancel, so `f = g = 0` exactly —
tessellating a flat panel introduces no spurious edges. And on a genuine
boundary the code recovers the finite value by averaging across it, since the
difference is analytic there even though its parts are not.

### Equivalent edge currents

The fringe coefficients become electric and magnetic currents along the edge,

```
eta I_e = -(j/k) (e . E_i) f / sin^2(beta0)
    I_m = -(j/k) (e . eta H_i) g / sin^2(beta0)
```

radiated along each straight segment with the same `length * sinc` phase factor
as the facet integral. `beta0` is the angle between the edge and the incident
ray — the half-angle of the Keller cone.

**The constant is measured, not asserted.** With `f` and `g` normalised as
above, one number remains. It is fixed by a check that has an exact answer:
feed the *PO* coefficients through the edge machinery instead of the fringe
ones, and for a rectangular plate the result must reproduce the exact Gordon
integral over that plate. It does — in magnitude and sign, to machine precision,
at every aspect, every polarisation and every plate size. Any error in the
constant, the sign, the power of `sin(beta0)` or the azimuth convention breaks
it. That identity is `tests/test_ptd.py::test_edge_machinery_reproduces_exact_po_on_a_rectangle`,
and getting it right is what makes the `f_total - f_PO` subtraction remove
exactly the edge wave physical optics already contains rather than doubling it.

(The identity is exact for rectangles, where the two antiparallel edge pairs
carry weights in the ratio `cos^2 : sin^2` that sum to the closed form. For a
triangle the per-edge split differs, because the canonical coefficient is a
half-plane quantity while Gordon's is a finite-polygon one. That is expected and
does not affect the calibration, which needs only one geometry with an exact
answer.)

### On and off the Keller cone

The coefficients above hold on each edge's Keller cone. **Backscatter always
lies on it**, since `s = -i` makes the same angle with the edge as `i` does, so
for monostatic work — what ECHO computed, and what the signature problem needs —
this is the appropriate form. `bistatic_rcs` reuses the same coefficients with
projected azimuths, which is an approximation well off the cone; the docstring
says so.

Looking along an edge (`sin beta0 -> 0`) is a caustic of the theory and is
floored rather than allowed to diverge.

---

## 3. Visibility

A facet of a closed body carries current only when its outward normal has a
component towards the radar; an open sheet is lit from whichever side the radar
is on. Either may still be hidden behind the body, which is resolved by casting
a ray from the facet centroid (or edge midpoint) towards the radar. For a convex
body, orientation alone is exact and the ray trace can be switched off.

For a two-sided sheet lit from behind its stored normal, both `P` and the
winding of the Gordon contour flip; only `P`'s flip survives in the product, so
the amplitude carries an explicit sign. Without it, facets lit from opposite
sides combine with the wrong relative sign.

---

## 4. Where this stops being true

- **One bounce.** Corner reflectors need two and three, so `shapes.dihedral`
  and `shapes.trihedral` are counter-examples, not predictions;
  `echo1.analytic` has their closed forms.
- **One diffraction per edge.** No edge-to-edge interaction. This is the
  measured near-grazing shortfall in `validation/`: about 5 dB at 70–86° off
  normal, against under 1.5 dB at moderate aspects.
- **No creeping or travelling waves**, no surface-wave launch at a discontinuity.
- **Perfect conductors only** — no absorber, no coatings.
- **High frequency**, far field, single frequency.
