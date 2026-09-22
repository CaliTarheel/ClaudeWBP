# echo1

A faceted **radar-cross-section predictor** in Python, rebuilding the capability
of Lockheed's *ECHO 1*: physical optics over flat panels, plus Ufimtsev's
physical theory of diffraction for their edges.

```
pip install -e ".[dev]"
echo1 sweep --shape hopeless-diamond --freq 10GHz --az 0:360:0.25 --polar rcs.png
```

## What ECHO 1 was

In 1975 Denys Overholser, a radar specialist at Lockheed's Skunk Works, wrote a
program he called **ECHO 1**. Lockheed's own account:

> For the first time, we were able to develop a computer program to precisely
> predict the radar cross section (RCS) return of any given shape — provided it
> could be made from flat panels.

It was built on a 1962 monograph by the Soviet physicist Pyotr Ufimtsev,
*Method of Edge Waves in the Physical Theory of Diffraction*, translated by the
USAF Foreign Technology Division in 1971 — a paper that showed how to compute
the scattering from an edge, and which nobody had thought to point at aircraft.
The proposal ECHO 1 supported won the DARPA contract that became Have Blue, and
then the F-117. The "provided it could be made from flat panels" clause is why
the F-117 is faceted: the airframe was shaped to fit the only code that could
analyse it.

This package is that capability, reconstructed from the open physics. It is not
Lockheed's code, which is not public; it is an independent implementation of
what that code computed, checked against closed-form results and against an
independent numerical solution of Maxwell's equations.

## What it does

Give it a body made of flat triangular facets and it returns monostatic (or
bistatic) RCS against aspect angle, polarisation and frequency.

| | mechanism | status |
|---|---|---|
| Facet returns | physical optics, Gordon's closed-form polygon integral | exact for the PO current, no surface meshing |
| Edge returns | Ufimtsev fringe waves via equivalent edge currents | first order, exact on the Keller cone |
| Hiding | facet orientation plus grid-accelerated ray tracing | exact for convex bodies without the ray trace |
| Polarisation | VV, HH, HV, VH | co- and cross-polarised |

```python
import numpy as np
from echo1 import shapes, monostatic_rcs

body = shapes.hopeless_diamond()
r = monostatic_rcs(body, freq_hz=10e9, az_deg=np.arange(0, 360, 0.25))
print(r.summary("VV"))
r.to_csv("signature.csv")
```

```
hopeless-diamond: VV at 10.000 GHz (lambda = 3.00 cm), 1440 aspects
  peak       13.19 dBsm at az 90.00 deg, el 0.00 deg
  median    -40.28 dBsm
  mean      -12.92 dBsm (power-averaged)
  minimum   -77.46 dBsm
```

A 53 dB spread between the peak and the median *is* the lesson ECHO taught. The
design problem is not "reduce the RCS" but "point the spikes where no radar is
looking" — and you cannot do that without a code that tells you where they are.

### What it is for

`examples/signatures.py` runs the two design rules that fall out of a code like
this, and prints the numbers below (it regenerates its own figures, so they are
not committed).

**Tilt every panel out of the threat plane.** The hopeless diamond's aft transom
is a flat face. Left vertical, it is a mirror pointed straight down the tail:

```
  flat aft face   peak  52.5 dBsm at az 180.0
  ridge raked     peak  13.2 dBsm at az  90.0      -> 39 dB, for moving one vertex
```

**Then make the edges parallel.** Once no panel faces the radar, the edges set
the floor, and an edge cannot be tilted away — it throws its energy into a fan
perpendicular to itself. What you can do is give the outline fewer distinct
directions, so the fans pile up into fewer, narrower spikes. Two flat plates of
equal area, 25° off the plate plane:

```
  aligned diamond      1 edge direction  ->  8 lobes, covering 2.33% of azimuth
  irregular outline    4 edge directions -> 12 lobes, covering 3.21% of azimuth
```

That is planform alignment, and it is why the F-117's wing, tail, intakes and
door edges all run along the same few lines.

**Then check that you actually did.** `examples/faceted_fighter.py` builds a
20 m faceted airframe out of those two rules and then audits it. Two predictors
answer in closed form, before anything is solved:

| | what it answers | how |
|---|---|---|
| `shapes.spike_azimuths` | where the edges will throw | an edge radiates perpendicular to itself, weighted by that wedge's fringe coefficient |
| `shapes.specular_aspects` | where each panel will mirror, and how loud | `4πA²/λ²` straight back along its own normal |

Weighting by the coefficient rather than by length is what makes the first one
usable: a 41 m fore-and-aft spine at `n = 1.15` is a fifth as loud, per metre,
as an 11 m knife at `n = 1.81`, and ranking by length puts it first. Run
against the aircraft, the two say four azimuths carry 23.5 knife-edge-equivalent
metres each, the beam carries 8.6, and nothing mirrors within 23.8° of the
horizon. The solve then puts every lobe within 0.03° of where they said.

The audit found four faults, which is the point of having it:

| found by | what was wrong | what it cost |
|---|---|---|
| `check_mesh` | right-angle re-entrant edges nobody designed — the skin fanned from a peak outside the planform's kernel had folded through itself | corner reflectors, and a mesh that sweeps without complaining |
| `spike_azimuths` | fins swept conventionally, adding a third and fourth edge direction | 5 dB on the nose-sector mean, and a new lobe pair 20° off the tail |
| `specular_aspects` | fins given thickness the obvious way, leaving a 12 cm flat strip round the rim | 31 dBsm at 10° elevation, against 14 once the rim is sharp |
| `specular_aspects` | fin cant, which sets the elevation that strip-free surface mirrors at | 28 dB between a vertical fin and one canted 20° |

None of that is about the aircraft. It is about how the loop runs: state the
rule, build to it, predict in closed form, solve, and believe the disagreement.


### Command line

### From CAD

STEP and other B-rep files are tessellated first; a *planar* face tessellates
exactly, so a faceted airframe loses nothing in the conversion.

```
pip install gmsh                              # only needed for STEP import
python tools/step_to_mesh.py jet.step --out-dir meshes --scale 0.001
python examples/cad_signature.py meshes/jet.obj --freq 10GHz --axes zxy
```

`step_to_mesh.py` keeps each solid separate, welds the sliver triangles CAD
tessellation leaves behind, and reports whether each solid came out closed.
`cad_signature.py` reorients the model into echo1's frame and writes the
signature, a cut and a geometry render. `pipeline.py` runs the whole chain --
tessellate, reorient, symmetrize, cant, planform-sweep, sweep -- in one command.

Check a mesh before spending an hour sweeping it:

```
python tools/check_mesh.py model.obj --freq 10GHz
```

It separates what echo1 will refuse outright (non-manifold edges, zero-area
facets) from what it will accept but answer wrongly about (inward normals,
open boundaries, unmodelled dihedral corners) from what is merely worth
knowing (asymmetry, sub-wavelength detail), and names the modelling-package
operation that fixes each.

`check_mesh.py` asks whether the geometry is fit to solve. `audit.py` asks the
other question — whether it was *shaped* — using the two closed-form predictors
above, which run in under a second on a 25,000-facet import:

```
python tools/audit.py model.obj --freq 10GHz
python tools/audit.py model.obj --threat -90:-85 --threat -10:10
```

It reports the number of distinct edge-lobe azimuths, the aspect every flat
panel mirrors at — twice, per panel and bundled over merely *parallel* panels,
which brackets the answer from below and above — and how much metal points into
each elevation band you name. Negative elevation is a radar below, so
`--threat -90:-85` is the aspect every ground radar you overfly passes through.

```
echo1 shapes                                  # the built-in bodies
echo1 info  --shape faceted-delta             # facet and edge structure
echo1 sweep --shape faceted-delta --freq 10GHz --az 0:360:0.25 \
            --pol VV,HH --csv rcs.csv --polar polar.png --cut cut.png
echo1 sweep --mesh myjet.obj --freq 35GHz --az 0:360:0.5 --el 5
echo1 check                                   # compare against closed forms
```

`--no-ptd` drops the edge waves, which is the most instructive switch here: it
shows what a facet-only predictor would have told you, and why ECHO needed
Ufimtsev.

## How far to trust it

Every claim below is a test in `tests/`, run with `pytest`.

**Exact, to machine precision**

- Gordon's closed-form facet integral against brute-force quadrature of the same
  integral, over four decades of electrical size (agreement to ~1e-12).
- A flat plate at broadside against `4 pi A^2 / lambda^2`.
- A rectangular plate's off-broadside pattern against the analytic
  `cos^2 * sinc^2` physical-optics formula.
- Driving the edge kernel with the *PO* edge coefficients reproduces the exact
  Gordon integral for a rectangle, in amplitude and sign, at every aspect and
  polarisation. This identity is what fixes the equivalent-edge-current
  constant — it is measured, not asserted.
- Coplanar facets diffract exactly nothing, so tessellating a flat panel changes
  no answer.
- Physical optics produces no monostatic cross-polarised return, and
  `sigma_HV = sigma_VH` by reciprocity.

**Against an independent solver** (`validation/mom_strip.py`, a 2-D method of
moments — a direct numerical solution of the integral equation sharing no code
and no approximation with this package):

| aspect, off normal | MoM | physical optics | PO + PTD |
|---|---|---|---|
| 20–45° | 34.70 dBsm | −2.2 dB | **−1.3 dB** |
| 45–70° | 31.56 dBsm | −7.3 dB | **−3.5 dB** |
| 70–86° | 30.81 dBsm | −14.3 dB | **−4.9 dB** |

Physical optics on its own is wrong by 14 dB near grazing because it has no edge
waves. The Ufimtsev term recovers most of that, and the residual is first-order
PTD's known limit.

## What it does not do

Worth being blunt about, because RCS codes are easy to over-trust.

- **One bounce only.** No corner reflectors. `shapes.dihedral` and
  `shapes.trihedral` are in the library precisely as the counter-example: their
  real returns come from double and triple bounces this model does not carry,
  and `echo1.analytic` gives the closed forms to compare against.
- **One diffraction per edge.** No edge-to-edge or creeping waves, which is
  where the near-grazing error above comes from.
- **Convex and mildly re-entrant edges only.** A re-entrant corner is a
  multiple-bounce geometry. The right-angle case (wedge `n = 1/2`) is a
  dihedral retroreflector, and the single-diffraction coefficient has a pole
  exactly in its retroreflection direction, reaching 6,667 against `<= 1.0`
  across the whole convex range. Edges sharper than `n = 0.6` are dropped and
  the result reports how much that removed (`excluded_edges`); use
  `solver.dihedral_corners` to find out whether a body has corners whose
  unmodelled double bounce will make the real RCS *higher* than predicted.
- **Perfect conductors.** No radar-absorbing material, no coatings, no
  dielectrics — a real signature problem is half materials.
- **High frequency.** The body must be large compared with the wavelength; the
  CLI warns below about two wavelengths across.
- **Far field, single frequency**, and no propagation, clutter or receiver
  modelling.

## Layout

```
src/echo1/
  geometry.py   facets, and the wedge angle of every edge
  po.py         physical optics: Gordon's closed-form polygon integral
  ptd.py        Ufimtsev fringe coefficients and equivalent edge currents
  shadow.py     orientation and grid-accelerated occlusion
  solver.py     the sweep engine and the result object
  shapes.py     built-in bodies, and the two closed-form design predictors
  analytic.py   closed-form cross sections, for checking
  meshio.py     OBJ and STL
  plotting.py   polar signatures, cuts, and a look at the geometry
  cli.py        the echo1 command
tools/          STEP import, mesh preflight, and the design audit
docs/HISTORY.md the program this reconstructs
docs/THEORY.md  the derivations, including the ones done for this code
validation/     the independent method-of-moments comparison
examples/       worked examples that produce the figures
```

## References

- P. Ya. Ufimtsev, *Method of Edge Waves in the Physical Theory of Diffraction*,
  1962; US Air Force Foreign Technology Division translation, 1971.
- W. B. Gordon, "Far-field approximations to the Kirchhoff–Helmholtz
  representations of scattered fields", *IEEE Trans. Antennas Propag.*, 1975.
- E. F. Knott, J. F. Shaeffer, M. T. Tuley, *Radar Cross Section*, 2nd ed.
- R. F. Harrington, *Field Computation by Moment Methods* — the strip solution
  used in `validation/`.
- Lockheed Martin, ["1,000+ Stealth Aircraft"](https://www.lockheedmartin.com/en-us/news/features/2022/Now-Thats-a-Milestone-Worth-Celebrating-1000-Stealth-Air.html)
  and Air & Space Forces Magazine, ["How the Skunk Works Fielded Stealth"](https://www.airandspaceforces.com/article/1192stealth/).

MIT licensed.
