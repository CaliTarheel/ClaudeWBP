"""Audit a design the way a shop would, before spending anything on a sweep.

`check_mesh.py` asks whether the geometry is fit to solve.  This asks the other
question: whether it was *shaped*, and where it will be loud.  Both predictors
are closed form and run in under a second on a 25,000-facet import, so the
answer arrives while the drawing is still open.

    python tools/audit.py model.obj --freq 10GHz
    python tools/audit.py model.obj --threat -10:10 --threat -90:-60

What it reports:

  edges     the azimuths the edges throw into, weighted by each wedge's fringe
            coefficient.  A shaped aircraft has a handful; an unshaped one has
            a smear, and the count is the single most honest number here.
  panels    the aspect every flat panel mirrors at.  Reported twice: per panel,
            which is a lower bound, and bundled over parallel panels, which is
            the upper bound they reach if their phases line up.
  threat    how much metal points into each elevation band you name.

Elevation sign follows the solver: negative elevation is a radar BELOW the
aircraft, so `--threat -90:-60` is the look-up case.

What it does not do is integrate.  Both predictors rank scatterers one at a
time -- this edge, that panel -- and neither adds up what a radar actually
receives, which is every scatterer at once with its phase, its shadowing and
its neighbours.  So a band can get worse here and better in the sweep, because
one panel got louder while the rest got quieter.  Measured against a solve of
an imported airframe, this tool called a band 2 dB worse that the sweep found
3 dB better.  Where the two disagree the sweep is right; the audit's job is to
tell you where to point it.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from echo1 import shapes                                             # noqa: E402
from echo1.cli import _frequency                                     # noqa: E402
from echo1.meshio import load_mesh                                   # noqa: E402

#: The bands worth naming, taken from where a radar can actually stand.  At
#: 25,000 ft a ground set 150 km out is 3 deg below you, 50 km out is 9 deg,
#: 25 km out is 18 deg.  By 30 deg it is 15 km away and the engagement is
#: already decided; by 60 deg it is under 5 km and you are not being detected,
#: you are being overflown.
DEFAULT_THREATS = (
    (-20.0, -3.0, "ground radar at 25-150 km: the band detection happens in"),
    (-10.0, 10.0, "co-altitude: another aircraft, or a set on high ground"),
    (-60.0, -20.0, "inside 15 km, where it is already too late"),
    (-90.0, -60.0, "overhead, which almost nothing occupies"),
)


def lit_ground(area, el_deg, lam, altitude_m, earth_m=6371000.0):
    """Ground area a panel's specular lobe paints, in square metres.

    Peak alone ranks a design wrongly, because a bigger panel is louder *and*
    narrower and the two cancel exactly: ``peak x lobe = 4 pi A`` however the
    area is cut up.  What shaping changes is not how much return there is but
    what solid angle it is spread over -- and, against a ground threat, how
    much ground that solid angle lands on.  A lobe aimed at the horizon is
    smeared over half a county; the same lobe aimed at the nadir lands in a
    circle you could park a lorry in.

    The lobe is taken as a cone of half-angle ``lambda / 2 sqrt(A)``, mapped
    onto flat ground between its near and far edges and clipped at the radar
    horizon.  At the nadir the cone closes into a disc and the azimuth spread
    becomes the full circle, which the formula reaches on its own.
    """
    if el_deg >= 0.0:
        return float("inf")                  # points at or above the horizon
    theta = min(lam / (2.0 * np.sqrt(max(area, 1e-9))), np.pi / 2.0)
    horizon = np.sqrt(2.0 * earth_m * altitude_m)
    dep = np.radians(abs(el_deg))
    far, near = dep - theta, min(dep + theta, np.pi / 2.0)
    r_far = horizon if far <= 1e-9 else min(altitude_m / np.tan(far), horizon)
    r_near = 0.0 if near >= np.pi / 2.0 - 1e-9 else min(altitude_m / np.tan(near),
                                                        horizon)
    spread = min(2.0 * theta / max(np.cos(np.radians(el_deg)), 1e-9), 2.0 * np.pi)
    return 0.5 * spread * max(r_far ** 2 - r_near ** 2, 0.0)


def _ground(value):
    if not np.isfinite(value):
        return "  never lands"
    if value > 1e6:
        return f"{value / 1e6:9.2f} km2"
    return f"{value:9.0f} m2"


def audit(mesh, freq_hz, threats, top=8, min_length=0.3, bundle_deg=1.0,
          altitude_m=7620.0):
    lam = 299792458.0 / freq_hz
    e = mesh.edges
    group = shapes.panels(mesh)
    print(f"{mesh.name}: {mesh.n_faces} facets in "
          f"{len(np.unique(group))} flat panels, {mesh.total_area:.1f} m^2, "
          f"{mesh.max_dimension / lam:.0f} wavelengths across")

    dropped = e.wedge_n < 0.6
    if dropped.any():
        print(f"  {int(dropped.sum())} edges ({e.length[dropped].sum():.1f} m) are "
              "too re-entrant for single-bounce PTD and are left out below.  They "
              "are not quiet -- they are corners, and they are the reason the true "
              "RCS is above anything here")

    spikes = shapes.spike_azimuths(mesh, min_length=min_length, tol_deg=2.0)
    total = sum(w for _, w in spikes) or 1.0
    print(f"\nEDGES: {len(spikes)} distinct lobe azimuths, "
          f"{sum(w for _, w in spikes[:top]) / total * 100:.0f}% of the edge in "
          f"the top {top}")
    for az, w in spikes[:top]:
        print(f"     az {az:6.1f}   {w:7.2f} knife-metres  ({100 * w / total:4.1f}%)")

    one = shapes.specular_aspects(mesh, freq_hz=freq_hz)
    bun = shapes.specular_aspects(mesh, freq_hz=freq_hz, bundle_deg=bundle_deg)
    print(f"\nPANELS, ranked by the ground they light from "
          f"{altitude_m:.0f} m ({altitude_m / 0.3048:.0f} ft)")
    print(f"     {'area':>8} {'alone':>8} {'bundled':>8}  {'az':>7} {'el':>7}"
          f" {'lit ground':>13}")
    rows = []
    for az, el, a, pk in bun:
        alone = max((r[3] for r in one
                     if abs((r[0] - az + 180) % 360 - 180) < bundle_deg
                     and abs(r[1] - el) < bundle_deg), default=float("-inf"))
        lit = lit_ground(a, el, lam, altitude_m)
        rows.append((az, el, a, pk, alone, lit))
    # sort by how much return actually reaches the ground: the peak spread
    # over the patch it lands on.  A speck at the horizon beats the belly.
    catch = sorted((r for r in rows if np.isfinite(r[5]) and r[2] >= 0.01),
                   key=lambda r: -(10.0 ** (r[3] / 10.0) * r[5]))
    for az, el, a, pk, alone, lit in catch[:top]:
        print(f"     {a:8.2f} {alone:8.1f} {pk:8.1f}  {az:7.1f} {el:+7.1f}"
              f" {_ground(lit):>13}")

    print("\nTHREAT BANDS")
    for band in threats:
        lo, hi = band[0], band[1]
        label = band[2] if len(band) > 2 else ""
        got = [r for r in rows if lo <= r[1] <= hi and r[2] >= 0.01]
        area = sum(r[2] for r in got)
        worst = max(got, key=lambda r: r[3], default=None)
        # footprints overlap, so the band's own worst offender is the honest
        # number here, not a sum over every panel pointing into it
        lit = _ground(worst[5]) if worst else "           -"
        loud = f"{worst[3]:6.1f}" if worst else "     -"
        print(f"     el {lo:+6.1f} to {hi:+6.1f}: {area:7.2f} m^2 aimed in; "
              f"worst bundle {loud} dBsm over {lit}")
        print(f"                          {label}")


def _altitude(text):
    text = text.strip().lower()
    if text.endswith("ft"):
        return float(text[:-2]) * 0.3048
    return float(text[:-1] if text.endswith("m") else text)


def _band(text):
    lo, _, hi = text.partition(":")
    return float(lo), float(hi)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("meshes", nargs="+")
    ap.add_argument("--freq", default="10GHz")
    ap.add_argument("--threat", action="append", type=_band,
                    help="elevation band lo:hi, repeatable; "
                         "negative elevation is a radar below")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--min-length", type=float, default=0.3,
                    help="ignore edges shorter than this, in metres")
    ap.add_argument("--altitude", default="25000ft",
                    help="how high the aircraft is, for the lit-ground column")
    args = ap.parse_args(argv)
    for i, path in enumerate(args.meshes):
        if i:
            print()
        print("=" * 72)
        audit(load_mesh(path), _frequency(args.freq),
              args.threat or DEFAULT_THREATS, top=args.top,
              min_length=args.min_length, altitude_m=_altitude(args.altitude))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
