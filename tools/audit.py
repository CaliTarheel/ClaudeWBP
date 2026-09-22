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

#: The bands worth naming by default, and why each one is a band worth naming.
DEFAULT_THREATS = (
    (-10.0, 10.0, "co-altitude"),
    (-90.0, -85.0, "straight down -- every radar you overfly passes through it"),
    (-60.0, -20.0, "look-up, at a standoff"),
    (20.0, 60.0, "look-down, from a fighter above"),
)


def audit(mesh, freq_hz, threats, top=8, min_length=0.3, bundle_deg=1.0):
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
    print(f"\nPANELS: loudest bundle {bun[0][3]:.1f} dBsm "
          f"({bun[0][2]:.2f} m^2) at az {bun[0][0]:.1f}, el {bun[0][1]:+.1f}")
    print(f"     {'area':>8} {'alone':>8} {'bundled':>8}  {'az':>7} {'el':>7}")
    for az, el, a, pk in bun[:top]:
        alone = max((r[3] for r in one
                     if abs((r[0] - az + 180) % 360 - 180) < bundle_deg
                     and abs(r[1] - el) < bundle_deg), default=float("-inf"))
        print(f"     {a:8.2f} {alone:8.1f} {pk:8.1f}  {az:7.1f} {el:+7.1f}")

    print("\nTHREAT BANDS (area whose mirror points into the band)")
    print("     a wide band hides what matters: 40 m^2 aimed at one azimuth "
          "70 deg down is\n     a pencil you may never fly through, the same "
          "40 m^2 aimed at 90 deg down is\n     under you on every pass.")
    for band in threats:
        lo, hi = band[0], band[1]
        label = band[2] if len(band) > 2 else ""
        rows = [r for r in bun if lo <= r[1] <= hi]
        area = sum(r[2] for r in rows)
        loud = max((r[3] for r in rows), default=float("-inf"))
        print(f"     el {lo:+6.1f} to {hi:+6.1f}: {area:8.2f} m^2 "
              f"({100 * area / mesh.total_area:4.1f}% of skin), "
              f"loudest bundle {loud:6.1f} dBsm   {label}")


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
    args = ap.parse_args(argv)
    for i, path in enumerate(args.meshes):
        if i:
            print()
        print("=" * 72)
        audit(load_mesh(path), _frequency(args.freq),
              args.threat or DEFAULT_THREATS, top=args.top,
              min_length=args.min_length)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
