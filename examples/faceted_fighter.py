"""Designing a faceted aircraft the way ECHO 1 let Lockheed design one.

Not a replica of anything.  The point is the loop: state a shaping rule, build
geometry that obeys it, *predict in closed form* where the returns will go,
then solve and see whether they went there.  When the prediction and the solve
disagree the geometry is wrong, and that is the whole value of having a code --
you find out at the drawing stage instead of at the range.

Produces, in ``examples/output``:

* the aircraft, and its signature at the horizon;
* the same aircraft with conventionally swept fins, which is the rule being
  broken and what it costs;
* an azimuth-elevation map of where the facet specular flashes actually sit.

Run:  python examples/faceted_fighter.py
"""

from __future__ import annotations

import os

import numpy as np

from echo1 import plotting, shapes
from echo1.plotting import to_dbsm
from echo1.solver import monostatic_rcs

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
FREQ = 10e9
AZ = np.arange(0.0, 360.0, 0.02)          # fine: a 11 m edge flashes over 0.15 deg
SECTORS = (("nose +/-30", 330.0, 30.0), ("beam +/-30", 60.0, 120.0),
           ("tail +/-30", 150.0, 210.0))


def sector(az, db, lo, hi):
    m = (az >= lo) | (az <= hi) if lo > hi else (az >= lo) & (az <= hi)
    return db[m].max(), 10.0 * np.log10(np.mean(10.0 ** (db[m] / 10.0)))


def nearest(az, db, target, window=2.0):
    """The highest return within ``window`` of a predicted azimuth."""
    d = np.abs((az - target + 180.0) % 360.0 - 180.0)
    m = d <= window
    i = np.argmax(np.where(m, db, -np.inf))
    return float(az[i]), float(db[i])


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    body = shapes.faceted_fighter()
    loose = shapes.faceted_fighter(fin_align=False, name="conventional-fins")

    e = body.edges
    print(f"{body.name}: {body.n_faces} facets, {len(e.length)} edges, "
          f"{body.volume:.1f} m^3, {body.max_dimension:.1f} m across")
    print(f"  wedge angles n = {e.wedge_n.min():.2f} to {e.wedge_n.max():.2f}, "
          f"{int((e.wedge_n < 1.0).sum())} re-entrant")

    print("\n1. WHERE THE EDGES WILL THROW  (closed form, no solve)")
    pred = shapes.spike_azimuths(body)
    for az, w in pred[:6]:
        print(f"     az {az:6.1f}   {w:5.2f} knife-edge-equivalent metres")

    print("\n2. WHERE THE FACETS WILL MIRROR  (closed form, no solve)")
    spec = shapes.specular_aspects(body)
    print(f"     loudest: {spec[0][3]:.0f} dBsm at az {spec[0][0]:.1f}, "
          f"el {spec[0][1]:+.1f}")
    flat = min(spec, key=lambda r: abs(r[1]))
    print(f"     closest to the horizon: {flat[3]:.0f} dBsm at el {flat[1]:+.1f} "
          f"({flat[2]:.2f} m^2)")
    print(f"     no facet mirrors within {min(abs(r[1]) for r in spec):.1f} deg "
          "of the horizon")

    print("\n3. THE SOLVE, AND WHETHER IT AGREES")
    results = {}
    for m in (body, loose):
        r = monostatic_rcs(m, FREQ, AZ, el_deg=0.0, pols=("VV", "HH"))
        results[m.name] = r
    db = to_dbsm(results[body.name].sigma["VV"])
    for az, w in pred[:6]:
        a, v = nearest(AZ, db, az)
        print(f"     predicted az {az:6.1f}  ->  measured {v:6.2f} dBsm "
              f"at az {a:6.2f}  ({abs(a - az):.2f} deg out)")

    print("\n4. WHAT BREAKING THE RULE COSTS")
    head = "".join(f"{s[0]:>24}" for s in SECTORS)
    print(f"     {'':22s} {'peak':>7}{head}")
    print(f"     {'':22s} {'':>7}" + "".join(f"{'peak':>12}{'mean':>12}"
                                             for _ in SECTORS))
    for m in (body, loose):
        d = to_dbsm(results[m.name].sigma["VV"])
        cells = "".join("%12.1f%12.1f" % sector(AZ, d, lo, hi)
                        for _, lo, hi in SECTORS)
        print(f"     {m.name:22s} {d.max():7.1f}{cells}")
    print("     (both aircraft are otherwise identical; only the fin sweep differs)")

    print("\n5. THE ELEVATION THE SPECULARS LIVE AT")
    els = np.arange(-60.0, 60.1, 1.0)
    coarse = np.arange(0.0, 360.0, 0.5)
    grid = np.array([to_dbsm(monostatic_rcs(body, FREQ, coarse, el_deg=el,
                                            pols=("VV",)).sigma["VV"])
                     for el in els])
    loud = np.abs(els[grid.max(axis=1) > 25.0])
    print(f"     nothing above 25 dBsm inside +/-{loud.min():.0f} deg of the horizon")
    print(f"     worst {grid.max():.1f} dBsm at el "
          f"{els[np.unravel_index(grid.argmax(), grid.shape)[0]]:+.0f}")
    _map(coarse, els, grid, os.path.join(OUT, "faceted-fighter-elevation.png"))

    print("\n6. WHAT THE FIN CANT BUYS")
    print("     a fin is a mirror; canting it does not shrink the mirror,")
    print("     it aims it somewhere no radar stands.")
    print(f"     {'cant':>5} {'nearest specular':>17} {'horizon peak':>14} "
          f"{'nose mean':>11}")
    for cant in (0.0, 20.0, 35.0, 50.0):
        m = shapes.faceted_fighter(fin_cant_deg=cant)
        near = min(abs(el) for _, el, _, _ in shapes.specular_aspects(m))
        d = to_dbsm(monostatic_rcs(m, FREQ, coarse, pols=("VV",)).sigma["VV"])
        nose = 10.0 * np.log10(np.mean(10.0 ** (
            d[(coarse <= 30.0) | (coarse >= 330.0)] / 10.0)))
        print(f"     {cant:5.0f} {near:14.1f} deg {d.max():11.1f} dB "
              f"{nose:8.1f} dB")

    print(f"\nwrote geometry, polar and elevation plots to {OUT}")


def _map(az, el, grid, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    im = ax.pcolormesh(az, el, np.clip(grid, -40.0, 40.0), cmap="magma",
                       shading="nearest")
    ax.set_xlabel("azimuth (deg)")
    ax.set_ylabel("elevation (deg)")
    ax.set_title("faceted-fighter, VV at 10 GHz (dBsm)")
    ax.set_xticks(np.arange(0, 361, 45))
    fig.colorbar(im, ax=ax, label="dBsm")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
