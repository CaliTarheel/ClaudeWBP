"""The two lessons ECHO taught, as pictures.

Produces, in ``examples/output``:

* the geometry of both aircraft shapes;
* what one badly placed vertex costs, by un-raking the diamond's aft transom;
* the hopeless diamond's signature with and without the edge-wave term, which
  is the case for having bothered with Ufimtsev at all;
* the aligned faceted delta against the unaligned diamond, which is the case
  for planform alignment.

Run:  python examples/signatures.py
"""

from __future__ import annotations

import os

import numpy as np

from echo1 import analytic, plotting, shapes
from echo1.solver import monostatic_rcs

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
FREQ = 10e9
AZ = np.arange(0.0, 360.0, 0.25)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    diamond = shapes.hopeless_diamond()
    delta = shapes.faceted_delta()

    for body in (diamond, delta):
        ax = plotting.plot_mesh(body)
        plotting.save_figure(ax, os.path.join(OUT, f"{body.name}-geometry.png"))

    print(f"{'':22s} {'peak':>10s} {'median':>10s} {'mean':>10s}")
    results = {}
    for body in (diamond, delta):
        r = monostatic_rcs(body, FREQ, AZ, el_deg=0.0, pols=("VV", "HH"))
        results[body.name] = r
        s = r.sigma["VV"]
        print(f"{body.name:22s} {analytic.to_dbsm(s.max()):9.1f}dB "
              f"{analytic.to_dbsm(np.median(s)):9.1f}dB "
              f"{analytic.to_dbsm(s.mean()):9.1f}dB")

        ax = plotting.polar_rcs(r, floor_db=-50.0)
        plotting.save_figure(ax, os.path.join(OUT, f"{body.name}-polar.png"))

    # One vertex: leaving the aft transom flat makes it a mirror pointed down
    # the tail.  Raking the ridge point leans it out of the horizontal plane.
    flat_tail = shapes.hopeless_diamond(transom_rake=0.0)
    flat = monostatic_rcs(flat_tail, FREQ, AZ, pols=("VV",))
    raked = results[diamond.name]
    print(f"\none vertex, the aft transom:")
    print(f"  flat aft face   peak {analytic.to_dbsm(flat.sigma['VV'].max()):6.1f} dBsm "
          f"at az {flat.peak('VV')[0]:.1f}, median "
          f"{analytic.to_dbsm(np.median(flat.sigma['VV'])):6.1f} dBsm")
    print(f"  ridge raked     peak {analytic.to_dbsm(raked.sigma['VV'].max()):6.1f} dBsm "
          f"at az {raked.peak('VV')[0]:.1f}, median "
          f"{analytic.to_dbsm(np.median(raked.sigma['VV'])):6.1f} dBsm")
    print(f"  -> {analytic.to_dbsm(flat.sigma['VV'].max()) - analytic.to_dbsm(raked.sigma['VV'].max()):.0f} dB "
          f"off the peak, for moving one point forward")

    ax = plotting.cartesian_rcs(raked, pols=("VV",), floor_db=-60.0,
                                title="Hopeless diamond: one vertex, 35 dB")
    ax.plot(flat.az_deg, np.clip(flat.dbsm("VV"), -60.0, None),
            label="VV: flat aft transom", linewidth=0.9, alpha=0.6)
    ax.legend(fontsize=8, ncol=2)
    plotting.save_figure(ax, os.path.join(OUT, "one-vertex.png"))

    # What a facet-only predictor would have said.
    with_edges = results[diamond.name]
    facets_only = monostatic_rcs(diamond, FREQ, AZ, pols=("VV",), use_ptd=False)
    ax = plotting.cartesian_rcs(with_edges, pols=("VV",), floor_db=-60.0,
                                title="Hopeless diamond: what the edge waves add")
    ax.plot(facets_only.az_deg, np.clip(facets_only.dbsm("VV"), -60.0, None),
            label="VV: facets only (no PTD)", linewidth=0.9, alpha=0.6)
    ax.legend(fontsize=8, ncol=2)
    plotting.save_figure(ax, os.path.join(OUT, "edge-waves-matter.png"))

    med = lambda r: analytic.to_dbsm(np.median(r.sigma["VV"]))
    print("\nwhat the edge waves are worth, on the diamond:")
    print(f"  median RCS with edge waves     {med(with_edges):7.1f} dBsm")
    print(f"  median RCS from facets alone   {med(facets_only):7.1f} dBsm"
          f"   ({med(facets_only) - med(with_edges):+.1f} dB of wishful thinking)")
    quiet = lambda r, t: float(np.mean(r.sigma["VV"] < analytic.from_dbsm(t)) * 100)
    for thresh in (-20.0, -40.0):
        print(f"  aspects below {thresh:.0f} dBsm: "
              f"{quiet(with_edges, thresh):5.1f}% real, "
              f"{quiet(facets_only, thresh):5.1f}% claimed by facets alone")

    alignment_demo()


def alignment_demo() -> None:
    """Planform alignment, isolated on two flat plates of equal area.

    An aircraft has ridge lines, keels and fin edges that muddy the picture, so
    this strips the question down: same area, same material, only the outline
    differs.  One outline uses two edge directions, the other four.
    """
    diamond_outline = [(0.8, 0.0), (0.0, 0.5), (-0.8, 0.0), (0.0, -0.5)]
    irregular = np.array([(0.86, -0.18), (0.12, 0.62), (-0.78, 0.10), (-0.10, -0.66)])

    aligned = shapes.polygon_plate(diamond_outline, "aligned-diamond")
    scale = np.sqrt(aligned.total_area
                    / shapes.polygon_plate(irregular).total_area)
    unaligned = shapes.polygon_plate(irregular * scale, "irregular-outline")

    az = np.arange(0.0, 360.0, 0.05)
    print("\nplanform alignment, two flat plates of equal area "
          f"({aligned.total_area:.3f} m^2), 25 deg off the plate plane:")
    ax = None
    for body in (aligned, unaligned):
        rim = body.edges.n_adjacent == 1
        d = body.edges.e_hat[rim]
        dirs = np.unique(np.round(
            np.degrees(np.arctan2(np.abs(d[:, 1]), np.abs(d[:, 0]))), 2))
        r = monostatic_rcs(body, FREQ, az, 25.0, pols=("HH",))
        db = r.dbsm("HH")
        hot = db > db.max() - 15.0
        lobes = int(np.sum(hot.astype(int) - np.roll(hot, 1).astype(int) == 1))
        print(f"  {body.name:20s} {len(dirs)} edge direction(s) -> {lobes:2d} lobes "
              f"within 15 dB of peak, covering {100*hot.mean():.2f}% of azimuth; "
              f"median {np.median(db):.1f} dBsm")
        ax = plotting.cartesian_rcs(r, pols=("HH",), floor_db=-70.0, ax=ax,
                                    title="Planform alignment: equal area, "
                                          "different outlines")
        ax.lines[-1].set_label(f"{body.name} ({len(dirs)} edge directions)")
    ax.legend(fontsize=8)
    plotting.save_figure(ax, os.path.join(OUT, "planform-alignment.png"))
    print("  -> the same edge length, gathered into fewer directions, makes "
          "fewer and narrower spikes")

    print(f"\nwrote figures to {OUT}")


if __name__ == "__main__":
    main()
