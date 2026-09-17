"""Command line front end.

    echo1 shapes
    echo1 info   --shape hopeless-diamond
    echo1 sweep  --shape hopeless-diamond --freq 10e9 --az 0:360:0.25 \\
                 --pol VV,HH --csv rcs.csv --polar rcs.png
    echo1 check
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import analytic, shapes
from .geometry import Mesh
from .meshio import load_mesh, save_mesh
from .solver import monostatic_rcs

_EPILOG = """\
angles are given as START:STOP:STEP in degrees (STOP excluded), or as a single
number.  frequency accepts plain hertz or a suffix, e.g. 10e9, 10GHz, 35ghz.
"""


def _angles(spec: str) -> np.ndarray:
    parts = str(spec).split(":")
    try:
        if len(parts) == 1:
            return np.array([float(parts[0])])
        if len(parts) == 3:
            start, stop, step = (float(x) for x in parts)
            if step <= 0:
                raise ValueError("step must be positive")
            return np.arange(start, stop, step)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"bad angle spec {spec!r}: {exc}") from exc
    raise argparse.ArgumentTypeError(
        f"bad angle spec {spec!r}; use START:STOP:STEP or a single number"
    )


def _frequency(spec: str) -> float:
    text = str(spec).strip().lower().replace(" ", "")
    for suffix, scale in (("thz", 1e12), ("ghz", 1e9), ("mhz", 1e6), ("khz", 1e3), ("hz", 1.0)):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            return float(text) * scale
    return float(text)


def _build_mesh(args) -> Mesh:
    if args.mesh:
        mesh = load_mesh(args.mesh, two_sided=args.two_sided)
    else:
        if args.shape not in shapes.BUILTIN:
            raise SystemExit(
                f"unknown shape {args.shape!r}; try 'echo1 shapes' for the list"
            )
        mesh = shapes.BUILTIN[args.shape]()
    if args.scale != 1.0:
        mesh = mesh.scaled(args.scale)
    return mesh


def _add_geometry_args(p) -> None:
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--shape", help="a built-in shape (see 'echo1 shapes')")
    src.add_argument("--mesh", help="path to an .obj or .stl model")
    p.add_argument("--two-sided", action="store_true",
                   help="treat the model as an open sheet rather than a closed body")
    p.add_argument("--scale", type=float, default=1.0, help="scale the model")


def _cmd_shapes(_args) -> int:
    print("built-in shapes:")
    for name, fn in shapes.BUILTIN.items():
        first = (fn.__doc__ or "").strip().splitlines()[0]
        print(f"  {name:18s} {first}")
    return 0


def _cmd_info(args) -> int:
    mesh = _build_mesh(args)
    lo, hi = mesh.extent
    n = mesh.edges.wedge_n
    print(f"{mesh.name}")
    print(f"  facets          {mesh.n_faces}")
    print(f"  edges           {len(mesh.edges)}  "
          f"({int(np.sum(n > 1.0 + 1e-9))} diffracting, "
          f"{int(np.sum(np.abs(n - 1.0) <= 1e-9))} coplanar)")
    print(f"  surface area    {mesh.total_area:.4f} m^2")
    print(f"  closed body     {mesh.is_closed}")
    if mesh.is_closed:
        print(f"  volume          {mesh.volume:.4f} m^3")
    print(f"  bounding box    x {lo[0]:+.3f}..{hi[0]:+.3f}   "
          f"y {lo[1]:+.3f}..{hi[1]:+.3f}   z {lo[2]:+.3f}..{hi[2]:+.3f}")
    if args.out:
        save_mesh(mesh, args.out)
        print(f"  written to      {args.out}")
    return 0


def _cmd_sweep(args) -> int:
    mesh = _build_mesh(args)
    freq = _frequency(args.freq)
    az, el = _angles(args.az), _angles(args.el)
    if len(az) > 1 and len(el) > 1 and len(az) != len(el):
        raise SystemExit("--az and --el must have equal length, or one must be a single value")
    pols = tuple(p.strip().upper() for p in args.pol.split(","))

    lam = analytic.wavelength(freq)
    if mesh.max_dimension < 2.0 * lam:
        print(f"warning: the model is only {mesh.max_dimension/lam:.1f} wavelengths across; "
              "high-frequency asymptotics are unreliable below about 2.", file=sys.stderr)

    result = monostatic_rcs(
        mesh, freq, az, el, pols=pols,
        use_po=not args.no_po, use_ptd=not args.no_ptd,
        occlusion=not args.no_occlusion, chunk=args.chunk,
    )
    for p in pols:
        print(result.summary(p))

    if args.csv:
        result.to_csv(args.csv)
        print(f"wrote {args.csv}")
    if args.polar or args.cut:
        from . import plotting
        if args.polar:
            plotting.save_figure(plotting.polar_rcs(result), args.polar)
            print(f"wrote {args.polar}")
        if args.cut:
            plotting.save_figure(
                plotting.cartesian_rcs(result, show_components=True), args.cut)
            print(f"wrote {args.cut}")
    return 0


def _cmd_check(_args) -> int:
    """Compare against the closed forms that have known answers."""
    lam, ok = 0.03, True
    freq = analytic.C0 / lam
    rows = []

    a = 0.30
    plate = shapes.plate(a, a)
    got = monostatic_rcs(plate, freq, [0.0], [90.0], pols=("VV",), use_ptd=False)
    want = analytic.flat_plate_broadside(a * a, lam)
    rows.append(("flat plate, broadside, PO", got.sigma["VV"][0], want))

    r = 0.5
    sph = shapes.sphere(r, subdivisions=4)
    got = monostatic_rcs(sph, freq, [0.0], [0.0], pols=("VV",), use_ptd=False,
                         occlusion=False)
    rows.append(("sphere, optical limit", got.sigma["VV"][0], analytic.sphere_optical(r)))

    print(f"{'case':34s} {'echo1':>12s} {'closed form':>12s} {'delta':>9s}")
    for name, g, w in rows:
        d = analytic.to_dbsm(g) - analytic.to_dbsm(w)
        flag = "" if abs(d) < 1.0 else "   <-- check"
        ok &= abs(d) < 1.0
        print(f"{name:34s} {analytic.to_dbsm(g):9.2f} dB {analytic.to_dbsm(w):9.2f} dB "
              f"{d:+8.2f}{flag}")
    print("\nfor the full validation (including an independent method-of-moments "
          "comparison) run: python validation/mom_strip.py")
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="echo1",
        description="Faceted radar cross section by physical optics + Ufimtsev PTD.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("shapes", help="list the built-in shapes").set_defaults(func=_cmd_shapes)

    p_info = sub.add_parser("info", help="report a model's facet and edge structure")
    _add_geometry_args(p_info)
    p_info.add_argument("--out", help="also write the model to this .obj/.stl path")
    p_info.set_defaults(func=_cmd_info)

    p_sweep = sub.add_parser("sweep", help="compute monostatic RCS over a sweep")
    _add_geometry_args(p_sweep)
    p_sweep.add_argument("--freq", default="10e9", help="radar frequency (default 10e9)")
    p_sweep.add_argument("--az", default="0:360:0.5", help="azimuth sweep (default 0:360:0.5)")
    p_sweep.add_argument("--el", default="0", help="elevation (default 0)")
    p_sweep.add_argument("--pol", default="VV,HH", help="channels, comma separated")
    p_sweep.add_argument("--csv", help="write results to this CSV path")
    p_sweep.add_argument("--polar", help="write a polar signature plot here")
    p_sweep.add_argument("--cut", help="write a linear azimuth cut here")
    p_sweep.add_argument("--no-po", action="store_true", help="drop the facet term")
    p_sweep.add_argument("--no-ptd", action="store_true", help="drop the edge-wave term")
    p_sweep.add_argument("--no-occlusion", action="store_true",
                         help="skip ray-traced hiding (exact for convex bodies, faster)")
    p_sweep.add_argument("--chunk", type=int, default=64, help="look angles per batch")
    p_sweep.set_defaults(func=_cmd_sweep)

    sub.add_parser("check", help="compare against closed-form results").set_defaults(
        func=_cmd_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
