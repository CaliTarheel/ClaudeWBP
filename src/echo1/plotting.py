"""Plots: the polar RCS signature, cuts in dBsm, and a look at the geometry.

matplotlib is imported lazily so that the solver has no plotting dependency.
"""

from __future__ import annotations

import numpy as np

from .analytic import to_dbsm
from .geometry import Mesh
from .solver import RcsResult

__all__ = ["polar_rcs", "cartesian_rcs", "plot_mesh", "save_figure"]

_STYLE = dict(linewidth=1.1)


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:                              # pragma: no cover
        raise ImportError(
            "plotting needs matplotlib; install with: pip install 'echo1[plot]'"
        ) from exc
    return plt


def polar_rcs(result: RcsResult, pols=None, floor_db: float = -40.0,
              ceiling_db=None, title=None, ax=None):
    """Polar plot of RCS against azimuth, the classic signature diagram."""
    plt = _pyplot()
    pols = tuple(pols or result.channels)
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7.2, 7.2))

    top = ceiling_db
    for p in pols:
        db = np.clip(result.dbsm(p), floor_db, None)
        top = max(top, db.max()) if top is not None else db.max()
        th = np.radians(result.az_deg)
        ax.plot(np.append(th, th[:1]), np.append(db, db[:1]), label=p, **_STYLE)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(floor_db, np.ceil((top + 5) / 10) * 10)
    ax.set_rlabel_position(135)
    ax.grid(True, alpha=0.35)
    ax.set_title(title or f"{result.mesh_name} @ {result.freq_hz/1e9:.2f} GHz  (dBsm)",
                 pad=18)
    ax.legend(loc="lower right", bbox_to_anchor=(1.12, -0.04), frameon=False)
    return ax


def cartesian_rcs(result: RcsResult, pols=None, show_components: bool = False,
                  floor_db: float = -40.0, title=None, ax=None):
    """RCS against azimuth on a linear axis, optionally splitting PO from PTD."""
    plt = _pyplot()
    pols = tuple(pols or result.channels)
    if ax is None:
        _, ax = plt.subplots(figsize=(10.5, 5.0))

    for p in pols:
        ax.plot(result.az_deg, np.clip(result.dbsm(p), floor_db, None),
                label=p, **_STYLE)
        if show_components and p in result.sigma_po:
            ax.plot(result.az_deg,
                    np.clip(to_dbsm(result.sigma_po[p]), floor_db, None),
                    label=f"{p}: facets only (PO)", alpha=0.45, linewidth=0.9)
            ax.plot(result.az_deg,
                    np.clip(to_dbsm(result.sigma_ptd[p]), floor_db, None),
                    label=f"{p}: edges only (PTD)", alpha=0.45,
                    linewidth=0.9, linestyle="--")

    ax.set_xlabel("azimuth (deg)")
    ax.set_ylabel("RCS (dBsm)")
    ax.set_xlim(result.az_deg.min(), result.az_deg.max())
    ax.set_ylim(bottom=floor_db)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(title or f"{result.mesh_name} @ {result.freq_hz/1e9:.2f} GHz")
    return ax


def plot_mesh(mesh: Mesh, ax=None, title=None, elev: float = 22.0, azim: float = -58.0):
    """Shaded 3-D view of the facets, to check a model before sweeping it."""
    plt = _pyplot()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if ax is None:
        fig = plt.figure(figsize=(8.0, 6.0))
        ax = fig.add_subplot(111, projection="3d")

    tri = mesh.triangles
    shade = 0.35 + 0.65 * np.clip(
        mesh.face_normals @ np.array([0.38, 0.52, 0.76]), 0.0, 1.0
    )
    colours = np.stack([shade * 0.55, shade * 0.68, shade * 0.85, np.ones_like(shade)], 1)
    coll = Poly3DCollection(tri, facecolors=colours, edgecolors="k", linewidths=0.35)
    ax.add_collection3d(coll)

    lo, hi = mesh.extent
    mid, span = (lo + hi) / 2.0, float(np.max(hi - lo)) / 2.0 or 1.0
    for setter, m in ((ax.set_xlim, mid[0]), (ax.set_ylim, mid[1]), (ax.set_zlim, mid[2])):
        setter(m - span, m + span)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title or f"{mesh.name}: {mesh.n_faces} facets, {len(mesh.edges)} edges")
    return ax


def save_figure(ax, path, dpi: int = 150) -> None:
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    _pyplot().close(fig)
