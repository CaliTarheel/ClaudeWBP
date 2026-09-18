"""The sweep engine: point a radar at a faceted body and read off its RCS.

This is the part that corresponds to what ECHO 1 actually *did* -- take a shape
built from flat panels, an illumination geometry and a wavelength, and return
the radar cross section.  The scattering itself lives in :mod:`echo1.po`
(specular returns from the panels) and :mod:`echo1.ptd` (edge waves), and the
visibility bookkeeping in :mod:`echo1.shadow`.

Angle convention
----------------
``az`` is measured in the x-y plane from ``+x`` towards ``+y``; ``el`` is
measured up from the x-y plane.  The unit vector towards the radar is

    r = (cos el cos az, cos el sin az, sin el)

so for an aircraft modelled nose-along-``+x`` and wings along ``+/-y``,
``az = 0`` is nose-on and ``az = 90`` is the port beam.

Polarisation
------------
``V`` is the vertical unit vector at the radar and ``H`` the horizontal one.
Channel names are ``"<transmit><receive>"``.  For monostatic backscatter
reciprocity makes ``HV`` and ``VH`` equal, so the ordering only matters
bistatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from .analytic import C0, to_dbsm
from .geometry import Mesh, normalize
from .po import po_amplitude
from .ptd import _MIN_WEDGE_N, diffracting, ptd_amplitude
from .shadow import edge_illumination, facet_illumination

__all__ = ["RcsResult", "look_vectors", "monostatic_rcs", "bistatic_rcs"]

_POLS = ("VV", "HH", "HV", "VH")


def look_vectors(az_deg, el_deg):
    """Return ``(r_hat, v_hat, h_hat)`` for the given azimuth/elevation, in degrees."""
    az = np.radians(np.atleast_1d(np.asarray(az_deg, float)))
    el = np.radians(np.atleast_1d(np.asarray(el_deg, float)))
    az, el = np.broadcast_arrays(az, el)
    r = np.stack(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1
    )
    z = np.array([0.0, 0.0, 1.0])
    h = np.cross(z, r)
    tiny = np.linalg.norm(h, axis=-1) < 1e-9
    if np.any(tiny):                     # looking straight up or down
        h[tiny] = np.array([0.0, 1.0, 0.0])
    h = normalize(h)
    v = normalize(np.cross(r, h))
    return r, v, h


@dataclass
class RcsResult:
    """Radar cross section over a sweep, with the pieces it was built from."""

    az_deg: np.ndarray
    el_deg: np.ndarray
    freq_hz: float
    sigma: Dict[str, np.ndarray]              # m^2, by polarisation channel
    sigma_po: Dict[str, np.ndarray] = field(default_factory=dict)
    sigma_ptd: Dict[str, np.ndarray] = field(default_factory=dict)
    mesh_name: str = "mesh"
    bistatic: bool = False
    #: ``(count, length)`` of edges dropped for having too sharp a re-entrant
    #: wedge for single-bounce PTD.  Non-zero means some geometry was ignored.
    excluded_edges: tuple = (0, 0.0)

    @property
    def wavelength(self) -> float:
        return C0 / self.freq_hz

    @property
    def channels(self):
        return tuple(self.sigma.keys())

    def dbsm(self, pol: str = "VV") -> np.ndarray:
        return to_dbsm(self.sigma[pol])

    def peak(self, pol: str = "VV"):
        """``(az, el, sigma)`` of the strongest return in a channel."""
        i = int(np.argmax(self.sigma[pol]))
        return float(self.az_deg[i]), float(self.el_deg[i]), float(self.sigma[pol][i])

    def median(self, pol: str = "VV") -> float:
        return float(np.median(self.sigma[pol]))

    def summary(self, pol: str = "VV") -> str:
        az, el, pk = self.peak(pol)
        s = self.sigma[pol]
        return (
            f"{self.mesh_name}: {pol} at {self.freq_hz/1e9:.3f} GHz "
            f"(lambda = {self.wavelength*100:.2f} cm), {len(s)} aspects\n"
            f"  peak    {to_dbsm(pk):8.2f} dBsm at az {az:.2f} deg, el {el:.2f} deg\n"
            f"  median  {to_dbsm(np.median(s)):8.2f} dBsm\n"
            f"  mean    {to_dbsm(np.mean(s)):8.2f} dBsm (power-averaged)\n"
            f"  minimum {to_dbsm(np.min(s)):8.2f} dBsm"
        )

    def to_csv(self, path) -> None:
        cols = ["az_deg", "el_deg"] + [f"{p}_dbsm" for p in self.channels]
        data = [self.az_deg, self.el_deg] + [self.dbsm(p) for p in self.channels]
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"# echo1 RCS: {self.mesh_name}, {self.freq_hz:.6g} Hz\n")
            fh.write(",".join(cols) + "\n")
            for row in zip(*data):
                fh.write(",".join(f"{x:.6g}" for x in row) + "\n")


def _channel_vectors(pol, v_i, h_i, v_s, h_s):
    tx, rx = pol[0], pol[1]
    e_inc = v_i if tx == "V" else h_i
    e_rec = v_s if rx == "V" else h_s
    return e_inc, e_rec


def _sweep(
    mesh: Mesh,
    r_inc: np.ndarray,
    r_sca: np.ndarray,
    freq_hz: float,
    pols: Sequence[str],
    use_po: bool,
    use_ptd: bool,
    occlusion: bool,
    chunk: int,
    min_wedge_n: float,
):
    """Shared core for monostatic and bistatic sweeps."""
    k = 2.0 * np.pi * freq_hz / C0
    n = len(r_inc)
    sig = {p: np.zeros(n) for p in pols}
    sig_po = {p: np.zeros(n) for p in pols}
    sig_pt = {p: np.zeros(n) for p in pols}

    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        ri, rs = r_inc[sl], r_sca[sl]
        i_hat = -ri
        s_hat = rs

        # Polarisation frames at the transmitter and at the receiver.
        _, v_i, h_i = _frame(ri)
        _, v_s, h_s = _frame(rs)

        lit = facet_illumination(mesh, ri, occlusion) if use_po else None
        if use_ptd:
            act_i = edge_illumination(mesh, ri, occlusion)
            act = act_i if np.array_equal(ri, rs) else (
                act_i & edge_illumination(mesh, rs, occlusion)
            )

        for p in pols:
            e_inc, e_rec = _channel_vectors(p, v_i, h_i, v_s, h_s)
            amp = np.zeros(len(ri), dtype=complex)
            if use_po:
                a_po = po_amplitude(mesh, i_hat, s_hat, e_inc, e_rec, k, lit).sum(axis=1)
                sig_po[p][sl] = np.abs(a_po) ** 2
                amp = amp + a_po
            if use_ptd and len(mesh.edges):
                a_pt = ptd_amplitude(
                    mesh.edges, i_hat, s_hat, e_inc, e_rec, k, act,
                    min_wedge_n=min_wedge_n,
                ).sum(axis=1)
                sig_pt[p][sl] = np.abs(a_pt) ** 2
                amp = amp + a_pt
            sig[p][sl] = np.abs(amp) ** 2
    return sig, sig_po, sig_pt


def _edge_census(mesh: Mesh, min_wedge_n: float):
    """How much edge length was dropped as too re-entrant to model."""
    edges = mesh.edges
    dropped = (edges.wedge_n < min_wedge_n)
    return int(dropped.sum()), float(edges.length[dropped].sum())


def _frame(r):
    z = np.array([0.0, 0.0, 1.0])
    h = np.cross(z, r)
    tiny = np.linalg.norm(h, axis=-1) < 1e-9
    if np.any(tiny):
        h[tiny] = np.array([0.0, 1.0, 0.0])
    h = normalize(h)
    return r, normalize(np.cross(r, h)), h


def monostatic_rcs(
    mesh: Mesh,
    freq_hz: float,
    az_deg,
    el_deg=0.0,
    pols: Iterable[str] = ("VV", "HH"),
    use_po: bool = True,
    use_ptd: bool = True,
    occlusion: bool = True,
    chunk: int = 64,
    min_wedge_n: float = _MIN_WEDGE_N,
) -> RcsResult:
    """Backscatter RCS of ``mesh`` over a sweep of look directions.

    Parameters
    ----------
    mesh : Mesh
        The faceted body.
    freq_hz : float
        Radar frequency.
    az_deg, el_deg : array-like
        Look angles; broadcast against each other, so a fixed elevation and a
        swept azimuth is the common case.
    pols : iterable of str
        Any of ``"VV"``, ``"HH"``, ``"HV"``, ``"VH"``.
    use_po, use_ptd : bool
        Enable the physical-optics and edge-wave contributions.  Turning PTD
        off reproduces what a facet-only predictor would have said, which is a
        useful thing to plot against.
    occlusion : bool
        Ray-trace hidden facets and edges.  Safe to disable for convex bodies,
        where facet orientation alone is exact, and much faster.
    chunk : int
        Look directions processed per batch (memory/speed trade-off).
    min_wedge_n : float
        Smallest exterior wedge angle, in units of pi, that is allowed to
        diffract.  See :data:`echo1.ptd._MIN_WEDGE_N`; the returned result
        records what this excluded in ``excluded_edges``.
    """
    pols = tuple(pols)
    bad = [p for p in pols if p not in _POLS]
    if bad:
        raise ValueError(f"unknown polarisation channel(s) {bad}; choose from {_POLS}")
    if freq_hz <= 0:
        raise ValueError("freq_hz must be positive")

    r, _, _ = look_vectors(az_deg, el_deg)
    az = np.broadcast_to(np.atleast_1d(np.asarray(az_deg, float)), (len(r),)).copy()
    el = np.broadcast_to(np.atleast_1d(np.asarray(el_deg, float)), (len(r),)).copy()

    sig, sig_po, sig_pt = _sweep(
        mesh, r, r, freq_hz, pols, use_po, use_ptd, occlusion, chunk, min_wedge_n
    )
    result = RcsResult(az, el, float(freq_hz), sig, sig_po, sig_pt, mesh.name, False)
    result.excluded_edges = _edge_census(mesh, min_wedge_n)
    return result


def bistatic_rcs(
    mesh: Mesh,
    freq_hz: float,
    inc_az_deg,
    inc_el_deg,
    sca_az_deg,
    sca_el_deg,
    pols: Iterable[str] = ("VV", "HH"),
    use_po: bool = True,
    use_ptd: bool = True,
    occlusion: bool = True,
    chunk: int = 64,
    min_wedge_n: float = _MIN_WEDGE_N,
) -> RcsResult:
    """Bistatic RCS for separate transmit and receive directions.

    Note that the edge-wave term is exact only on each edge's Keller cone,
    which backscatter always satisfies but a general bistatic geometry does
    not; well off the cone the PTD contribution should be read as indicative.
    """
    pols = tuple(pols)
    r_i, _, _ = look_vectors(inc_az_deg, inc_el_deg)
    r_s, _, _ = look_vectors(sca_az_deg, sca_el_deg)
    if len(r_i) != len(r_s):
        r_i, r_s = np.broadcast_arrays(r_i, r_s)
    sig, sig_po, sig_pt = _sweep(
        mesh, r_i, r_s, freq_hz, pols, use_po, use_ptd, occlusion, chunk, min_wedge_n
    )
    az = np.broadcast_to(np.atleast_1d(np.asarray(sca_az_deg, float)), (len(r_i),)).copy()
    el = np.broadcast_to(np.atleast_1d(np.asarray(sca_el_deg, float)), (len(r_i),)).copy()
    return RcsResult(az, el, float(freq_hz), sig, sig_po, sig_pt, mesh.name, True)
