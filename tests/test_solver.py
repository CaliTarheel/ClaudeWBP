"""The sweep engine: conventions, bookkeeping and self-consistency."""

import numpy as np
import pytest

from conftest import FREQ, rect_plate

from echo1 import analytic, shapes
from echo1.solver import bistatic_rcs, look_vectors, monostatic_rcs


def test_look_vector_conventions():
    r, v, h = look_vectors([0.0, 90.0, 0.0], [0.0, 0.0, 90.0])
    assert np.allclose(r[0], [1, 0, 0])
    assert np.allclose(r[1], [0, 1, 0])
    assert np.allclose(r[2], [0, 0, 1])
    # (h, v, r) is right-handed and orthonormal.
    assert np.allclose(np.cross(h, v), r, atol=1e-12)
    assert np.allclose(np.einsum("ak,ak->a", h, v), 0.0, atol=1e-12)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)


def test_azimuth_zero_is_nose_on():
    """A long body along +x must peak on the beam, not on the nose."""
    body = shapes.box(4.0, 0.6, 0.6)
    got = monostatic_rcs(body, FREQ, np.arange(0, 180, 0.5), 0.0, pols=("VV",))
    az, _, _ = got.peak("VV")
    assert min(abs(az - 90.0), abs(az - 270.0)) < 1.0


def test_occlusion_is_a_no_op_for_a_convex_body(unit_cube):
    az = np.arange(0, 360, 13.0)
    a = monostatic_rcs(unit_cube, FREQ, az, 20.0, pols=("VV",), occlusion=True)
    b = monostatic_rcs(unit_cube, FREQ, az, 20.0, pols=("VV",), occlusion=False)
    assert np.allclose(a.sigma["VV"], b.sigma["VV"], rtol=1e-12)


def test_occlusion_hides_a_blocked_body():
    """A plate parked in front of a second plate must shadow it."""
    front = shapes.plate(2.0, 2.0).translated([0, 0, 1.0])
    back = shapes.plate(0.5, 0.5).translated([0, 0, -1.0])
    both = front + back
    on = monostatic_rcs(both, FREQ, [0.0], [90.0], pols=("VV",), occlusion=True)
    off = monostatic_rcs(both, FREQ, [0.0], [90.0], pols=("VV",), occlusion=False)
    alone = monostatic_rcs(front, FREQ, [0.0], [90.0], pols=("VV",), occlusion=True)
    assert on.sigma["VV"][0] == pytest.approx(alone.sigma["VV"][0], rel=1e-9)
    assert abs(off.sigma["VV"][0] - alone.sigma["VV"][0]) > 1e-6


def test_components_sum_coherently():
    """sigma is |PO + PTD|^2, so it must sit inside the coherent bounds."""
    got = monostatic_rcs(shapes.hopeless_diamond(), 10e9,
                         np.arange(0, 360, 7.0), 5.0, pols=("VV",))
    lo = (np.sqrt(got.sigma_po["VV"]) - np.sqrt(got.sigma_ptd["VV"])) ** 2
    hi = (np.sqrt(got.sigma_po["VV"]) + np.sqrt(got.sigma_ptd["VV"])) ** 2
    assert np.all(got.sigma["VV"] >= lo - 1e-12)
    assert np.all(got.sigma["VV"] <= hi + 1e-12)


def test_disabling_terms():
    az = np.arange(0, 90, 11.0)
    body = shapes.hopeless_diamond()
    only_po = monostatic_rcs(body, 10e9, az, pols=("VV",), use_ptd=False)
    only_pt = monostatic_rcs(body, 10e9, az, pols=("VV",), use_po=False)
    both = monostatic_rcs(body, 10e9, az, pols=("VV",))
    assert np.allclose(only_po.sigma["VV"], both.sigma_po["VV"])
    assert np.allclose(only_pt.sigma["VV"], both.sigma_ptd["VV"])


def test_chunking_does_not_change_the_answer():
    az = np.arange(0, 360, 3.0)
    body = shapes.faceted_delta()
    a = monostatic_rcs(body, 10e9, az, 10.0, pols=("VV",), chunk=7)
    b = monostatic_rcs(body, 10e9, az, 10.0, pols=("VV",), chunk=512)
    assert np.allclose(a.sigma["VV"], b.sigma["VV"], rtol=1e-12)


def test_bistatic_reduces_to_monostatic():
    az = np.arange(0, 180, 17.0)
    body = shapes.hopeless_diamond()
    mono = monostatic_rcs(body, 10e9, az, 12.0, pols=("VV", "HH"))
    bi = bistatic_rcs(body, 10e9, az, 12.0, az, 12.0, pols=("VV", "HH"))
    for p in ("VV", "HH"):
        assert np.allclose(mono.sigma[p], bi.sigma[p], rtol=1e-10)


def test_result_reporting(tmp_path):
    got = monostatic_rcs(shapes.plate(0.3, 0.3), FREQ, np.arange(0, 360, 30.0),
                         40.0, pols=("VV", "HH"))
    assert got.channels == ("VV", "HH")
    assert got.wavelength == pytest.approx(0.03, rel=1e-6)
    assert np.allclose(got.dbsm("VV"), analytic.to_dbsm(got.sigma["VV"]))
    assert "peak" in got.summary("VV")

    path = tmp_path / "rcs.csv"
    got.to_csv(path)
    lines = path.read_text().strip().splitlines()
    assert lines[1] == "az_deg,el_deg,VV_dbsm,HH_dbsm"
    assert len(lines) == 2 + len(got.az_deg)


def test_rejects_bad_arguments():
    body = shapes.plate()
    with pytest.raises(ValueError, match="polarisation"):
        monostatic_rcs(body, FREQ, [0.0], pols=("XY",))
    with pytest.raises(ValueError, match="positive"):
        monostatic_rcs(body, -1.0, [0.0])
