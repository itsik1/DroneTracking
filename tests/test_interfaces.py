import numpy as np
import pytest

from dronetracking.estimation.interfaces import RelativeLayout, ClockEstimates, Track


def test_relative_layout_lookup_and_count():
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    rl = RelativeLayout(device_ids=("a", "b"), positions_local=pos, covariances=None)
    assert rl.n_devices == 2
    assert np.allclose(rl.position_of("b"), [1.0, 2.0, 3.0])


def test_clock_to_reference_exactly_inverts_local_clock():
    # Locked convention: local = t*(1 + ppm*1e-6) + offset ; to_reference is the exact inverse.
    ce = ClockEstimates(
        device_ids=("a", "b"),
        offsets_s={"a": 0.0, "b": 0.1},
        drifts_ppm={"a": 0.0, "b": 50.0},
        reference_id="a",
    )
    t_true = 12.34
    local_b = t_true * (1 + 50e-6) + 0.1
    assert ce.to_reference("b", local_b) == pytest.approx(t_true, abs=1e-12)
    # The reference device's local time IS the reference timebase.
    assert ce.to_reference("a", 5.0) == pytest.approx(5.0, abs=1e-12)


def test_clock_to_reference_is_vectorized():
    ce = ClockEstimates(("a",), {"a": 0.2}, {"a": 0.0}, reference_id="a")
    local = np.array([1.0, 2.0, 3.0])
    assert np.allclose(ce.to_reference("a", local), local - 0.2)


def test_track_final_position():
    track = Track(
        times_s=np.array([0.0, 1.0, 2.0]),
        positions_local=np.arange(9, dtype=float).reshape(3, 3),
        covariances=np.zeros((3, 3, 3)),
    )
    assert np.allclose(track.final_position, [6.0, 7.0, 8.0])
