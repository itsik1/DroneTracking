import numpy as np
import pytest

from dronetracking.datatypes import DistanceMatrix, TargetFix


def test_distance_matrix_counts_valid_undirected_edges():
    valid = np.array(
        [[False, True, True], [True, False, False], [True, False, False]]
    )
    dm = DistanceMatrix(
        device_ids=("a", "b", "c"),
        D=np.zeros((3, 3)),
        W=np.zeros((3, 3)),
        counts=np.zeros((3, 3), int),
        valid=valid,
    )
    assert dm.n_devices == 3
    assert dm.n_valid_edges == 2  # (a,b) and (a,c); (b,c) invalid


def test_targetfix_error_radius_and_vertical_split():
    cov = np.diag([1.0, 4.0, 100.0])
    fix = TargetFix(
        t=0.0, position=np.zeros(3), cov=cov, gdop=2.0, residual_rms=0.01, n_devices=5
    )
    assert fix.error_radius == pytest.approx(np.sqrt(105.0))
    assert fix.vertical_std == pytest.approx(10.0)
    assert fix.horizontal_std == pytest.approx(np.sqrt(5.0))


def test_targetfix_flags_weak_vertical_observability():
    weak = TargetFix(np.zeros(3), np.diag([1.0, 1.0, 100.0]), gdop=3.0, residual_rms=0.0, n_devices=4, t=0.0)
    strong = TargetFix(np.zeros(3), np.diag([1.0, 1.0, 1.5]), gdop=1.5, residual_rms=0.0, n_devices=4, t=0.0)
    assert weak.weak_vertical
    assert not strong.weak_vertical
