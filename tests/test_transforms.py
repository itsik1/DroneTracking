import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dronetracking.transforms import umeyama, gdop, gn_covariance

R0 = Rotation.from_euler("xyz", [20, 30, 40], degrees=True).as_matrix()
T0 = np.array([10.0, -5.0, 3.0])


def _cloud(n=8, seed=0):
    return np.random.default_rng(seed).standard_normal((n, 3)) * 10.0


def test_umeyama_recovers_rotation_and_translation():
    src = _cloud()
    dst = src @ R0.T + T0
    sim = umeyama(src, dst, with_scaling=False)
    assert np.allclose(sim.R, R0, atol=1e-9)
    assert np.allclose(sim.t, T0, atol=1e-9)
    assert sim.scale == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(sim.apply(src), dst, atol=1e-9)
    assert not sim.is_reflection


def test_umeyama_recovers_scale():
    src = _cloud()
    dst = 2.5 * (src @ R0.T) + T0
    sim = umeyama(src, dst, with_scaling=True)
    assert sim.scale == pytest.approx(2.5, abs=1e-9)
    assert np.allclose(sim.apply(src), dst, atol=1e-8)


def test_umeyama_forbids_reflection_by_default():
    src = _cloud()
    dst = src @ np.diag([1.0, 1.0, -1.0])  # reflected cloud (not a proper rotation)
    sim = umeyama(src, dst, with_scaling=False, allow_reflection=False)
    assert np.linalg.det(sim.R) == pytest.approx(1.0, abs=1e-9)  # stays proper
    assert np.max(np.abs(sim.apply(src) - dst)) > 1.0  # can't fit a reflection


def test_umeyama_allows_reflection_when_requested():
    src = _cloud()
    dst = src @ np.diag([1.0, 1.0, -1.0])
    sim = umeyama(src, dst, with_scaling=False, allow_reflection=True)
    assert sim.is_reflection
    assert np.allclose(sim.apply(src), dst, atol=1e-9)  # reflection fits exactly


def test_apply_single_point_returns_1d():
    src = _cloud()
    sim = umeyama(src, src @ R0.T + T0, with_scaling=False)
    out = sim.apply(np.array([1.0, 2.0, 3.0]))
    assert out.shape == (3,)


def test_gdop_larger_for_collinear_than_spread_sensors():
    target = np.zeros(3)
    spread = np.array([[10, 0, 0], [-10, 0, 0], [0, 10, 0], [0, 0, 10.0]])
    collinear = np.array([[10, 0, 0], [20, 0, 0], [30, 0, 0], [40, 0, 0.0]])
    g_spread = gdop(target, spread)
    g_coll = gdop(target, collinear)
    assert np.isfinite(g_spread) and g_spread > 0
    assert g_coll > g_spread


def test_gn_covariance_full_rank():
    jac = np.diag([2.0, 1.0])
    cov = gn_covariance(jac, residual_variance=4.0)
    assert np.allclose(cov, np.diag([1.0, 4.0]))


def test_gn_covariance_rank_deficient_is_finite():
    jac = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # null direction along axis 2
    cov = gn_covariance(jac, residual_variance=1.0)
    assert np.all(np.isfinite(cov))
    assert cov[2, 2] == pytest.approx(0.0, abs=1e-12)  # pinv zeroes the gauge direction
