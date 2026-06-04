"""Acceptance tests for the geometry estimation stages.

Covers ``estimation.ranging.build_distance_matrix`` and
``estimation.relative_localization.estimate_layout``.

Per the iteration-1 contract, estimation *source* must never import
``dronetracking.sim``; these *tests* may import the frozen sim leaf functions to
build realistic fixtures (``generate_ranging_records``), which is what we do here.

Layout tolerances:
- Noise-free, drift-free: each true pairwise distance recovered to < 1e-6 m, and
  the layout (after similarity alignment to truth, reflection allowed) has
  per-device RMSE < 1e-3 m.
- Noisy field scenario: distances within a few cm of truth, and layout RMSE
  comfortably under 1.0 m. Seeded for reproducibility.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from dronetracking.config import load_scenario
from dronetracking.sim.scenario import NoiseSpec
from dronetracking.sim.ranging import generate_ranging_records
from dronetracking.transforms import umeyama

from dronetracking.estimation.ranging import build_distance_matrix
from dronetracking.estimation.relative_localization import estimate_layout

from pathlib import Path

SCN = Path(__file__).resolve().parents[1] / "scenarios"


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _drift_free_noise_free(scenario):
    """Return a copy of ``scenario`` with all clock drift and all noise zeroed.

    Two-way ranging cancels clock *offset* exactly, but a nonzero clock *drift*
    (skew) leaves a residual scale factor on the time-of-flight. To recover the
    geometric distance to machine precision we must drop both drift and the
    timestamp/processing jitter — exactly the regime the contract specifies for
    the exact-recovery test.
    """
    devices = tuple(
        dataclasses.replace(d, clock_drift_ppm=0.0) for d in scenario.devices
    )
    return dataclasses.replace(scenario, devices=devices, noise=NoiseSpec())


def _true_positions(scenario):
    """(K, 3) true device positions in scenario ``device_ids`` order."""
    return np.array([d.position_m for d in scenario.devices], dtype=float)


def _true_distance_matrix(scenario):
    """(K, K) true pairwise Euclidean distances."""
    P = _true_positions(scenario)
    diff = P[:, None, :] - P[None, :, :]
    return np.linalg.norm(diff, axis=2)


def _build_observations(scenario, seed):
    """Generate ranging records and wrap them in a minimal observations object.

    ``build_distance_matrix`` only consumes ``device_ids``, ``ranging`` and
    ``speed_of_sound_mps``, so a lightweight stand-in keeps the test focused on
    geometry rather than dragging in the full simulator (forbidden seam anyway).
    """
    rng = np.random.default_rng(seed)
    records = generate_ranging_records(scenario, rng)

    @dataclasses.dataclass
    class _Obs:
        device_ids: tuple
        ranging: tuple
        speed_of_sound_mps: float

    return _Obs(
        device_ids=scenario.device_ids,
        ranging=records,
        speed_of_sound_mps=scenario.speed_of_sound_mps,
    )


def _layout_rmse_to_truth(layout, scenario):
    """Per-device RMSE of the recovered layout after similarity alignment.

    A relative layout is gauge-free up to rotation/translation/reflection, so we
    align to truth with umeyama (no scaling, reflection allowed) before scoring.
    """
    P = layout.positions_local
    truth = _true_positions(scenario)
    sim = umeyama(P, truth, with_scaling=False, allow_reflection=True)
    aligned = sim.apply(P)
    per_device = np.linalg.norm(aligned - truth, axis=1)
    return float(np.sqrt(np.mean(per_device**2)))


# --------------------------------------------------------------------------- #
# build_distance_matrix
# --------------------------------------------------------------------------- #
def test_distance_matrix_shape_and_symmetry():
    sc = _drift_free_noise_free(load_scenario(SCN / "noisefree_ideal.yaml"))
    obs = _build_observations(sc, seed=0)
    dm = build_distance_matrix(obs)

    K = len(sc.device_ids)
    assert dm.device_ids == sc.device_ids
    assert dm.D.shape == (K, K)
    assert dm.W.shape == (K, K)
    assert dm.counts.shape == (K, K)
    assert dm.valid.shape == (K, K)
    # symmetric, zero diagonal
    assert np.allclose(dm.D, dm.D.T, equal_nan=True)
    assert np.allclose(np.diag(dm.D), 0.0)
    # every off-diagonal pair was measured here
    assert dm.n_valid_edges == K * (K - 1) // 2


def test_distance_matrix_noise_free_recovers_truth():
    sc = _drift_free_noise_free(load_scenario(SCN / "noisefree_ideal.yaml"))
    obs = _build_observations(sc, seed=0)
    dm = build_distance_matrix(obs)

    truth = _true_distance_matrix(sc)
    K = len(sc.device_ids)
    for i in range(K):
        for j in range(i + 1, K):
            assert dm.D[i, j] == pytest.approx(truth[i, j], abs=1e-6)
            assert dm.D[i, j] == dm.D[j, i]
    # noise-free => weights are huge (variance hit the tiny floor) and finite
    upper = np.triu_indices(K, 1)
    assert np.all(dm.W[upper] > 0)
    assert np.all(np.isfinite(dm.W[upper]))
    # all edges measured at every round
    assert np.all(dm.counts[upper] == sc.ranging_rounds)


def test_distance_matrix_weights_zero_on_diagonal():
    sc = _drift_free_noise_free(load_scenario(SCN / "noisefree_ideal.yaml"))
    obs = _build_observations(sc, seed=0)
    dm = build_distance_matrix(obs)
    assert np.all(np.diag(dm.W) == 0.0)
    assert not np.any(np.diag(dm.valid))


def test_distance_matrix_noisy_within_a_few_cm():
    sc = load_scenario(SCN / "field_5dev.yaml")  # consumer-grade noise, with drift
    obs = _build_observations(sc, seed=0)
    dm = build_distance_matrix(obs)

    truth = _true_distance_matrix(sc)
    K = len(sc.device_ids)
    errs = []
    for i in range(K):
        for j in range(i + 1, K):
            if dm.valid[i, j]:
                errs.append(abs(dm.D[i, j] - truth[i, j]))
    errs = np.array(errs)
    # robust collapse over 40 rounds should pin distances to a few cm
    assert errs.max() < 0.05
    assert errs.mean() < 0.03


def test_triangle_inequality_flags_a_planted_outlier():
    """A grossly wrong edge that breaks the triangle inequality is marked invalid."""
    sc = _drift_free_noise_free(load_scenario(SCN / "noisefree_ideal.yaml"))
    obs = _build_observations(sc, seed=0)
    dm = build_distance_matrix(obs)

    # Corrupt one edge (dev0<->dev1) to be absurdly long, breaking every triangle
    # it participates in, then re-run the triangle check via a fresh matrix build
    # is not possible (clean records) — instead assert the clean matrix keeps all
    # edges valid, and that a hand-built broken matrix is caught by the same rule.
    assert dm.valid[0, 1]  # clean data: nothing flagged

    from dronetracking.estimation.ranging import flag_triangle_outliers

    K = dm.n_devices
    D = dm.D.copy()
    D[0, 1] = D[1, 0] = D.max() * 5.0  # impossible edge
    valid = np.ones((K, K), dtype=bool)
    np.fill_diagonal(valid, False)
    flagged = flag_triangle_outliers(D, valid, slack_m=1.0)
    assert not flagged[0, 1]
    assert not flagged[1, 0]
    # an untouched edge that is consistent should survive
    assert flagged[2, 3]


# --------------------------------------------------------------------------- #
# estimate_layout
# --------------------------------------------------------------------------- #
def test_estimate_layout_noise_free_recovers_truth():
    sc = _drift_free_noise_free(load_scenario(SCN / "noisefree_ideal.yaml"))
    obs = _build_observations(sc, seed=0)
    dm = build_distance_matrix(obs)
    layout = estimate_layout(dm)

    assert layout.device_ids == sc.device_ids
    assert layout.positions_local.shape == (len(sc.device_ids), 3)
    assert layout.covariances is not None
    assert layout.covariances.shape == (len(sc.device_ids), 3, 3)

    rmse = _layout_rmse_to_truth(layout, sc)
    assert rmse < 1e-3, f"noise-free layout RMSE too high: {rmse}"


def test_estimate_layout_is_centered():
    sc = _drift_free_noise_free(load_scenario(SCN / "noisefree_ideal.yaml"))
    obs = _build_observations(sc, seed=0)
    layout = estimate_layout(build_distance_matrix(obs))
    centroid = layout.positions_local.mean(axis=0)
    assert np.allclose(centroid, 0.0, atol=1e-6)


def test_estimate_layout_preserves_pairwise_distances_noise_free():
    sc = _drift_free_noise_free(load_scenario(SCN / "noisefree_ideal.yaml"))
    obs = _build_observations(sc, seed=0)
    dm = build_distance_matrix(obs)
    layout = estimate_layout(dm)

    P = layout.positions_local
    K = layout.n_devices
    for i in range(K):
        for j in range(i + 1, K):
            recovered = np.linalg.norm(P[i] - P[j])
            assert recovered == pytest.approx(dm.D[i, j], abs=1e-3)


def test_estimate_layout_noisy_under_tolerance():
    sc = load_scenario(SCN / "field_5dev.yaml")
    obs = _build_observations(sc, seed=0)
    dm = build_distance_matrix(obs)
    layout = estimate_layout(dm)

    rmse = _layout_rmse_to_truth(layout, sc)
    # documented tolerance: comfortably sub-meter on a ~200 m array with cm ranges
    assert rmse < 1.0, f"noisy layout RMSE too high: {rmse}"
