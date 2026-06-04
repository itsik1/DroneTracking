"""Acceptance tests for joint clock + position estimation.

The plain TDOA solver in :mod:`dronetracking.estimation.tdoa` trusts the supplied
:class:`ClockEstimates` completely: it lifts every arrival onto the reference timebase
and assumes the result is exact. When clock sync leaves a *residual* per-device timing
error, that bias maps straight into the range differences and biases every position fix.

:func:`dronetracking.estimation.joint_clock.localize_emission_joint` (and its per-emission
list form :func:`~dronetracking.estimation.joint_clock.localize_all_joint`) co-estimate one
residual clock offset ``δ_i`` per device — **shared across all emissions** — alongside the
positions, each pulled toward zero by a Gaussian prior. Sharing the offsets across several
emissions is what makes them identifiable, so this absorbs the residual clock error the
plain solver cannot.

These tests ISOLATE the joint solver by feeding it the TRUE relative layout and acoustic
arrivals from the FROZEN sim leaf function ``generate_acoustic_arrivals`` (importing sim
leaf functions in *tests* is permitted by the contract — only estimation *source* must
not import sim). The clocks fed in are deliberately PERTURBED from truth.
"""

from __future__ import annotations

import numpy as np
import pytest

# Frozen estimation contract / shared types (allowed in source and tests).
from dronetracking.estimation.interfaces import RelativeLayout, ClockEstimates

# Modules under test + the plain TDOA baseline we must beat / match.
from dronetracking.estimation import tdoa
from dronetracking.estimation.joint_clock import (
    localize_emission_joint,
    localize_all_joint,
)

# Frozen sim leaf functions — TESTS may use these for realistic fixtures.
from dronetracking.sim.scenario import Scenario, TrajectorySpec, NoiseSpec, DeviceSpec
from dronetracking.sim.trajectory import trajectory_position
from dronetracking.sim.acoustic import generate_acoustic_arrivals, emission_times

C = 343.0


# --------------------------------------------------------------------------------------
# Fixtures: a non-coplanar ground array (spread in x, y AND z) so the 3D fix is well
# conditioned, drone flying overhead and moving so geometry varies per emission (which is
# what makes the shared clock offsets observable). dev0 is the clock reference with true
# (offset, drift) = (0, 0) per the locked convention. Six devices gives the redundancy
# the joint solver needs to separate the clock nuisance from position.
# --------------------------------------------------------------------------------------

def _devices():
    return [
        DeviceSpec("dev0", (0.0, 0.0, 0.0), clock_offset_s=0.0, clock_drift_ppm=0.0),
        DeviceSpec("dev1", (200.0, 0.0, 12.0), clock_offset_s=0.10, clock_drift_ppm=25.0),
        DeviceSpec("dev2", (0.0, 200.0, 8.0), clock_offset_s=-0.07, clock_drift_ppm=-18.0),
        DeviceSpec("dev3", (200.0, 200.0, 20.0), clock_offset_s=0.04, clock_drift_ppm=40.0),
        DeviceSpec("dev4", (100.0, 100.0, 35.0), clock_offset_s=-0.12, clock_drift_ppm=15.0),
        DeviceSpec("dev5", (50.0, 220.0, 28.0), clock_offset_s=0.06, clock_drift_ppm=-30.0),
    ]


def _scenario(devices, duration=10.0, dt=1.0):
    return Scenario(
        name="joint_clock", seed=0, speed_of_sound_mps=C, sample_rate_hz=48000.0,
        duration_s=duration, dt_s=dt, ranging_rounds=10, origin_latlon=(32.0, 34.0),
        devices=tuple(devices),
        trajectory=TrajectorySpec(
            "linear", {"start_m": [-40.0, 60.0], "end_m": [260.0, 140.0]}, z_m=120.0
        ),
        noise=NoiseSpec(),  # NOISE-FREE: any position error is the clock residual, not jitter.
    )


def _truth_layout(scenario) -> RelativeLayout:
    device_ids = scenario.device_ids
    positions = np.array([d.position_m for d in scenario.devices], dtype=float)
    return RelativeLayout(device_ids=device_ids, positions_local=positions, covariances=None)


def _clocks(scenario, offset_perturb=None) -> ClockEstimates:
    """ClockEstimates built from the scenario's TRUE clocks, then optionally PERTURBED.

    ``offset_perturb`` maps device_id -> extra seconds added to that device's estimated
    offset, simulating a residual sync error clock-sync failed to remove. The reference
    device (dev0) is never perturbed — δ is defined relative to it.
    """
    device_ids = scenario.device_ids
    offsets = {d.id: float(d.clock_offset_s) for d in scenario.devices}
    drifts = {d.id: float(d.clock_drift_ppm) for d in scenario.devices}
    if offset_perturb:
        for dev_id, extra in offset_perturb.items():
            offsets[dev_id] = offsets[dev_id] + float(extra)
    return ClockEstimates(
        device_ids=device_ids,
        offsets_s=offsets,
        drifts_ppm=drifts,
        reference_id=device_ids[0],
    )


def _plain_fixes_by_emission(arrivals, clocks, layout, scenario):
    """Plain per-emission TDOA fixes, keyed by emission_idx."""
    groups = {}
    for arr in arrivals:
        groups.setdefault(arr.emission_idx, []).append(arr)
    out = {}
    for k, g in groups.items():
        out[k] = tdoa.localize_emission(g, clocks, layout, scenario.speed_of_sound_mps)
    return out


# --------------------------------------------------------------------------------------
# (a) Under PERTURBED clocks, the joint solver beats plain TDOA by a wide margin.
#     The shared-δ solve is fed ALL emissions (that is the redundancy that makes the
#     clock offsets identifiable).
# --------------------------------------------------------------------------------------

def test_joint_beats_plain_tdoa_under_perturbed_clocks():
    sc = _scenario(_devices())
    layout = _truth_layout(sc)

    # Inject a residual offset on two non-reference devices (1e-4 .. 5e-4 s) — the size
    # clock sync can plausibly leave behind. dev0 (reference) stays exact.
    perturb = {"dev1": 5.0e-4, "dev3": -5.0e-4}
    clocks_bad = _clocks(sc, offset_perturb=perturb)

    arrivals = generate_acoustic_arrivals(sc, np.random.default_rng(0))
    times = emission_times(sc)

    plain_by_em = _plain_fixes_by_emission(arrivals, clocks_bad, layout, sc)
    # Operating point: a prior sized to the expected residual scale (~5e-4 s, matching the
    # injected perturbation). This is the realistic setting — the prior std encodes how
    # large a residual clock error clock sync is expected to leave behind.
    joint_fixes = localize_all_joint(
        arrivals, clocks_bad, layout, sc.speed_of_sound_mps, clock_prior_s=5e-4
    )
    assert len(joint_fixes) == len(plain_by_em) >= 5  # several emissions -> identifiable δ

    # Both the joint fixes (sorted by t) and the emissions are monotonic in time, so the
    # i-th joint fix corresponds to the i-th emission index. This avoids fragile
    # timestamp matching (a fix's t is the mean ARRIVAL time, offset from the emission time).
    emission_order = sorted(plain_by_em)  # emission indices in time order
    plain_errs, joint_errs = [], []
    for jf, k in zip(joint_fixes, emission_order):
        truth = trajectory_position(sc, float(times[k]))
        plain_err = float(np.linalg.norm(plain_by_em[k].position - truth))
        joint_err = float(np.linalg.norm(jf.position - truth))
        plain_errs.append(plain_err)
        joint_errs.append(joint_err)

        # Diagnostics are populated and sane.
        assert jf.cov.shape == (3, 3)
        assert np.all(np.isfinite(jf.cov))
        assert np.isfinite(jf.gdop) and jf.gdop > 0
        assert np.isfinite(jf.residual_rms)
        assert jf.n_devices == 6

    mean_plain = float(np.mean(plain_errs))
    mean_joint = float(np.mean(joint_errs))

    # The perturbation must actually hurt plain TDOA (otherwise the test proves nothing).
    assert mean_plain > 0.3, f"perturbation too weak: plain mean error only {mean_plain:.3f} m"
    # The joint solve must cut the error by a wide margin at the matched prior.
    assert mean_joint < 0.25 * mean_plain, (
        f"joint mean {mean_joint:.3f} m vs plain mean {mean_plain:.3f} m: insufficient gain"
    )
    # And land close to truth in absolute terms.
    assert mean_joint < 0.15, f"joint mean error {mean_joint:.3f} m too large"

    # Even the (stiffer) DEFAULT prior still materially beats plain TDOA out of the box.
    default_fixes = localize_all_joint(arrivals, clocks_bad, layout, sc.speed_of_sound_mps)
    default_errs = [
        float(np.linalg.norm(jf.position - trajectory_position(sc, float(times[k]))))
        for jf, k in zip(default_fixes, emission_order)
    ]
    assert float(np.mean(default_errs)) < 0.5 * mean_plain, (
        f"default-prior joint mean {np.mean(default_errs):.3f} m not materially below "
        f"plain {mean_plain:.3f} m"
    )


def test_localize_emission_joint_returns_latest_emission_fix():
    """The single-fix entry point returns the latest emission, far better than plain TDOA."""
    sc = _scenario(_devices())
    layout = _truth_layout(sc)
    perturb = {"dev1": 5.0e-4, "dev3": -5.0e-4}
    clocks_bad = _clocks(sc, offset_perturb=perturb)

    arrivals = generate_acoustic_arrivals(sc, np.random.default_rng(0))
    times = emission_times(sc)

    fix = localize_emission_joint(arrivals, clocks_bad, layout, sc.speed_of_sound_mps)

    # It is the latest emission.
    last_k = len(times) - 1
    assert fix.t == pytest.approx(
        max(f.t for f in localize_all_joint(arrivals, clocks_bad, layout, sc.speed_of_sound_mps))
    )
    truth = trajectory_position(sc, float(times[last_k]))
    plain = tdoa.localize_emission(
        [a for a in arrivals if a.emission_idx == last_k],
        clocks_bad, layout, sc.speed_of_sound_mps,
    )
    joint_err = float(np.linalg.norm(fix.position - truth))
    plain_err = float(np.linalg.norm(plain.position - truth))
    assert joint_err < 0.5 * plain_err, (
        f"latest-emission joint {joint_err:.3f} m not materially below plain {plain_err:.3f} m"
    )


# --------------------------------------------------------------------------------------
# (b) On CLEAN (true) clocks the joint solver matches plain TDOA — no regression.
# --------------------------------------------------------------------------------------

def test_joint_matches_plain_tdoa_on_clean_clocks():
    sc = _scenario(_devices())
    layout = _truth_layout(sc)
    clocks_true = _clocks(sc)

    arrivals = generate_acoustic_arrivals(sc, np.random.default_rng(1))
    times = emission_times(sc)

    plain_by_em = _plain_fixes_by_emission(arrivals, clocks_true, layout, sc)
    joint_fixes = localize_all_joint(arrivals, clocks_true, layout, sc.speed_of_sound_mps)
    emission_order = sorted(plain_by_em)

    for jf, k in zip(joint_fixes, emission_order):
        truth = trajectory_position(sc, float(times[k]))

        # Both recover truth on clean clocks; joint must not regress vs plain.
        joint_err = float(np.linalg.norm(jf.position - truth))
        assert joint_err < 1e-2, f"emission {k}: joint clean error {joint_err:.2e} m"
        # Joint and plain agree closely (prior keeps δ≈0 when clocks are already right).
        assert np.linalg.norm(jf.position - plain_by_em[k].position) < 1e-2, (
            f"emission {k}: joint {jf.position} disagrees with plain {plain_by_em[k].position}"
        )


# --------------------------------------------------------------------------------------
# Guard: the joint solve needs the extra unknowns, so an emission needs >= 5 devices.
# --------------------------------------------------------------------------------------

def test_joint_requires_at_least_five_devices():
    sc = _scenario(_devices())
    layout = _truth_layout(sc)
    clocks_true = _clocks(sc)
    arrivals = generate_acoustic_arrivals(sc, np.random.default_rng(2))

    # Keep only 4 devices per emission -> no emission qualifies -> ValueError.
    four_per_em = [a for a in arrivals if a.device_id in ("dev0", "dev1", "dev2", "dev3")]
    with pytest.raises(ValueError):
        localize_emission_joint(four_per_em, clocks_true, layout, sc.speed_of_sound_mps)

    # And the list form simply returns nothing for an all-too-small batch.
    assert localize_all_joint(four_per_em, clocks_true, layout, sc.speed_of_sound_mps) == []
