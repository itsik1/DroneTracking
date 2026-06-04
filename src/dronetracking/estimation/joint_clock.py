"""Joint clock + position estimation: co-estimate residual clock offsets with the fixes.

The plain TDOA solver in :mod:`dronetracking.estimation.tdoa` trusts the supplied
:class:`~dronetracking.estimation.interfaces.ClockEstimates` completely — it lifts every
arrival onto the reference timebase and treats the result as exact. When clock sync leaves
a *residual* per-device timing error ``δ_i``, that bias maps directly into the measured
range differences ``d_i = c·(τ_i − τ_ref)`` and biases the position fix; the plain solver
has no degree of freedom to absorb it.

This module augments the least-squares unknowns with one residual clock offset ``δ_i`` per
device, defined **relative to the clock reference device** (``clocks.reference_id``, whose
``δ`` is fixed at 0 — the gauge anchor, since only relative timing is observable from
arrival differences). The modeled arrival for device ``i`` becomes ``(τ_i_ref + δ_i)``, so
the per-emission, per-device range-difference residual against that emission's TDOA
reference ``r`` is

    res = ( ‖x − p_i‖ − ‖x − p_r‖ − c·(d_i + (δ_i − δ_r)) ) / σ

i.e. the plain-TDOA residual minus the differential clock correction ``c·(δ_i − δ_r)``.
Each ``δ_i`` is tied to zero by a Gaussian prior residual row ``δ_i / clock_prior_s`` so the
solver only spends a clock correction the data demands.

**Why redundancy matters (and where it comes from).** For a *single* emission the joint
Jacobian is rank-deficient: ``N`` arrivals give only ``N−1`` range-difference rows but the
unknowns are ``3`` position + ``N−1`` clock offsets, so the per-device ``δ_i`` is confounded
with position and only the prior pins it (≈ plain TDOA). The clock offsets become
*identifiable* when they are **shared across multiple emissions**: with ``M`` emissions the
system has ``M·(N−1)`` range-difference rows for only ``3M + (N−1)`` unknowns — the same
``δ`` must explain every emission's geometry at once, which separates the constant clock
bias from the moving target. That is exactly what this module does: it groups the supplied
arrivals by ``emission_idx`` and solves for one position **per emission** plus **one shared
δ per device**. Given several emissions this drives out the residual clock error the plain
solver cannot.

Position seeds come from the plain refined fix
(:func:`~dronetracking.estimation.tdoa.localize_emission`) per emission with the same
clocks; everything is then refined jointly with :func:`scipy.optimize.least_squares`
(``loss="soft_l1"``). Reported per-emission covariance is each emission's block of the
**position-marginal** Gauss-Newton covariance — the Schur complement of the joint
information that folds out (marginalizes) the shared clock unknowns, so residual clock
uncertainty propagates into the position covariance honestly. This mirrors the inverse-
Fisher convention of :func:`~dronetracking.transforms.gn_covariance` used by plain TDOA.

This module is part of the estimation package and therefore MUST NOT import
``dronetracking.sim`` (the ground-truth firewall).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.optimize import least_squares

from .. import transforms
from ..datatypes import TargetFix
from . import tdoa
from .interfaces import ClockEstimates, RelativeLayout

# A 3D fix needs 3 position unknowns; each non-reference device adds one (shared) clock
# unknown. We require an emission to carry at least this many devices so the extra clock
# degrees of freedom keep the system over-determined (the contract's redundancy
# requirement). True identifiability of the shared δ additionally needs several emissions.
MIN_DEVICES_JOINT = 5


def _group_by_emission(arrivals: Sequence) -> Dict[int, list]:
    groups: Dict[int, list] = {}
    for arr in arrivals:
        groups.setdefault(arr.emission_idx, []).append(arr)
    return groups


class _EmissionBlock:
    """Pre-computed per-emission quantities for the joint residual/Jacobian."""

    __slots__ = ("idx", "taus", "positions", "device_ids", "ref", "d", "non_ref", "t_mean")

    def __init__(self, idx, group, clocks: ClockEstimates, layout: RelativeLayout, c: float):
        taus = np.asarray(
            [float(clocks.to_reference(a.device_id, a.toa_local_s)) for a in group],
            dtype=float,
        )
        positions = np.asarray(
            [np.asarray(layout.position_of(a.device_id), dtype=float) for a in group],
            dtype=float,
        )
        device_ids = [a.device_id for a in group]
        ref = int(np.argmin(taus))  # earliest arrival -> best-conditioned differences
        self.idx = idx
        self.taus = taus
        self.positions = positions
        self.device_ids = device_ids
        self.ref = ref
        self.d = c * (taus - taus[ref])  # measured range diffs (d[ref] == 0)
        self.non_ref = [i for i in range(len(group)) if i != ref]
        self.t_mean = float(np.mean(taus))


def _solve_joint(
    blocks: List[_EmissionBlock],
    clocks: ClockEstimates,
    speed_of_sound_mps: float,
    clock_prior_s: float,
    toa_var_s2: float,
    seed_positions: List[np.ndarray],
):
    """Core joint least-squares: one position per block + one shared δ per device.

    Returns ``(positions, delta_index, res, delta)`` where ``positions`` is a list of solved
    ``(3,)`` arrays (one per block, in input order), ``delta_index`` maps each non-reference
    device_id to its slot in the δ sub-vector, ``res`` is the ``scipy`` result (with ``.jac``
    over the FULL prior-normalized parameter vector), and ``delta`` is the solved shared
    per-device clock offsets in **seconds**.
    """
    c = float(speed_of_sound_mps)
    prior = float(clock_prior_s)
    sigma = float(np.sqrt(max(2.0 * c * c * toa_var_s2, np.finfo(float).tiny)))
    ref_id = clocks.reference_id

    # One shared δ per device that ever appears as a non-reference device, EXCLUDING the
    # global clock reference (its δ is fixed at 0 — the gauge anchor).
    delta_ids: List[str] = []
    seen = set()
    for blk in blocks:
        for dev in blk.device_ids:
            if dev != ref_id and dev not in seen:
                seen.add(dev)
                delta_ids.append(dev)
    delta_index = {dev: j for j, dev in enumerate(delta_ids)}
    n_delta = len(delta_ids)
    M = len(blocks)

    # The clock unknowns are optimized in dimensionless, prior-normalized units
    # ``u_i = δ_i / clock_prior_s`` (so δ_i ≈ 1e-4 s becomes u_i ≈ O(1)). This keeps the
    # Jacobian well scaled — the Gaussian prior row is simply ``u_i`` (×1) instead of
    # ``δ_i / clock_prior_s`` (×1e4), which otherwise overflows the trust-region linear
    # algebra — while the measurement coupling becomes ``c · clock_prior_s · u_i``. δ in
    # seconds is recovered as ``clock_prior_s · u`` after the solve.
    def _u_of(dev: str, u: np.ndarray) -> float:
        if dev == ref_id:
            return 0.0
        return float(u[delta_index[dev]])

    theta0 = np.concatenate(
        [np.concatenate([np.asarray(p, dtype=float) for p in seed_positions]),
         np.zeros(n_delta)]
    )
    c_prior = c * prior

    def residuals(theta: np.ndarray) -> np.ndarray:
        u = theta[3 * M:]
        rows = []
        for m, blk in enumerate(blocks):
            x = theta[3 * m:3 * m + 3]
            p_ref = blk.positions[blk.ref]
            u_r = _u_of(blk.device_ids[blk.ref], u)
            r_ref = np.linalg.norm(x - p_ref)
            for i in blk.non_ref:
                u_i = _u_of(blk.device_ids[i], u)
                r_i = np.linalg.norm(blk.positions[i] - x)
                meas = (r_i - r_ref - (blk.d[i] + c_prior * (u_i - u_r))) / sigma
                rows.append(meas)
        prior_rows = u.tolist() if n_delta else []  # δ_i/clock_prior_s == u_i
        return np.asarray(rows + prior_rows, dtype=float)

    res = least_squares(residuals, theta0, loss="soft_l1", method="trf")
    positions = [res.x[3 * m:3 * m + 3].copy() for m in range(M)]
    delta = prior * res.x[3 * M:]  # back to seconds for downstream reporting
    return positions, delta_index, res, delta


def _position_marginal_covariance(res, n_pos: int) -> np.ndarray:
    """Position-marginal covariance ``(3M × 3M)`` with the shared clock unknowns folded in.

    From the Gauss-Newton information ``F = JᵀJ`` partitioned as ``[[F_pp, F_pd], [F_dp,
    F_dd]]`` (position block ``F_pp`` of size ``n_pos = 3M``, clock block ``F_dd``), the
    covariance of the position parameters after **marginalizing out** the clock nuisances is
    ``s2 · pinv(F_pp − F_pd · F_dd⁻¹ · F_dp)`` — the Schur complement. Because the clock
    unknowns are prior-normalized, every prior row contributes ``1`` to ``F_dd``'s diagonal,
    so ``F_dd`` is well-conditioned and invertible; the Schur complement is then formed and
    pseudo-inverted at ``O(1)`` magnitudes (no overflow, unlike inverting the whole badly
    scaled ``F`` at once). ``pinv`` keeps any residual gauge/weak-geometry direction finite.
    The empirical-residual-variance scale is floored at 1 (same convention as plain TDOA)
    so a near-perfect fit stays honest. Marginalizing the clocks correctly inflates the
    position covariance to reflect the residual clock uncertainty.
    """
    jac = np.asarray(res.jac, dtype=float)
    n_params = res.x.size
    dof = max(res.fun.size - n_params, 1)
    s2_hat = max(float(np.sum(res.fun ** 2) / dof), 1.0)

    # The joint Jacobian is block-sparse (each emission's position columns appear only in
    # that emission's rows). On some BLAS backends (notably Apple Accelerate) the vectorized
    # matmul spuriously raises divide-by-zero / overflow FPE *warnings* on such structured
    # matrices even though the result is exact and finite. Silence those benign flags around
    # the linear algebra; we assert finiteness below.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        F = jac.T @ jac
        F_pp = F[:n_pos, :n_pos]
        if n_params == n_pos:  # no clock unknowns (shouldn't happen, but stay safe)
            cov = s2_hat * np.linalg.pinv(F_pp)
        else:
            F_pd = F[:n_pos, n_pos:]
            F_dd = F[n_pos:, n_pos:]
            # F_dd is SPD thanks to the unit prior rows; solve rather than invert explicitly.
            schur = F_pp - F_pd @ np.linalg.solve(F_dd, F_pd.T)
            cov = s2_hat * np.linalg.pinv(schur)

    if not np.all(np.isfinite(cov)):
        # Defensive fallback: drop the clock marginalization and report the (always finite)
        # position-only information block. Should not trigger in practice.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            cov = s2_hat * np.linalg.pinv(F_pp)
    return cov


def localize_emission_joint(
    arrivals: Sequence,
    clocks: ClockEstimates,
    layout: RelativeLayout,
    speed_of_sound_mps: float,
    *,
    clock_prior_s: float = 1e-4,
    toa_var_s2: float = 1e-8,
) -> TargetFix:
    """Localize an emission while co-estimating residual per-device clock offsets.

    Same range-difference (TDOA) geometry as
    :func:`dronetracking.estimation.tdoa.localize_emission`, but the least-squares unknowns
    are augmented with a residual clock offset ``δ_i`` per device (relative to
    ``clocks.reference_id``, whose ``δ`` is fixed at 0). Each arrival is first mapped onto
    the reference timebase via ``clocks.to_reference``; ``δ_i`` then absorbs the *residual*
    timing error, and every ``δ_i`` is regularized toward zero by a Gaussian prior row
    ``δ_i / clock_prior_s``.

    ``arrivals`` may span **multiple emissions** (grouped internally by ``emission_idx``).
    Sharing one ``δ`` set across several emissions is what makes the clock offsets
    *identifiable* — see the module docstring. With a single emission the result is
    prior-limited and reduces to plain TDOA. The returned :class:`TargetFix` is for the
    **latest** emission present (the most-constrained one once the shared clocks are solved);
    use :func:`localize_all_joint` to get a fix per emission.

    Parameters
    ----------
    arrivals
        ``AcousticArrival``-like objects (``.device_id``, ``.toa_local_s``, ``.emission_idx``).
        Pass all emissions of a pass together to let the shared clock offsets be observed.
    clocks
        Clock estimates used to lift each local arrival onto the reference timebase; any
        residual error left after that mapping is what ``δ_i`` is free to correct.
    layout
        Relative device layout providing each device's position.
    speed_of_sound_mps
        Propagation speed ``c``.
    clock_prior_s
        Standard deviation (seconds) of the zero-mean Gaussian prior on each ``δ_i``. Smaller
        ⇒ stiffer (trusts the supplied clocks more); larger ⇒ lets the data move the clocks
        more. Default ``1e-4`` s.
    toa_var_s2
        Per-arrival time-of-arrival variance; the range-difference measurement variance is
        ``2·c²·toa_var_s2`` (two independent ToAs differenced).

    Returns
    -------
    TargetFix
        Fix for the latest emission: position, position-marginal covariance (clock
        nuisances folded out via the Schur complement), GDOP, residual RMS, device count
        and emission timestamp.

    Raises
    ------
    ValueError
        If the latest emission carries fewer than :data:`MIN_DEVICES_JOINT` devices (the
        extra clock unknowns need that redundancy).
    """
    fixes = localize_all_joint(
        arrivals, clocks, layout, speed_of_sound_mps,
        clock_prior_s=clock_prior_s, toa_var_s2=toa_var_s2,
        min_devices=MIN_DEVICES_JOINT, _require_at_least_one=True,
    )
    # Return the latest-time fix.
    return max(fixes, key=lambda f: f.t)


def localize_all_joint(
    arrivals: Sequence,
    clocks: ClockEstimates,
    layout: RelativeLayout,
    speed_of_sound_mps: float,
    *,
    clock_prior_s: float = 1e-4,
    toa_var_s2: float = 1e-8,
    min_devices: int = MIN_DEVICES_JOINT,
    _require_at_least_one: bool = False,
) -> List[TargetFix]:
    """Jointly localize every emission with one **shared** per-device clock-offset solve.

    Drop-in analogue of :func:`dronetracking.estimation.tdoa.localize_all` that co-estimates
    the residual clock offsets. Groups ``arrivals`` by ``emission_idx``; every emission with
    at least ``min_devices`` devices contributes a position block, and all blocks share one
    ``δ`` per device. Returns one :class:`TargetFix` per qualifying emission, sorted by time.

    ``arrivals`` may be a raw sequence of arrivals or anything with an ``.acoustic``
    attribute (e.g. an ``Observations``); in the latter case ``speed_of_sound_mps`` may be
    taken from the object if not supplied positionally — but here it is always required for
    a stable signature.
    """
    # Accept an Observations-like object transparently.
    raw = getattr(arrivals, "acoustic", arrivals)
    groups = _group_by_emission(raw)

    c = float(speed_of_sound_mps)
    usable_idxs = sorted(k for k, g in groups.items() if len(g) >= min_devices)
    if not usable_idxs:
        if _require_at_least_one:
            biggest = max((len(g) for g in groups.values()), default=0)
            raise ValueError(
                f"need >= {min_devices} devices in an emission for a joint fix; "
                f"largest emission has {biggest}"
            )
        return []

    blocks = [_EmissionBlock(k, groups[k], clocks, layout, c) for k in usable_idxs]
    seeds = [
        tdoa.localize_emission(groups[k], clocks, layout, c, toa_var_s2=toa_var_s2).position
        for k in usable_idxs
    ]

    positions, delta_index, res, delta = _solve_joint(
        blocks, clocks, c, clock_prior_s, toa_var_s2, seeds
    )

    # Position-marginal covariance once (clock nuisances folded in via Schur complement);
    # each fix takes its own 3x3 diagonal block.
    pos_cov = _position_marginal_covariance(res, 3 * len(blocks))

    fixes = [
        _assemble_fix(blocks, positions, delta_index, delta, pos_cov, c, clocks, which)
        for which in range(len(blocks))
    ]
    fixes.sort(key=lambda f: f.t)
    return fixes


def _assemble_fix(
    blocks: List[_EmissionBlock],
    positions: List[np.ndarray],
    delta_index: Dict[str, int],
    delta: np.ndarray,
    pos_cov: np.ndarray,
    speed_of_sound_mps: float,
    clocks: ClockEstimates,
    which: int,
) -> TargetFix:
    """Build the :class:`TargetFix` for block ``which`` of a solved joint problem.

    ``delta`` is the solved shared per-device clock offsets in **seconds** (indexed by
    ``delta_index``); ``pos_cov`` is the ``(3M × 3M)`` position-marginal covariance from
    :func:`_position_marginal_covariance` (clock nuisances already folded in), of which this
    emission's ``3 × 3`` diagonal block is reported.
    """
    c = float(speed_of_sound_mps)
    blk = blocks[which]
    x = positions[which]
    ref_id = clocks.reference_id

    def _delta_of(dev: str) -> float:
        return 0.0 if dev == ref_id else float(delta[delta_index[dev]])

    # Covariance: this emission's 3x3 block of the position-marginal covariance (the shared
    # clock nuisances are marginalized out, so residual clock uncertainty is reflected here).
    p0 = 3 * which
    cov = np.asarray(pos_cov[p0:p0 + 3, p0:p0 + 3], dtype=float)

    gdop = transforms.gdop(x, blk.positions)

    # Residual RMS over this block's measurement rows only (meters).
    p_ref = blk.positions[blk.ref]
    r_ref = np.linalg.norm(x - p_ref)
    del_r = _delta_of(blk.device_ids[blk.ref])
    resid_m = []
    for i in blk.non_ref:
        del_i = _delta_of(blk.device_ids[i])
        r_i = np.linalg.norm(blk.positions[i] - x)
        resid_m.append(r_i - r_ref - (blk.d[i] + c * (del_i - del_r)))
    resid_m = np.asarray(resid_m, dtype=float)
    residual_rms = float(np.sqrt(np.mean(resid_m ** 2))) if resid_m.size else 0.0

    return TargetFix(
        position=np.asarray(x, dtype=float),
        cov=cov,
        gdop=float(gdop),
        residual_rms=residual_rms,
        n_devices=len(blk.device_ids),
        t=blk.t_mean,
    )
