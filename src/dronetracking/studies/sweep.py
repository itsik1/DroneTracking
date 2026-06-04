"""Parameter sweeps over the estimation pipeline.

Each sweep walks a 1-D grid of one knob, runs :func:`dronetracking.pipeline.run_pipeline`
``seeds`` times per grid point (varying only the RNG seed via ``dataclasses.replace``),
and aggregates the three headline accuracy metrics — tracking, device-localization and
georeferencing RMSE — into mean/median per point. Results are plain dicts of Python
floats so they ``json.dumps`` cleanly and feed straight into :func:`plot_sweep`.

Two knobs:

``sweep_noise``
    Scale every field of the scenario's :class:`~dronetracking.sim.scenario.NoiseSpec`
    by a common factor. More noise must degrade accuracy — the canonical
    monotonic-trend check.

``sweep_devices``
    Vary the number of participating devices (the first ``N``). Georeferencing needs a
    well-posed anchor set, so any count that cannot keep >=4 GPS anchors is skipped. A
    representative GDOP (geometric dilution of precision) of the recovered constellation
    is reported alongside the RMSEs.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .. import transforms
from ..pipeline import PipelineResult, run_pipeline
from ..sim.scenario import NoiseSpec, Scenario

# The three accuracy metrics every sweep reports, in display order. These are flat keys
# on ``PipelineResult.metrics`` (see the eval harness / iteration-4 contract).
METRIC_KEYS: List[str] = [
    "tracking.rmse_m",
    "device_localization.rmse_m",
    "georeferencing.rmse_m",
]

# Georeferencing is only well-posed with at least this many GPS anchors; counts that
# would drop below it are skipped rather than producing a degenerate fix.
MIN_ANCHORS = 4


# --------------------------------------------------------------------------- helpers

def _run_seeds(scenario: Scenario, seeds: int) -> List[PipelineResult]:
    """Run the pipeline ``seeds`` times, overriding only the RNG seed each time."""
    if seeds < 1:
        raise ValueError(f"seeds must be >= 1, got {seeds}")
    results: List[PipelineResult] = []
    for i in range(seeds):
        seeded = dataclasses.replace(scenario, seed=scenario.seed + i)
        results.append(run_pipeline(seeded))
    return results


def _collect_metric(results: Sequence[PipelineResult], key: str) -> np.ndarray:
    """Finite values of ``key`` across runs (NaN/missing dropped before aggregating)."""
    vals = [float(r.metrics.get(key, math.nan)) for r in results]
    arr = np.asarray(vals, dtype=float)
    return arr[np.isfinite(arr)]


def _aggregate(results: Sequence[PipelineResult]) -> Dict[str, Dict[str, float]]:
    """Mean and median of each headline metric across ``results`` (JSON-safe floats)."""
    mean: Dict[str, float] = {}
    median: Dict[str, float] = {}
    for key in METRIC_KEYS:
        finite = _collect_metric(results, key)
        if finite.size:
            mean[key] = float(np.mean(finite))
            median[key] = float(np.median(finite))
        else:
            # Keep the key (stable shape) but mark it absent in a JSON-friendly way.
            mean[key] = float("nan")
            median[key] = float("nan")
    return {"mean": mean, "median": median}


def _representative_gdop(result: PipelineResult) -> Optional[float]:
    """GDOP of the recovered device constellation about a representative target point.

    Built only from objects the pipeline surfaces on its result — the recovered device
    layout and the estimated track — via the public :func:`transforms.gdop`. (The raw
    per-emission ``TargetFix`` objects, which carry their own ``gdop``, are not exposed
    on ``PipelineResult``; this layout-based GDOP is the available, equivalent geometry
    diagnostic.) Returns ``None`` if the geometry is unavailable or degenerate.
    """
    layout = getattr(result.estimates, "layout", None)
    track = getattr(result.estimates, "track", None)
    if layout is None or track is None:
        return None
    sensors = np.asarray(layout.positions_local, dtype=float)
    positions = np.asarray(track.positions_local, dtype=float)
    if sensors.ndim != 2 or sensors.shape[0] < 4 or positions.size == 0:
        return None
    target = positions.mean(axis=0)  # mid-trajectory representative point
    value = transforms.gdop(target, sensors)
    if not math.isfinite(value):
        return None
    return float(value)


def _scaled_noise(noise: NoiseSpec, factor: float) -> NoiseSpec:
    """Scale every NoiseSpec field by ``factor`` (``dataclasses.replace``)."""
    return dataclasses.replace(
        noise,
        ranging_timestamp_std_s=noise.ranging_timestamp_std_s * factor,
        toa_std_s=noise.toa_std_s * factor,
        proc_delay_jitter_s=noise.proc_delay_jitter_s * factor,
        gps_pos_std_m=noise.gps_pos_std_m * factor,
    )


# ----------------------------------------------------------------------- noise sweep

def sweep_noise(
    base_scenario: Scenario, factors: Sequence[float], seeds: int = 3
) -> Dict[str, Any]:
    """Sweep the measurement-noise scale and report accuracy vs the scale factor.

    For each ``factor`` the scenario's :class:`NoiseSpec` is multiplied through by that
    factor (``dataclasses.replace`` on ``scenario.noise`` then ``scenario``), the
    pipeline is run over ``seeds`` RNG seeds, and the headline RMSE metrics are
    aggregated (mean + median) per factor.

    Returns a JSON-serializable dict::

        {"kind": "noise", "scenario": <name>, "metric_keys": [...], "seeds": int,
         "points": [{"factor": f, "seeds": int,
                     "mean": {metric: v, ...}, "median": {metric: v, ...}}, ...]}
    """
    points: List[Dict[str, Any]] = []
    for factor in factors:
        factor = float(factor)
        scaled = dataclasses.replace(
            base_scenario, noise=_scaled_noise(base_scenario.noise, factor)
        )
        results = _run_seeds(scaled, seeds)
        agg = _aggregate(results)
        points.append({"factor": factor, "seeds": seeds, **agg})

    return {
        "kind": "noise",
        "param": "factor",
        "scenario": base_scenario.name,
        "metric_keys": list(METRIC_KEYS),
        "seeds": seeds,
        "points": points,
    }


# ---------------------------------------------------------------------- device sweep

def sweep_devices(
    base_scenario: Scenario, counts: Sequence[int], seeds: int = 3
) -> Dict[str, Any]:
    """Sweep the number of participating devices and report accuracy + GDOP.

    For each count ``N`` the first ``N`` devices of the scenario are kept. A count is
    *skipped* (recorded in ``"skipped"``) unless the retained subset still has at least
    :data:`MIN_ANCHORS` GPS anchors, so georeferencing stays well-posed. Each surviving
    point reports the same mean/median RMSEs as the noise sweep plus a representative
    ``gdop`` of the recovered constellation (median across seeds; ``None`` if
    unavailable).

    Returns a JSON-serializable dict with ``"points"`` (one per valid count) and
    ``"skipped"`` (the under-determined counts).
    """
    n_total = len(base_scenario.devices)
    points: List[Dict[str, Any]] = []
    skipped: List[int] = []

    for count in counts:
        count = int(count)
        if count < 1 or count > n_total:
            skipped.append(count)
            continue
        subset = base_scenario.devices[:count]
        n_anchors = sum(1 for d in subset if d.has_gps)
        if n_anchors < MIN_ANCHORS:
            skipped.append(count)
            continue

        scen = dataclasses.replace(base_scenario, devices=subset)
        results = _run_seeds(scen, seeds)
        agg = _aggregate(results)

        gdops = [g for g in (_representative_gdop(r) for r in results) if g is not None]
        gdop = float(np.median(gdops)) if gdops else None

        points.append(
            {
                "n_devices": count,
                "n_anchors": n_anchors,
                "seeds": seeds,
                "gdop": gdop,
                **agg,
            }
        )

    return {
        "kind": "devices",
        "param": "n_devices",
        "scenario": base_scenario.name,
        "metric_keys": list(METRIC_KEYS),
        "seeds": seeds,
        "points": points,
        "skipped": skipped,
    }


# -------------------------------------------------------------------------- plotting

def plot_sweep(
    result: Dict[str, Any], out_dir, title: Optional[str] = None
) -> List[Path]:
    """Render error-vs-parameter curves (one line per metric) as PNG(s).

    Uses the non-interactive Agg backend so it is safe headless. Each finite metric
    series is drawn as its own line against the swept parameter (``factor`` for noise,
    ``n_devices`` for the device sweep); points where a metric is NaN/missing are
    dropped from that line. Returns the list of written :class:`Path` objects.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless, before importing pyplot
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    param = result.get("param", "factor")
    metric_keys = result.get("metric_keys", METRIC_KEYS)
    points = result.get("points", [])
    scenario = result.get("scenario", "scenario")
    kind = result.get("kind", "sweep")

    xs_all = [p[param] for p in points]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    plotted_any = False
    for key in metric_keys:
        xs: List[float] = []
        ys: List[float] = []
        for p in points:
            y = p.get("mean", {}).get(key, math.nan)
            if y is None or not math.isfinite(float(y)):
                continue
            xs.append(float(p[param]))
            ys.append(float(y))
        if xs:
            ax.plot(xs, ys, marker="o", label=key)
            plotted_any = True

    if not plotted_any and xs_all:
        # Degenerate but non-empty grid: draw a flat zero baseline so the PNG is valid.
        ax.plot([float(x) for x in xs_all], [0.0] * len(xs_all), marker="o",
                label="(no finite metrics)")

    ax.set_xlabel("noise factor" if param == "factor" else "participating devices")
    ax.set_ylabel("RMSE (m)")
    ax.set_title(title or f"{scenario}: accuracy vs {param}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small")
    fig.tight_layout()

    out_path = out_dir / f"{scenario}_{kind}_sweep.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return [out_path]
