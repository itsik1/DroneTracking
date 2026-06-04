"""End-to-end orchestration: synthetic world -> estimation -> evaluation.

This module is the conductor. Like ``eval``, it is allowed to import both the ``sim``
side (to generate a world) and the ``estimation`` side (to process it) — the
ground-truth firewall only forbids the *estimation* package from importing ``sim``.

The estimators only ever receive ``observations``; the ``world`` is handed solely to
``compute_metrics`` for scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .sim.scenario import Scenario
from .sim.simulator import simulate
from .estimation.ranging import build_distance_matrix
from .estimation.relative_localization import estimate_layout
from .estimation.clock_sync import estimate_clocks
from .estimation.tdoa import localize_all
from .estimation.tracking import track_target
from .estimation.georeference import solve_transform, georeference_track
from .estimation.interfaces import Estimates
from .eval.metrics import compute_metrics


@dataclass
class PipelineResult:
    scenario: Scenario
    observations: Any
    world: Any
    estimates: Estimates
    metrics: Dict[str, Any]


def run_pipeline(scenario: Scenario, *, model: str = "cv", sigma_a: float = 2.0) -> PipelineResult:
    """Run the full estimation pipeline on a scenario and score it against truth."""
    # 1. Simulate the synthetic world. Estimators see only `observations`.
    observations, world = simulate(scenario)

    # 2. Relative geometry: pairwise ranging -> distance matrix -> device layout.
    distance_matrix = build_distance_matrix(observations)
    layout = estimate_layout(distance_matrix)

    # 3. Clock offset/drift recovery (no shared clock assumed).
    clocks = estimate_clocks(observations)

    # 4. TDOA target localization per emission, then track the target over time.
    fixes = localize_all(observations, clocks, layout)
    track = track_target(fixes, model=model, sigma_a=sigma_a)

    # 5. Georeference the local frame to real-world coordinates via GPS anchors.
    transform = solve_transform(layout, observations.anchor_gps, scenario.origin_latlon)
    geo_track = georeference_track(track, transform, scenario.origin_latlon)

    estimates = Estimates(layout=layout, clocks=clocks, track=track, geo_track=geo_track)

    # 6. Score everything against ground truth (eval is the only truth-aware stage).
    metrics = compute_metrics(world, observations, estimates)

    return PipelineResult(
        scenario=scenario,
        observations=observations,
        world=world,
        estimates=estimates,
        metrics=metrics,
    )
