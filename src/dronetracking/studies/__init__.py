"""Robustness / accuracy studies: parameter sweeps that *quantify* the system.

Where ``eval`` scores a single run, this package answers "how does accuracy move as
a knob turns?". It drives the real :func:`dronetracking.pipeline.run_pipeline` across
a grid (measurement noise, participating-device count), aggregates the headline RMSE
metrics over several random seeds, and emits a JSON-serializable summary plus
error-vs-parameter plots.

This is *not* an estimator, so (per the iteration-4 contract) it may import ``config``,
``pipeline`` and ``sim.scenario`` freely — it never touches estimation internals.
"""

from __future__ import annotations

from .sweep import plot_sweep, sweep_devices, sweep_noise

__all__ = ["sweep_noise", "sweep_devices", "plot_sweep"]
