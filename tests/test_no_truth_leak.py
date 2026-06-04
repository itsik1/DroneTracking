"""The ground-truth firewall, enforced by tests rather than by discipline.

1. STATIC: no source file in `dronetracking.estimation` may import `dronetracking.sim`
   (AST-checked, so docstring mentions of "sim" don't count).
2. STRUCTURAL: `Observations` exposes only measurable quantities — no positions,
   clocks, emission times, or track.
"""

from __future__ import annotations

import ast
from pathlib import Path

from dronetracking.sim.observations import Observations

EST_DIR = Path(__file__).resolve().parents[1] / "src" / "dronetracking" / "estimation"


def _sim_references(path: Path):
    """Return any import in ``path`` that references the ``sim`` package."""
    tree = ast.parse(path.read_text())
    refs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "sim" in alias.name.split("."):
                    refs.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if "sim" in module.split("."):
                refs.append(module)
            for alias in node.names:  # e.g. `from dronetracking import sim`
                if alias.name == "sim":
                    refs.append(f"{module}.{alias.name}")
    return refs


def test_estimation_package_never_imports_sim():
    offenders = {}
    for py in sorted(EST_DIR.glob("*.py")):
        bad = _sim_references(py)
        if bad:
            offenders[py.name] = bad
    assert not offenders, f"ground-truth firewall breached: {offenders}"


def test_estimation_package_has_files_to_check():
    # Guard against the firewall test silently passing on an empty glob.
    assert len(list(EST_DIR.glob("*.py"))) >= 6


def test_observations_exposes_no_ground_truth():
    fields = set(Observations.__dataclass_fields__)
    forbidden = {
        "device_positions", "positions", "clock_offsets", "clock_drifts_ppm",
        "clock_drifts", "emission_times", "emission_times_hint", "true_track",
        "true_track_times", "world",
    }
    assert fields.isdisjoint(forbidden)
    # Only the measurable boundary fields exist.
    assert fields == {
        "device_ids", "ranging", "acoustic", "anchor_gps",
        "speed_of_sound_mps", "sample_rate_hz",
    }
