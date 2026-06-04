"""Human- and machine-readable rendering of a metrics dict.

:func:`print_report` writes a grouped, unit-annotated scorecard to stdout.
:func:`save_report` dumps the raw flat dict to JSON for diffing across runs.

The metrics dict is flat with ``"<group>.<name>"`` keys (see
:mod:`dronetracking.eval.metrics`); this module groups by the prefix purely for
display and does not depend on which specific metrics are present.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple, Union

# Display order and the unit suffix shown for each known group.
_GROUP_ORDER = (
    "scenario",
    "device_localization",
    "clock_sync",
    "tracking",
    "georeferencing",
)

_GROUP_TITLES = {
    "scenario": "Scenario",
    "device_localization": "Device localization",
    "clock_sync": "Clock synchronization",
    "tracking": "Tracking",
    "georeferencing": "Georeferencing",
}


def _split_key(key: str) -> Tuple[str, str]:
    group, _, rest = key.partition(".")
    return group, (rest or group)


def _fmt_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.6g}"
    return str(value)


def _grouped(metrics: Dict[str, Any]) -> "Dict[str, list]":
    groups: Dict[str, list] = {}
    for key, value in metrics.items():
        group, rest = _split_key(key)
        groups.setdefault(group, []).append((rest, value))
    return groups


def print_report(metrics: Dict[str, Any]) -> None:
    """Print a formatted, grouped scorecard (units in the key names) to stdout."""
    groups = _grouped(metrics)
    ordered = list(_GROUP_ORDER) + [g for g in groups if g not in _GROUP_ORDER]

    lines = ["=" * 56, "EVALUATION REPORT", "=" * 56]
    for group in ordered:
        if group not in groups:
            continue
        lines.append("")
        lines.append(_GROUP_TITLES.get(group, group.replace("_", " ").title()))
        lines.append("-" * 56)
        for rest, value in groups[group]:
            lines.append(f"  {rest:<32} {_fmt_value(value)}")
    lines.append("=" * 56)
    print("\n".join(lines))


def save_report(metrics: Dict[str, Any], path: Union[str, Path]) -> Path:
    """Write ``metrics`` to ``path`` as JSON. Returns the path written.

    Uses the standard library's ``NaN``/``Infinity`` extension so degraded-run
    fields survive a Python ``json`` round-trip without being silently dropped.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)
    return path
