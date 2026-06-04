"""Synthetic world: owns ground truth, emits only measurable Observations.

Nothing in `dronetracking.estimation` may import from this package (the
ground-truth firewall, enforced by tests/test_no_truth_leak.py).
"""
