"""Estimation pipeline: consumes Observations, produces Estimates.

This package must never import `dronetracking.sim` (the ground-truth firewall).
Every stage returns an estimate *and* a covariance.
"""
