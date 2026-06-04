"""Live / streaming mode: process emissions in time order and emit state snapshots.

This is the hardware-facing shape of the system — instead of batch-processing a whole
run, the :class:`~dronetracking.live.engine.StreamEngine` calibrates geometry/clocks
once and then consumes drone emissions one at a time (as they would arrive from real
devices), updating the tracks and yielding a :class:`Snapshot` per step. The web
dashboard (:mod:`dronetracking.live.server`) renders those snapshots live in a browser.
"""
