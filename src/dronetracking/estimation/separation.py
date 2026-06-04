"""Overlapping-source acoustic separation via a matched-filter BANK (Ph6 separation).

When several drones emit at once, each device's microphone records a MIX of their
signatures. If every drone uses a DISTINCT pulse — e.g. a chirp in a different
frequency band — a *bank* of matched filters separates them: correlating a device's
capture against source ``k``'s reference pulse responds strongly to source ``k`` and
largely rejects the others (near-orthogonal templates have a small cross-correlation,
so an out-of-band pulse barely lifts the wrong filter's response).

This module is a thin orchestration layer over :mod:`estimation.detection`: it runs the
SAME frozen matched-filter detector once per reference pulse and tags the resulting
arrivals with the source they came from. Reusing :func:`estimation.detection.detect_arrivals`
means the index->time convention, the band-pass guard, the peak-picking and the
confidence model are all shared — separation only adds the per-source labelling.

GROUND-TRUTH FIREWALL: this module lives under ``estimation`` and therefore must NOT
import ``dronetracking.sim`` (enforced by ``tests/test_no_truth_leak.py`` over the
top-level ``estimation/*.py`` files). Like :mod:`estimation.detection`, it returns its
OWN lightweight record — :class:`SeparatedArrival` — which is a :class:`detection.DetectedArrival`
plus a ``source`` tag. :func:`to_acoustic_arrivals` flattens these into the same kind of
light, source-tagged records (carrying ``device_id``, ``emission_idx``, ``toa_local_s``,
``source`` and ``confidence``). They are FIELD-COMPATIBLE with
:class:`sim.observations.AcousticArrival` by design, and the multi-target TDOA path only
duck-types on those attributes (it never constructs or ``isinstance``-checks the contract
type), so they flow straight through. The orchestrator can still trivially re-stamp them
as real ``AcousticArrival``\\ s OUTSIDE the firewall if it wants the contract type (the
field-for-field mapping is documented on :func:`to_acoustic_arrivals`).

Feeding the multi-target tracker
--------------------------------
The existing multi-target path (:func:`estimation.multi_target.localize_frames`) groups
arrivals by ``(emission_idx, source)`` to form clean, single-target TDOA solves. So the
orchestrator wiring is::

    captures = synthesize_captures(scenario, rng)           # sim
    refs = {k: reference_pulse_for_source(k) for k in ...}  # one template per drone
    separated = separate_arrivals(captures, refs, n_emissions, scenario.dt_s)
    arrivals = to_acoustic_arrivals(separated)              # carries the .source label
    frames = multi_target.localize_frames(arrivals, clocks, layout, c)
    tracks = multi_target.track_targets(frames)

Each source's arrivals become their own per-emission TDOA frame (one drone per frame),
exactly the un-mixed input ``localize_frames`` expects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, List, Mapping, Tuple

import numpy as np

from .detection import DetectedArrival, detect_arrivals


@dataclass(frozen=True)
class SeparatedArrival:
    """A detected arrival attributed to one source (firewall-clean, source-tagged).

    Identical to :class:`detection.DetectedArrival` but adds a ``source`` key naming
    which reference pulse (hence which drone) produced it. ``toa_local_s`` is in the
    recording device's own clock; ``confidence`` is in ``[0, 1]``. The ``source`` key is
    whatever the caller used in ``reference_pulses`` (a name, an int id, ...).
    """

    source: Hashable
    device_id: str
    emission_idx: int
    toa_local_s: float
    confidence: float = 1.0


@dataclass(frozen=True)
class AcousticArrivalLike:
    """Firewall-clean, field-compatible analogue of :class:`sim.observations.AcousticArrival`.

    Carries exactly the fields the multi-target TDOA path reads — ``device_id``,
    ``emission_idx``, ``toa_local_s``, ``source`` (an INTEGER target label) and
    ``confidence``. :func:`estimation.multi_target.localize_frames` groups arrivals by
    ``(emission_idx, source)`` and reads ``toa_local_s`` via duck typing; it never
    constructs or ``isinstance``-checks the contract type, so these records consume
    directly. The orchestrator may instead re-stamp them as real ``AcousticArrival``\\ s
    outside the firewall — the mapping is a plain field copy.
    """

    device_id: str
    emission_idx: int
    toa_local_s: float
    source: int = 0
    confidence: float = 1.0


def separate_arrivals(
    captures: Mapping[str, "object"],
    reference_pulses: Mapping[Hashable, np.ndarray],
    n_emissions: int,
    dt_s: float,
) -> Dict[Hashable, List[SeparatedArrival]]:
    """Separate overlapping sources by running a matched filter per reference pulse.

    For each source, the SAME matched-filter detector (:func:`detection.detect_arrivals`)
    is run over every capture with that source's reference pulse. Because the templates
    occupy distinct bands, source ``k``'s filter peaks on source ``k``'s pulses and
    rejects the others, so the ``n_emissions`` strongest peaks it picks are that source's
    arrivals. Each :class:`detection.DetectedArrival` is then re-stamped as a
    :class:`SeparatedArrival` carrying the source key.

    Parameters
    ----------
    captures
        ``{device_id: capture}`` where each capture exposes ``samples`` (1-D array),
        ``sample_rate_hz`` and ``t0_local_s`` (e.g. :class:`sim.audio.AudioCapture`, or
        any duck-typed object with those attributes — keeping this module ``sim``-free).
    reference_pulses
        ``{source_key: reference_pulse}`` — one known pulse template per source/drone.
        ``source_key`` may be any hashable (a string name, an int id, ...).
    n_emissions
        Number of emissions to recover per device PER source.
    dt_s
        Nominal seconds between a source's emissions (sets the peak-separation guard,
        forwarded to the detector).

    Returns
    -------
    dict
        ``{source_key: [SeparatedArrival, ...]}`` — for each source, ``n_emissions`` per
        device (``emission_idx`` in time order, matching :func:`detection.detect_arrivals`),
        every record tagged with that ``source_key``. Sources are processed in a stable,
        sorted order where the keys are sortable (falling back to insertion order).
    """
    # Stable iteration: sort keys when comparable, else preserve insertion order.
    try:
        source_keys = sorted(reference_pulses)
    except TypeError:
        source_keys = list(reference_pulses)

    separated: Dict[Hashable, List[SeparatedArrival]] = {}
    for source_key in source_keys:
        detected: Tuple[DetectedArrival, ...] = detect_arrivals(
            captures, reference_pulses[source_key], n_emissions=n_emissions, dt_s=dt_s
        )
        separated[source_key] = [
            SeparatedArrival(
                source=source_key,
                device_id=d.device_id,
                emission_idx=d.emission_idx,
                toa_local_s=d.toa_local_s,
                confidence=d.confidence,
            )
            for d in detected
        ]
    return separated


def to_acoustic_arrivals(
    separated: Mapping[Hashable, List[SeparatedArrival]],
) -> Tuple[AcousticArrivalLike, ...]:
    """Flatten a per-source separation into source-tagged, TDOA-ready arrivals.

    The multi-target TDOA path (:func:`estimation.multi_target.localize_frames`) groups
    arrivals by ``(emission_idx, source)`` to form one clean single-target solve per drone
    per emission. It expects an INTEGER ``source`` label on each arrival, so the (possibly
    non-integer) source keys are mapped to distinct integers ``0, 1, ...`` in the same
    stable order :func:`separate_arrivals` used (sorted keys where comparable, else
    insertion order). Two distinct source keys therefore get two distinct labels.

    Returns firewall-clean :class:`AcousticArrivalLike` records (no ``sim`` import). Each
    is FIELD-COMPATIBLE with :class:`sim.observations.AcousticArrival`, so the orchestrator
    may consume them as-is OR re-stamp them outside the firewall via a plain field copy::

        AcousticArrival(device_id    = a.device_id,
                        emission_idx = a.emission_idx,
                        toa_local_s  = a.toa_local_s,
                        source       = a.source,        # already the stable int label
                        confidence   = a.confidence)

    Returns
    -------
    tuple of :class:`AcousticArrivalLike`
        One per detection across all sources, ready to hand straight to
        :func:`estimation.multi_target.localize_frames`.
    """
    try:
        ordered_keys = sorted(separated)
    except TypeError:
        ordered_keys = list(separated)
    source_label = {key: idx for idx, key in enumerate(ordered_keys)}

    arrivals: List[AcousticArrivalLike] = []
    for key in ordered_keys:
        for s in separated[key]:
            arrivals.append(
                AcousticArrivalLike(
                    device_id=s.device_id,
                    emission_idx=s.emission_idx,
                    toa_local_s=s.toa_local_s,
                    source=source_label[key],
                    confidence=s.confidence,
                )
            )
    return tuple(arrivals)
