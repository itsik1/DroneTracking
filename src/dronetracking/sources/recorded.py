"""Recorded-audio device feed: ingest real per-device recordings from a directory.

:class:`RecordedAudioFeed` is the concrete bridge for **real recorded device data**. Where
:class:`~dronetracking.sources.simulated.SimulatedDeviceFeed` runs the simulator and
:class:`~dronetracking.sources.live.LiveDeviceFeed` is the (raising) skeleton for a live
network, this feed reads a *directory of recordings a real deployment produced* and turns
it into the standard :class:`~dronetracking.sim.observations.Observations` bundle the whole
estimation stack consumes — so the batch pipeline and streaming engine run on real captured
audio with no downstream change.

The acoustic half is genuinely *recovered*, not passed through: each device's waveform is
matched-filtered (:func:`estimation.detection.detect_arrivals`) against the known drone
pulse to recover per-emission arrival times in that device's local clock. The non-acoustic
measurements a microphone cannot supply — two-way-ranging timestamps, anchor GPS, and the
two timebase constants — are carried in a sidecar ``meta.json``.

On-disk layout (see :func:`write_recorded_dataset` for the writer)::

    {dir}/
      meta.json            # ranging + anchor GPS + timebase + detection params
      {device_id}.wav      # one mono WAV per acoustic device (float32 or int PCM)

``meta.json`` schema (version 1)::

    {
      "version": 1,
      "device_ids": ["dev0", "dev1", ...],   # the stable downstream device order
      "speed_of_sound_mps": 343.0,
      "sample_rate_hz": 16000.0,
      "n_emissions": 9,                        # drone emissions to recover per device
      "dt_s": 2.0,                             # nominal spacing (peak-separation guard)
      "reference": {                           # reference-pulse params (sim.audio keys)
          "pulse": "chirp", "f0": 1500.0, "f1": 5500.0, "pulse_dur_s": 0.02, ...
      },
      "t0_local_s": {"dev0": 0.0, ...},        # local-clock time of WAV sample 0 (per device)
      "audio_files": {"dev0": "dev0.wav", ...},# optional explicit filenames (default {id}.wav)
      "batches": [ <protocol batch dict>, ... ]# one per device: ranging + anchor_gps + timebase
    }

The ``batches`` entries are exactly the JSON objects the live wire protocol
(:mod:`dronetracking.live.protocol`) emits — one per device, carrying that device's ranging
exchanges, anchor GPS, and the timebase constants (the ``acoustic`` list is empty there
because arrivals are recovered from the WAV). So a real coordinator can persist each
:class:`~dronetracking.live.agent.DeviceAgent`'s published batch verbatim, drop the
recorded WAV beside it, and replay the whole session offline through this feed.

This package is a sim *adapter* layer (not estimation), so it MAY import ``sim``; reusing
:func:`sim.audio.reference_pulse` keeps the matched-filter template bit-identical to the one
that generated/recorded the audio.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.io import wavfile

from ..estimation.detection import detect_arrivals
from ..live import protocol
from ..sim.audio import reference_pulse
from ..sim.observations import AcousticArrival, AnchorGps, Observations, RangingRecord
from ..sim.scenario import Scenario
from .base import DeviceFeed

META_FILENAME = "meta.json"
META_VERSION = 1


# --------------------------------------------------------------------------- #
# a tiny duck-typed capture: just what estimation.detection.detect_arrivals reads
# --------------------------------------------------------------------------- #
class _RecordedCapture:
    """A WAV-backed stand-in for :class:`sim.audio.AudioCapture`.

    The detector only reads ``.samples`` (1-D float array), ``.sample_rate_hz`` and
    ``.t0_local_s``, so this thin object keeps :mod:`recorded` free of any dependency on
    the simulator's ``AudioCapture`` while feeding the same matched-filter detection path.
    """

    __slots__ = ("device_id", "samples", "sample_rate_hz", "t0_local_s")

    def __init__(self, device_id: str, samples: np.ndarray, sample_rate_hz: float, t0_local_s: float):
        self.device_id = device_id
        self.samples = np.asarray(samples, dtype=float)
        self.sample_rate_hz = float(sample_rate_hz)
        self.t0_local_s = float(t0_local_s)


# --------------------------------------------------------------------------- #
# the feed
# --------------------------------------------------------------------------- #
class RecordedAudioFeed(DeviceFeed):
    """A :class:`DeviceFeed` assembled from on-disk per-device recordings + a meta sidecar.

    Construction is cheap and does no I/O of the audio; the work happens lazily on the
    first accessor call (then it is cached), so building the feed never fails for a missing
    file until you actually read from it. Reads are idempotent and identical thereafter.
    """

    def __init__(self, directory, *, reference_pulse_override: Optional[np.ndarray] = None) -> None:
        """Args:
        directory: path to the recording directory (must contain ``meta.json`` + WAVs).
        reference_pulse_override: optional explicit matched-filter template. If omitted,
            the template is rebuilt from ``meta['reference']`` via
            :func:`sim.audio.reference_pulse` (bit-identical to the recording's pulse).
        """
        self.directory = Path(directory)
        self._reference_override = (
            None if reference_pulse_override is None
            else np.asarray(reference_pulse_override, dtype=float)
        )

        # Lazily populated by _load() and cached.
        self._meta: Optional[dict] = None
        self._device_ids: Tuple[str, ...] = ()
        self._ranging: Tuple[RangingRecord, ...] = ()
        self._anchor_gps: Tuple[AnchorGps, ...] = ()
        self._acoustic: Tuple[AcousticArrival, ...] = ()
        self._speed_of_sound_mps: float = 0.0
        self._sample_rate_hz: float = 0.0
        self._loaded = False

    # -- loading ----------------------------------------------------------- #
    def _load(self) -> None:
        """Read meta.json + every device WAV, run detection, assemble the streams (once)."""
        if self._loaded:
            return

        meta_path = self.directory / META_FILENAME
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"RecordedAudioFeed: no {META_FILENAME} in {self.directory!s}. A recording "
                f"directory must contain {META_FILENAME} plus one WAV per acoustic device."
            )
        meta = json.loads(meta_path.read_text())
        self._meta = meta

        # Device order is authoritative from meta (fixes downstream matrix row order).
        device_ids = tuple(str(d) for d in meta["device_ids"])
        self._speed_of_sound_mps = float(meta["speed_of_sound_mps"])
        self._sample_rate_hz = float(meta["sample_rate_hz"])

        # Non-acoustic measurements: decode the per-device protocol batches (ranging +
        # anchor GPS + timebase). Reusing protocol.decode_batch reconstructs the frozen
        # contract dataclasses exactly, identical to how the live coordinator assembles them.
        ranging: list[RangingRecord] = []
        anchor_gps: list[AnchorGps] = []
        for batch_obj in meta.get("batches", []):
            decoded = protocol.decode_batch(json.dumps(batch_obj))
            ranging.extend(decoded["ranging"])
            anchor_gps.extend(decoded["anchor_gps"])
        self._ranging = tuple(ranging)
        self._anchor_gps = tuple(anchor_gps)

        # Acoustic: matched-filter detection on each device's recorded waveform.
        self._acoustic = self._detect_from_wavs(meta, device_ids)
        self._device_ids = device_ids
        self._loaded = True

    def _reference_template(self, meta: dict) -> np.ndarray:
        """The matched-filter template: an explicit override, else rebuilt from meta params."""
        if self._reference_override is not None:
            return self._reference_override
        # Reconstruct the exact reference pulse from the stored params. reference_pulse only
        # consults scenario.sample_rate_hz + scenario.audio, so a minimal stub suffices and
        # the template is bit-identical to the one that synthesized/templated the recording.
        stub = _reference_scenario(meta["sample_rate_hz"], meta.get("reference", {}))
        return reference_pulse(stub)

    def _detect_from_wavs(self, meta: dict, device_ids: Tuple[str, ...]) -> Tuple[AcousticArrival, ...]:
        """Load WAVs, run :func:`detect_arrivals`, map detections -> ``AcousticArrival``."""
        template = self._reference_template(meta)
        n_emissions = int(meta["n_emissions"])
        dt_s = float(meta["dt_s"])
        fs_meta = float(meta["sample_rate_hz"])

        t0_map = meta.get("t0_local_s", {})
        audio_files = meta.get("audio_files", {})

        captures: Dict[str, _RecordedCapture] = {}
        for did in device_ids:
            wav_name = audio_files.get(did, f"{did}.wav")
            wav_path = self.directory / wav_name
            if not wav_path.is_file():
                # A device with no recorded audio (e.g. no microphone) simply contributes
                # no acoustic arrivals — skip it rather than failing the whole load.
                continue
            fs_wav, samples = wavfile.read(wav_path)
            samples = _to_float_mono(samples)
            # Prefer the WAV's own sample rate; fall back to meta if absent (always present).
            fs = float(fs_wav) if fs_wav else fs_meta
            t0 = float(t0_map.get(did, 0.0))
            captures[did] = _RecordedCapture(did, samples, fs, t0)

        if not captures:
            return ()

        detected = detect_arrivals(captures, template, n_emissions=n_emissions, dt_s=dt_s)
        # detect_arrivals is firewall-clean (its own DetectedArrival); map to the contract
        # AcousticArrival the rest of the system expects. Recordings are single-target
        # (source=0), matching sim.audio.synthesize_captures / pipeline._detect_arrivals_into.
        return tuple(
            AcousticArrival(
                device_id=d.device_id,
                emission_idx=d.emission_idx,
                toa_local_s=d.toa_local_s,
                source=0,
                confidence=getattr(d, "confidence", 1.0),
            )
            for d in detected
        )

    # -- DeviceFeed measurement surface (served from the cached load) ------- #
    def device_ids(self) -> Tuple[str, ...]:
        self._load()
        return self._device_ids

    def ranging_records(self) -> Tuple[RangingRecord, ...]:
        self._load()
        return self._ranging

    def acoustic_arrivals(self) -> Tuple[AcousticArrival, ...]:
        self._load()
        return self._acoustic

    def anchor_gps(self) -> Tuple[AnchorGps, ...]:
        self._load()
        return self._anchor_gps

    @property
    def speed_of_sound_mps(self) -> float:
        self._load()
        return self._speed_of_sound_mps

    @property
    def sample_rate_hz(self) -> float:
        self._load()
        return self._sample_rate_hz


# --------------------------------------------------------------------------- #
# writer: lay a recording (or a synthesized stand-in) down in the on-disk layout
# --------------------------------------------------------------------------- #
def write_recorded_dataset(
    directory,
    feed,
    captures: Dict[str, "object"],
    scenario: Scenario,
    *,
    wav_dtype=np.float32,
) -> Path:
    """Write a :class:`RecordedAudioFeed`-readable dataset (WAVs + meta.json) to ``directory``.

    This is the inverse of :meth:`RecordedAudioFeed._load` and the bridge tests/tools use to
    produce a recording from a :class:`~dronetracking.sources.simulated.SimulatedDeviceFeed`
    (or, on real hardware, from captured audio + the agents' published batches).

    Args:
        directory: target directory (created if absent).
        feed: a feed whose ``ranging_records()`` / ``anchor_gps()`` / timebase populate the
            non-acoustic half of meta (e.g. a ``SimulatedDeviceFeed``). Its ranging/anchor
            are partitioned per device and encoded with :mod:`live.protocol`, exactly as an
            on-device :class:`~dronetracking.live.agent.DeviceAgent` would publish them.
        captures: ``{device_id: capture}`` waveforms to write as WAVs; each capture needs
            ``samples`` (1-D array), ``sample_rate_hz`` and ``t0_local_s`` (e.g.
            :class:`sim.audio.AudioCapture`).
        scenario: source of ``n_emissions`` (= len(emission_times)), ``dt_s`` and the
            reference-pulse (``scenario.audio``) params persisted to meta.
        wav_dtype: sample dtype for the written WAVs (``np.float32`` by default; integer
            dtypes are scaled to full range). The feed reads any of these back as float.

    Returns:
        The path to the written ``meta.json``.
    """
    from ..sim.acoustic import emission_times  # local import: writer-only, avoids cycles

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    obs = feed.as_observations()
    device_ids = list(obs.device_ids)

    # Per-device protocol batches: ranging the device initiated + its own GPS + timebase.
    # (Acoustic is intentionally empty on the wire — recovered from the WAV on read.)
    batches = []
    for did in device_ids:
        ranging = [r for r in obs.ranging if r.initiator == did]
        anchor_gps = [g for g in obs.anchor_gps if g.device_id == did]
        line = protocol.encode_batch(
            device_id=did,
            ranging=ranging,
            acoustic=(),
            anchor_gps=anchor_gps,
            speed_of_sound_mps=obs.speed_of_sound_mps,
            sample_rate_hz=obs.sample_rate_hz,
        )
        batches.append(json.loads(line.decode("utf-8")))

    # Write one WAV per device that has a capture, and record its t0 + filename.
    t0_local_s: Dict[str, float] = {}
    audio_files: Dict[str, str] = {}
    for did in device_ids:
        cap = captures.get(did)
        if cap is None:
            continue
        fs = int(round(float(cap.sample_rate_hz)))
        samples = _encode_wav_samples(np.asarray(cap.samples, dtype=float), wav_dtype)
        fname = f"{did}.wav"
        wavfile.write(directory / fname, fs, samples)
        t0_local_s[did] = float(cap.t0_local_s)
        audio_files[did] = fname

    meta = {
        "version": META_VERSION,
        "device_ids": device_ids,
        "speed_of_sound_mps": float(obs.speed_of_sound_mps),
        "sample_rate_hz": float(obs.sample_rate_hz),
        "n_emissions": int(len(emission_times(scenario))),
        "dt_s": float(scenario.dt_s),
        "reference": dict(scenario.audio or {}),
        "t0_local_s": t0_local_s,
        "audio_files": audio_files,
        "batches": batches,
    }
    meta_path = directory / META_FILENAME
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta_path


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _reference_scenario(sample_rate_hz, audio_params: dict) -> Scenario:
    """A minimal :class:`Scenario` carrying only what :func:`reference_pulse` reads.

    ``reference_pulse`` consults exactly ``scenario.sample_rate_hz`` and ``scenario.audio``;
    every other field is a harmless placeholder. Built so the matched-filter template is
    reconstructed bit-for-bit from the persisted params without needing the full scenario.
    """
    from ..sim.scenario import DeviceSpec, TrajectorySpec

    return Scenario(
        name="_recorded_reference",
        seed=0,
        speed_of_sound_mps=343.0,
        sample_rate_hz=float(sample_rate_hz),
        duration_s=1.0,
        dt_s=1.0,
        ranging_rounds=1,
        origin_latlon=(0.0, 0.0),
        devices=(DeviceSpec(id="_ref", position_m=(0.0, 0.0, 0.0)),),
        trajectory=TrajectorySpec(kind="linear", params={"start_m": (0.0, 0.0), "end_m": (1.0, 0.0)}),
        audio=dict(audio_params or {}),
    )


def _to_float_mono(samples: np.ndarray) -> np.ndarray:
    """Coerce a WAV array to a 1-D float waveform.

    Stereo/multi-channel input is averaged to mono; integer PCM is normalized to ~[-1, 1]
    by its dtype's full-scale; float input passes through. Matched filtering is amplitude-
    scale invariant for peak *location*, so exact gain does not matter — but normalizing
    keeps confidence/threshold behaviour comparable across PCM and float recordings.
    """
    arr = np.asarray(samples)
    if arr.ndim > 1:  # average channels to mono
        arr = arr.mean(axis=1)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        # Scale by the larger magnitude bound so the result sits in ~[-1, 1].
        scale = float(max(abs(info.min), abs(info.max)))
        return arr.astype(np.float64) / (scale if scale else 1.0)
    return arr.astype(np.float64)


def _encode_wav_samples(samples: np.ndarray, wav_dtype) -> np.ndarray:
    """Convert a float waveform to ``wav_dtype`` for writing.

    Float dtypes are written as-is (cast). Integer dtypes are scaled from ~[-1, 1] to the
    dtype's full range and clipped, the inverse of :func:`_to_float_mono`.
    """
    wav_dtype = np.dtype(wav_dtype)
    if np.issubdtype(wav_dtype, np.integer):
        info = np.iinfo(wav_dtype)
        scale = float(max(abs(info.min), abs(info.max)))
        scaled = np.clip(samples * scale, info.min, info.max)
        return scaled.astype(wav_dtype)
    return samples.astype(wav_dtype)
