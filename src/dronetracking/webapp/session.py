"""Adaptive session — the *brain* of the zero-install browser app.

A :class:`Session` ingests whatever the connected phones can report (a GPS fix,
a microphone level, both, or neither) and turns it into the richest possible
state snapshot for the map UI. It is intentionally **capability-adaptive**: it
never demands a capability, it just does more when more is available.

The headline method is :func:`Session.state`, which emits exactly the
``/api/events`` JSON documented in ``docs/webapp_contract.md``.

Two derived quantities are computed on demand:

**Positioning.** A device with a GPS fix is placed at its lat/lon
(``computed.positioning == "gps"``). With no fix its position is ``null``.

**Relative localization by acoustic ranging (GPS-free).** Devices measure their
pairwise *distances* acoustically (SDS-TWR; see :mod:`dronetracking.webapp.ranging`).
The session owns a :class:`~dronetracking.webapp.ranging.RangingCoordinator` that
schedules rounds and accumulates per-pair distances. :func:`Session.state` then
emits, on top of the energy-source fields:

* ``command`` — the current ranging instruction (which ordered pair should chirp
  next, and with what chirp), or ``None``;
* ``distances`` — the latest measured pairwise distances ``[{a, b, m}]``;
* ``relative`` — once ``>= 3`` online devices share a (near-)complete distance
  set, the recovered relative layout ``{device_ids, xy_m}`` from the MDS pipeline
  (:func:`estimation.relative_localization.estimate_layout`), centered and (when
  GPS devices are present) rigidly aligned to them via :func:`transforms.umeyama`.
  For exactly ``2`` devices with a measured distance, ``relative`` is ``None`` and
  the single distance appears in ``distances``.

**Source localization by acoustic energy (sync-free).** This is the key method
and needs *no* clock synchronization — only received levels and positions. For a
point emitter the received power falls off as ``P_i = A / (r_i**2 + eps)`` where
``r_i`` is the source-to-device distance and ``A`` an amplitude. Since the
browser reports a linear RMS amplitude ``level_i`` in ``[0, 1]``, the received
*power* is ``level_i**2`` and the model is::

    level_i**2  ~  A / (r_i**2 + eps)

Among the devices that are currently *detecting* **and** have a position, we
project their GPS to a local ENU frame about their centroid
(:func:`geo.latlon_to_enu`) and solve for the source ``(x, y)`` and amplitude
``A`` by nonlinear least squares (:func:`scipy.optimize.least_squares`) on the
power residuals. The ENU solution is converted back to lat/lon
(:func:`geo.enu_to_latlon`).

* ``>= 3`` detecting+positioned devices -> a point fix (``source == "energy"``).
* exactly ``2`` -> a coarse level-weighted midpoint / region (``"region"``);
  range from a single power ratio is ambiguous, so we only claim a region.
* otherwise -> ``source is None`` (``"none"``).

The clock is injectable (``time_fn``) so last-seen / pruning is deterministic in
tests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from .. import geo, transforms
from ..estimation.relative_localization import estimate_layout
from .ranging import RangingCoordinator

# Power-model softening: P = A / (r**2 + EPS). Keeps the model finite when a
# device sits essentially on top of the source. Small vs. field-scale r**2.
EPS = 1e-6

# A device is "online" if it reported within this window (seconds). Mirrors the
# contract's ~5 s staleness rule and is also the prune default.
ONLINE_WINDOW_S = 5.0


@dataclass
class _Device:
    """Mutable per-device record kept by the session."""

    device_id: str
    name: str
    last_seen: float
    last_gps: Optional[dict] = None  # {"lat", "lon", "accuracy_m"} or None
    last_audio: Optional[dict] = None  # {"level", "detected", "confidence", "peak_hz"}
    has_gps: bool = False  # latches True once any non-null gps is seen
    has_mic: bool = False  # latches True once any non-null audio is seen

    # --- convenience views over the latest audio/gps --------------------- #
    @property
    def level(self) -> float:
        if self.last_audio is None:
            return 0.0
        return float(self.last_audio.get("level", 0.0) or 0.0)

    @property
    def detected(self) -> bool:
        if self.last_audio is None:
            return False
        return bool(self.last_audio.get("detected", False))

    @property
    def confidence(self) -> float:
        if self.last_audio is None:
            return 0.0
        return float(self.last_audio.get("confidence", 0.0) or 0.0)

    @property
    def latlon(self) -> Optional[Tuple[float, float]]:
        if self.last_gps is None:
            return None
        lat = self.last_gps.get("lat")
        lon = self.last_gps.get("lon")
        if lat is None or lon is None:
            return None
        return float(lat), float(lon)


@dataclass
class _SourceFit:
    """Internal result of source localization, including the ``computed.source`` tag."""

    source: Optional[dict]  # {"lat","lon","confidence","error_m"} or None
    kind: str  # "energy" | "region" | "none"


class Session:
    """Tracks connected devices and computes the adaptive state snapshot.

    Parameters
    ----------
    time_fn:
        A zero-arg callable returning a monotonically increasing time in
        seconds. Defaults to :func:`time.monotonic`. Inject a controllable clock
        in tests to make last-seen / pruning deterministic.
    """

    def __init__(
        self,
        time_fn: Callable[[], float] = time.monotonic,
        debug: bool = False,
    ) -> None:
        self._time_fn = time_fn
        self._devices: Dict[str, _Device] = {}
        # Acoustic-ranging brain: schedules SDS-TWR rounds across online pairs and
        # accumulates the pairwise distances reported back by devices. `debug` makes
        # it log each stored half and each finalized round (raw distance + verdict).
        self._ranging = RangingCoordinator(debug=debug)

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #
    def upsert_device(self, device_id: str, name: Optional[str]) -> None:
        """Register a device, or rename it if it already exists.

        Upserting refreshes last-seen so a freshly-joined device counts as
        online immediately, before it has POSTed any report.
        """
        now = self._time_fn()
        dev = self._devices.get(device_id)
        resolved = self._resolve_name(device_id, name)
        if dev is None:
            self._devices[device_id] = _Device(
                device_id=device_id, name=resolved, last_seen=now
            )
        else:
            # Only overwrite the name when a non-empty one is supplied.
            if name:
                dev.name = resolved
            dev.last_seen = now

    def report(self, device_id: str, payload: dict) -> None:
        """Apply a ``/api/report`` body for ``device_id``.

        Body shape (per the contract)::

            {"t_client_ms": int,
             "gps": {"lat","lon","accuracy_m"} | None,
             "audio": {"level","detected","confidence","peak_hz"} | None}

        Unknown devices are auto-registered (the server normally joins first,
        but we stay robust to out-of-order calls). ``has_gps`` / ``has_mic``
        latch True the first time a non-null gps / audio is observed and never
        flip back, so a momentary dropout doesn't erase a known capability.
        """
        now = self._time_fn()
        dev = self._devices.get(device_id)
        if dev is None:
            dev = _Device(
                device_id=device_id,
                name=self._resolve_name(device_id, None),
                last_seen=now,
            )
            self._devices[device_id] = dev

        dev.last_seen = now

        gps = payload.get("gps")
        if gps is not None:
            dev.last_gps = gps
            dev.has_gps = True

        audio = payload.get("audio")
        if audio is not None:
            dev.last_audio = audio
            dev.has_mic = True

        # Acoustic-ranging half-exchanges (optional). The coordinator pairs the
        # two halves of a round and turns completed rounds into distances.
        ranging = payload.get("ranging")
        if ranging:
            try:
                self._ranging.submit(device_id, list(ranging))
            except Exception:
                # A malformed ranging payload must never break a report — the
                # coordinator already skips bad entries; this guards the rest.
                pass

    def prune(self, max_age_s: float = 5.0) -> None:
        """Drop devices whose last report is older than ``max_age_s`` seconds."""
        now = self._time_fn()
        stale = [
            did
            for did, dev in self._devices.items()
            if (now - dev.last_seen) > max_age_s
        ]
        for did in stale:
            del self._devices[did]

    # ------------------------------------------------------------------ #
    # Snapshot
    # ------------------------------------------------------------------ #
    def state(self) -> dict:
        """Return the ``/api/events`` snapshot (see module docstring / contract)."""
        now = self._time_fn()
        devices = list(self._devices.values())

        device_views: List[dict] = []
        for dev in devices:
            online = (now - dev.last_seen) <= ONLINE_WINDOW_S
            ll = dev.latlon
            device_views.append(
                {
                    "id": dev.device_id,
                    "name": dev.name,
                    "lat": (ll[0] if ll is not None else None),
                    "lon": (ll[1] if ll is not None else None),
                    "has_gps": dev.has_gps,
                    "has_mic": dev.has_mic,
                    "level": dev.level,
                    "detected": dev.detected,
                    "confidence": dev.confidence,
                    "online": online,
                }
            )

        online_devices = [
            dev for dev in devices if (now - dev.last_seen) <= ONLINE_WINDOW_S
        ]

        positioning = self._positioning_mode(online_devices)
        fit = self._localize_source(online_devices)

        n_devices = len(devices)
        n_gps = sum(1 for d in online_devices if d.latlon is not None)
        # Raw detection count (a status, parallel to n_gps). Position-gating is a
        # concern of the localization algorithm only, not the network summary.
        n_detecting = sum(1 for d in online_devices if d.detected)
        n_online = len(online_devices)

        network = {
            "n_devices": n_devices,
            "n_gps": n_gps,
            "n_detecting": n_detecting,
            "connected": n_online >= 1,
        }

        computed = {"positioning": positioning, "source": fit.kind}
        note = self._note(network, positioning, fit)

        # --- acoustic ranging: command, measured distances, relative layout --- #
        online_ids = [d.device_id for d in online_devices]
        command = {"ranging": self._ranging.current_command(online_ids, now)}
        distances = self._ranging.distances()
        relative = self._relative_layout(online_devices)

        return {
            "devices": device_views,
            "source": fit.source,
            "network": network,
            "computed": computed,
            "note": note,
            "command": command,
            "distances": distances,
            "relative": relative,
        }

    # ------------------------------------------------------------------ #
    # Derived: relative localization from acoustic ranging
    # ------------------------------------------------------------------ #
    def _relative_layout(self, online_devices: List[_Device]) -> Optional[dict]:
        """Recover a 2-D relative layout from measured pairwise distances.

        Returns ``{"device_ids": [...], "xy_m": [[x, y], ...]}`` (centered;
        arbitrary rotation unless GPS-aligned) once ``>= 3`` online devices share
        a near-complete distance set, else ``None``. For exactly 2 devices the
        single distance is surfaced via ``state()["distances"]`` instead, so no
        layout is emitted here.

        Selection: keep online devices that are each connected (by a measured
        distance) to at least two *other kept* devices, so every emitted node is
        constrained in the plane. The MDS pipeline
        (:func:`estimation.relative_localization.estimate_layout`) then turns the
        sub-:class:`DistanceMatrix` into a 3-D embedding; we take its ``x, y``.
        When some kept devices have a GPS fix we rigidly align (Umeyama, no
        scaling) the relative cloud onto their ENU positions so the layout shares
        the GPS frame; otherwise we emit the bare centered layout.
        """
        online_ids = [d.device_id for d in online_devices]
        if len(online_ids) < 3:
            return None

        # Iteratively drop devices with < 2 in-subset measured edges until the
        # kept set is stable (a node with one edge is unconstrained in 2-D).
        kept = list(online_ids)
        while True:
            dm = self._ranging.distance_matrix(kept)
            degree = (np.sum(dm.valid, axis=1) - 1).astype(int)  # exclude diagonal
            survivors = [did for did, deg in zip(kept, degree) if deg >= 2]
            if len(survivors) == len(kept):
                break
            kept = survivors
            if len(kept) < 3:
                return None

        dm = self._ranging.distance_matrix(kept)
        # Need enough measured edges for a non-degenerate 2-D constellation.
        if dm.n_valid_edges < 3:
            return None

        try:
            layout = estimate_layout(dm)
        except Exception:
            return None

        xy = np.asarray(layout.positions_local, dtype=float)[:, :2]
        xy = xy - xy.mean(axis=0, keepdims=True)  # center
        if not np.all(np.isfinite(xy)):
            return None

        xy = self._maybe_align_to_gps(kept, xy)

        return {
            "device_ids": list(kept),
            "xy_m": [[float(x), float(y)] for x, y in xy],
        }

    def _maybe_align_to_gps(self, ids: List[str], xy: np.ndarray) -> np.ndarray:
        """Rigidly align the relative cloud onto any GPS fixes among ``ids``.

        Devices with a lat/lon are projected to a shared ENU frame about their
        centroid; :func:`transforms.umeyama` (rotation + translation, reflection
        allowed since distances are chirality-blind, no scaling — ranging already
        sets the metric scale) maps the relative ``xy`` onto those ENU points. The
        whole cloud is transformed so GPS-denied devices land in the GPS frame
        too. With fewer than two GPS anchors the bare relative layout is returned.
        """
        gps_idx = []
        gps_ll = []
        for k, did in enumerate(ids):
            dev = self._devices.get(did)
            ll = dev.latlon if dev is not None else None
            if ll is not None:
                gps_idx.append(k)
                gps_ll.append(ll)
        if len(gps_idx) < 2:
            return xy

        lats = np.array([ll[0] for ll in gps_ll], dtype=float)
        lons = np.array([ll[1] for ll in gps_ll], dtype=float)
        origin = (float(lats.mean()), float(lons.mean()))
        east, north = geo.latlon_to_enu(lats, lons, origin)
        dst = np.column_stack(
            [np.asarray(east, dtype=float), np.asarray(north, dtype=float)]
        )
        src = xy[gps_idx, :]
        try:
            sim = transforms.umeyama(
                src, dst, with_scaling=False, allow_reflection=True
            )
            aligned = sim.apply(xy)
        except Exception:
            return xy
        if not np.all(np.isfinite(aligned)):
            return xy
        return np.asarray(aligned, dtype=float)

    # ------------------------------------------------------------------ #
    # Derived: positioning
    # ------------------------------------------------------------------ #
    @staticmethod
    def _positioning_mode(online_devices: List[_Device]) -> str:
        if any(d.latlon is not None for d in online_devices):
            return "gps"
        return "none"

    # ------------------------------------------------------------------ #
    # Derived: source localization by acoustic energy
    # ------------------------------------------------------------------ #
    def _localize_source(self, online_devices: List[_Device]) -> _SourceFit:
        """Localize the source from detecting+positioned online devices.

        >=3 -> nonlinear-least-squares point fix on power residuals;
         2  -> level-weighted region midpoint;
         else -> none.
        """
        usable = [
            d for d in online_devices if d.detected and d.latlon is not None
        ]
        n = len(usable)
        if n == 0 or n == 1:
            return _SourceFit(source=None, kind="none")

        # Local ENU frame about the centroid of the detecting devices.
        lats = np.array([d.latlon[0] for d in usable], dtype=float)
        lons = np.array([d.latlon[1] for d in usable], dtype=float)
        origin = (float(lats.mean()), float(lons.mean()))
        east, north = geo.latlon_to_enu(lats, lons, origin)
        east = np.asarray(east, dtype=float)
        north = np.asarray(north, dtype=float)
        pos = np.column_stack([east, north])  # (n, 2)

        levels = np.array([max(d.level, 0.0) for d in usable], dtype=float)
        confidences = np.array([d.confidence for d in usable], dtype=float)

        if n == 2:
            return self._region_fit(pos, levels, confidences, origin)
        return self._energy_fit(pos, levels, confidences, origin)

    @staticmethod
    def _level_weighted_centroid(
        pos: np.ndarray, levels: np.ndarray
    ) -> np.ndarray:
        """A loudness-weighted centroid in ENU — the init / region estimate.

        Louder (closer) devices pull the point toward themselves, which is the
        right bias since received level rises as the source nears.
        """
        w = levels.astype(float).copy()
        if not np.any(w > 0):
            w = np.ones_like(w)
        w = w / w.sum()
        return (pos * w[:, None]).sum(axis=0)

    def _region_fit(
        self,
        pos: np.ndarray,
        levels: np.ndarray,
        confidences: np.ndarray,
        origin: Tuple[float, float],
    ) -> _SourceFit:
        """Two-device fallback: a coarse level-weighted region midpoint.

        With only two power measurements the source's range is ambiguous (a
        single power ratio fixes only a ratio of distances), so we don't claim a
        point — just a weighted midpoint biased toward the louder device, with a
        deliberately coarse error equal to roughly the device separation.
        """
        center = self._level_weighted_centroid(pos, levels)
        lat, lon = geo.enu_to_latlon(float(center[0]), float(center[1]), origin)
        separation = float(np.linalg.norm(pos[0] - pos[1]))
        # Coarse: the source could be anywhere across an order-1 fraction of the
        # baseline. Half the separation is an honest, conservative spread.
        error_m = max(separation * 0.5, 1.0)
        confidence = 0.25 * float(np.clip(np.mean(confidences) if confidences.size else 0.0, 0.0, 1.0))
        source = {
            "lat": float(lat),
            "lon": float(lon),
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "error_m": float(error_m),
        }
        return _SourceFit(source=source, kind="region")

    def _energy_fit(
        self,
        pos: np.ndarray,
        levels: np.ndarray,
        confidences: np.ndarray,
        origin: Tuple[float, float],
    ) -> _SourceFit:
        """>=3 devices: solve (x, y, A) by NLS on power residuals P = A/(r^2+eps)."""
        power = levels ** 2  # received power proxy

        # --- initial guess ------------------------------------------------ #
        # Position: level-weighted centroid (louder -> closer). Amplitude: from
        # the loudest device, A0 = P_max * (r0^2 + eps) using its distance to
        # the init point so the scale is roughly right.
        xy0 = self._level_weighted_centroid(pos, levels)
        loud_idx = int(np.argmax(power))
        r0_loud_sq = float(np.sum((pos[loud_idx] - xy0) ** 2))
        A0 = float(power[loud_idx]) * (r0_loud_sq + EPS)
        if not np.isfinite(A0) or A0 <= 0:
            A0 = float(np.max(power)) if np.max(power) > 0 else 1.0
        p0 = np.array([xy0[0], xy0[1], A0], dtype=float)

        def residuals(params: np.ndarray) -> np.ndarray:
            x, y, A = params
            r_sq = (pos[:, 0] - x) ** 2 + (pos[:, 1] - y) ** 2
            model = A / (r_sq + EPS)
            return model - power

        # Amplitude is physically non-negative; (x, y) are free.
        lo = np.array([-np.inf, -np.inf, 0.0])
        hi = np.array([np.inf, np.inf, np.inf])
        try:
            res = least_squares(
                residuals, p0, bounds=(lo, hi), method="trf", max_nfev=200
            )
            x, y, A = res.x
            resid = res.fun
        except Exception:
            # Degenerate geometry — fall back to the weighted centroid as a
            # region-grade answer rather than failing the whole snapshot.
            return self._region_fit(pos, levels, confidences, origin)

        lat, lon = geo.enu_to_latlon(float(x), float(y), origin)

        # --- error estimate ---------------------------------------------- #
        # Map the RMS fractional power residual to a distance scale. Near the
        # source dP/dr ~ -2A/r^3, so dr ~ (r/2) * (dP/P). We use a characteristic
        # radius (median source-device distance) and the RMS fractional residual.
        r = np.sqrt((pos[:, 0] - x) ** 2 + (pos[:, 1] - y) ** 2)
        model = A / (r ** 2 + EPS)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(model > 0, np.abs(resid) / model, 0.0)
        rms_frac = float(np.sqrt(np.mean(frac ** 2))) if frac.size else 0.0
        r_char = float(np.median(r)) if r.size else 0.0
        error_m = max(0.5 * r_char * rms_frac, 0.5)
        # Never claim better than the geometric spread can support.
        error_m = float(min(error_m, max(r_char, 1.0)))

        confidence = self._confidence(len(pos), confidences, rms_frac)
        source = {
            "lat": float(lat),
            "lon": float(lon),
            "confidence": float(confidence),
            "error_m": float(error_m),
        }
        return _SourceFit(source=source, kind="energy")

    @staticmethod
    def _confidence(
        n_devices: int, confidences: np.ndarray, rms_frac: float
    ) -> float:
        """Confidence in [0, 1] from device count, detection confidence, and fit.

        More devices and higher mean detection confidence raise it; a large
        fractional residual (poor model fit) lowers it.
        """
        # Geometry term: saturates as devices accumulate (3 -> ~0.5, 6 -> ~0.8).
        geom = 1.0 - 1.0 / max(n_devices - 1, 1)
        det = float(np.clip(np.mean(confidences), 0.0, 1.0)) if confidences.size else 0.0
        fit = 1.0 / (1.0 + 5.0 * rms_frac)  # 1 at perfect fit, decays with residual
        conf = geom * (0.5 + 0.5 * det) * fit
        return float(np.clip(conf, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    # Human-readable note
    # ------------------------------------------------------------------ #
    @staticmethod
    def _note(network: dict, positioning: str, fit: _SourceFit) -> str:
        if network["n_devices"] == 0:
            return "Waiting for devices to join."
        if not network["connected"]:
            return "All devices offline — waiting for reports."

        parts = [f"{network['n_devices']} device(s)"]
        if network["n_gps"]:
            parts.append(f"{network['n_gps']} with GPS")
        parts.append(f"{network['n_detecting']} detecting")
        head = ", ".join(parts) + "."

        if fit.kind == "energy":
            tail = " Source localized by acoustic energy (multilateration)."
        elif fit.kind == "region":
            tail = " Coarse source region from 2 detecting devices."
        elif network["n_detecting"] >= 1:
            tail = " Detecting, but need positions on >=2 detecting devices to localize."
        else:
            tail = " Listening; no detection yet."
        return head + tail

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_name(device_id: str, name: Optional[str]) -> str:
        if name:
            return name
        return f"Device {device_id}"
