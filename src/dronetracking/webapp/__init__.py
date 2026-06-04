"""Zero-install browser app: devices join by opening a URL, grant mic + location, and
the coordinator does whatever the connected devices allow.

Design for what browsers can actually do:
- Mic level is easy and reliable; precise cross-device audio *timing* is not. So source
  localization is by **acoustic energy** (received level ~ 1/distance), which needs no
  clock sync — it works on ordinary phones.
- Device positions come from the browser's **GPS** (Geolocation API) when granted;
  acoustic ranging remains available as a GPS-denied fallback.

Capability-adaptive (does what it can):
  1 device  -> live detection + level
  2 devices -> + a coarse source region
  3+ devices-> + an energy-multilaterated source fix on a real map
  GPS on N  -> devices placed at real coordinates

Run: ``python -m dronetracking.webapp`` (open the printed URL on every device).
"""
