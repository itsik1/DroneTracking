"""Real-capture device runtime.

The code that runs *on a device*: capture sensor data through a :class:`CaptureBackend`,
detect the drone acoustically, and publish measurements to the coordinator over the wire
protocol. `MockBackend` drives it from the simulator (for testing); `SoundDeviceBackend`
drives it from a real microphone. Swapping the backend is the only difference between a
simulated run and a real device.
"""
