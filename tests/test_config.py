from pathlib import Path

import pytest

from dronetracking.config import load_scenario, scenario_from_dict

SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"


def _minimal_dict():
    return {
        "name": "t",
        "seed": 0,
        "speed_of_sound_mps": 343.0,
        "sample_rate_hz": 48000.0,
        "duration_s": 2.0,
        "dt_s": 1.0,
        "ranging_rounds": 5,
        "origin_latlon": [32.0, 34.0],
        "noise": {},
        "devices": [
            {"id": "d0", "position_m": [0, 0, 0], "has_gps": True},
            {"id": "d1", "position_m": [10, 0, 0]},
        ],
        "trajectory": {"kind": "linear", "z_m": 50.0, "params": {"start_m": [0, 0], "end_m": [10, 10]}},
    }


def test_loads_field_scenario_from_yaml():
    sc = load_scenario(SCENARIOS / "field_5dev.yaml")
    assert sc.name == "field_5dev"
    assert sc.seed == 42
    assert len(sc.devices) == 5
    assert sc.trajectory.kind == "linear"
    anchors = [d for d in sc.devices if d.has_gps]
    assert len(anchors) == 4  # >=4 non-coplanar anchors for unambiguous 3D georeferencing
    # device positions parsed as 3-tuples of floats
    assert sc.devices[1].position_m == (200.0, 0.0, 18.0)


def test_all_shipped_scenarios_parse():
    for name in ("field_5dev", "noisefree_ideal", "sparse_anchors_circular"):
        sc = load_scenario(SCENARIOS / f"{name}.yaml")
        assert len(sc.devices) >= 5
        assert sum(d.has_gps for d in sc.devices) >= 3  # georeferencing well-posed


def test_seed_override_replaces_scenario_seed():
    sc = scenario_from_dict(_minimal_dict(), seed_override=99)
    assert sc.seed == 99


def test_noise_defaults_to_zero_when_omitted():
    d = _minimal_dict()
    d["noise"] = {}
    sc = scenario_from_dict(d)
    assert sc.noise.ranging_timestamp_std_s == 0.0
    assert sc.noise.toa_std_s == 0.0


def test_rejects_unknown_trajectory_kind():
    d = _minimal_dict()
    d["trajectory"]["kind"] = "spiral"
    with pytest.raises(ValueError, match="trajectory"):
        scenario_from_dict(d)


def test_rejects_empty_devices():
    d = _minimal_dict()
    d["devices"] = []
    with pytest.raises(ValueError, match="device"):
        scenario_from_dict(d)


def test_rejects_duplicate_device_ids():
    d = _minimal_dict()
    d["devices"][1]["id"] = "d0"
    with pytest.raises(ValueError, match="duplicate"):
        scenario_from_dict(d)


def test_scenario_is_immutable():
    sc = scenario_from_dict(_minimal_dict())
    with pytest.raises(Exception):
        sc.seed = 5  # frozen dataclass
