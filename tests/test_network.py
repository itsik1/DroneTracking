"""Tests for Phase-1 network formation (node registry, transport, discovery).

Covers the contract's TDD list:
  * devices within comm_range_m are neighbors and farther ones are not;
  * the graph is connected for a dense layout and partitioned for a sparse one;
  * capabilities / battery surface from DeviceSpec;
  * SimulatedTransport drops ~loss_prob of packets over many seeded trials;
  * is_connected / health are correct on hand-built layouts.

Scenarios are built via config.scenario_from_dict / config.load_scenario.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dronetracking.config import load_scenario, scenario_from_dict
from dronetracking.network import (
    NetworkGraph,
    NetworkManager,
    Node,
    NodeRegistry,
    SimulatedTransport,
    Transport,
    capabilities,
    discover,
)
from dronetracking.network.transport import RADIO_PRESETS
from dronetracking.sim.scenario import DeviceSpec

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _scenario_dict(devices, network=None, **overrides):
    """Minimal valid scenario dict with the given devices and network block."""
    raw = {
        "name": "net_test",
        "seed": 1,
        "speed_of_sound_mps": 343.0,
        "sample_rate_hz": 48000.0,
        "duration_s": 5.0,
        "dt_s": 0.5,
        "ranging_rounds": 10,
        "origin_latlon": [32.0, 34.0],
        "devices": devices,
        "trajectory": {"kind": "linear", "z_m": 50.0,
                       "params": {"start_m": [0.0, 0.0], "end_m": [10.0, 10.0]}},
    }
    if network is not None:
        raw["network"] = network
    raw.update(overrides)
    return raw


def _dev(id_, pos, **kw):
    return {"id": id_, "position_m": list(pos), **kw}


# --------------------------------------------------------------------------- #
# Node / capabilities / registry
# --------------------------------------------------------------------------- #
def test_node_from_spec_surfaces_capabilities_and_battery():
    spec = DeviceSpec(
        id="d0", position_m=(0.0, 0.0, 0.0),
        has_gps=True, battery_frac=0.42, has_mic=True, has_speaker=False,
    )
    node = Node.from_spec(spec)
    assert node.id == "d0"
    assert node.battery_frac == pytest.approx(0.42)
    assert node.has_gps is True
    assert node.has_mic is True
    assert node.has_speaker is False
    assert node.online is True
    assert node.confidence == pytest.approx(1.0)


def test_capabilities_helper_is_sorted_and_capability_aware():
    full = Node(id="a", has_mic=True, has_speaker=True, has_gps=True)
    assert full.capabilities() == ("gps", "mic", "speaker")
    assert capabilities(full) == ("gps", "mic", "speaker")
    mic_only = Node(id="b", has_mic=True, has_speaker=False, has_gps=False)
    assert mic_only.capabilities() == ("mic",)
    none = Node(id="c", has_mic=False, has_speaker=False, has_gps=False)
    assert none.capabilities() == ()


def test_registry_membership_and_health_queries():
    reg = NodeRegistry.from_specs([
        DeviceSpec(id="a", position_m=(0, 0, 0), battery_frac=1.0, has_gps=True),
        DeviceSpec(id="b", position_m=(1, 0, 0), battery_frac=0.5),
        DeviceSpec(id="c", position_m=(2, 0, 0), battery_frac=0.0),
    ])
    assert len(reg) == 3
    assert "a" in reg and "z" not in reg
    assert reg.ids == ("a", "b", "c")
    assert reg["b"].battery_frac == pytest.approx(0.5)
    assert reg.anchors() == ("a",)
    assert reg.mean_battery() == pytest.approx(0.5)
    assert reg.online_count() == 3

    reg.set_online("c", False)
    assert reg.online_count() == 2
    assert reg.offline_ids() == ("c",)
    assert "c" not in set(reg.online_ids())


def test_registry_mean_battery_empty_is_zero():
    assert NodeRegistry().mean_battery() == 0.0


# --------------------------------------------------------------------------- #
# SimulatedTransport: reachability, quality, presets, loss
# --------------------------------------------------------------------------- #
def test_transport_reachability_within_range():
    positions = {"a": (0, 0, 0), "b": (50, 0, 0), "c": (300, 0, 0)}
    t = SimulatedTransport(positions, comm_range_m=100.0, loss_prob=0.0, kind="wifi")
    assert t.reachable("a", "b") is True   # 50 m <= 100 m
    assert t.reachable("a", "c") is False  # 300 m  > 100 m
    assert t.reachable("a", "a") is True   # self


def test_transport_link_quality_decays_with_distance_and_zero_out_of_range():
    positions = {"a": (0, 0, 0), "near": (10, 0, 0), "mid": (50, 0, 0), "far": (200, 0, 0)}
    t = SimulatedTransport(positions, comm_range_m=100.0, loss_prob=0.0, kind="wifi")
    q_near = t.link_quality("a", "near")
    q_mid = t.link_quality("a", "mid")
    assert 0.0 <= q_mid <= q_near <= 1.0
    assert q_mid == pytest.approx(0.5)             # 50/100 -> 0.5
    assert t.link_quality("a", "far") == 0.0       # out of range
    assert t.link_quality("a", "a") == 1.0


def test_transport_kind_presets_differ():
    # BLE / Wi-Fi / mesh must have distinct range / latency / loss defaults.
    assert RADIO_PRESETS["ble"].comm_range_m < RADIO_PRESETS["wifi"].comm_range_m < RADIO_PRESETS["mesh"].comm_range_m
    pos = {"a": (0, 0, 0), "b": (1, 0, 0)}
    for kind in ("ble", "wifi", "mesh"):
        t = SimulatedTransport(pos, kind=kind)
        assert t.comm_range_m == RADIO_PRESETS[kind].comm_range_m
        assert t.latency_s == RADIO_PRESETS[kind].latency_s
        assert t.loss_prob == RADIO_PRESETS[kind].loss_prob
    # wifi reliable, mesh lossier
    assert RADIO_PRESETS["wifi"].loss_prob < RADIO_PRESETS["mesh"].loss_prob


def test_transport_explicit_params_override_preset():
    t = SimulatedTransport({"a": (0, 0, 0)}, comm_range_m=42.0, latency_s=0.123, loss_prob=0.3, kind="ble")
    assert t.comm_range_m == 42.0
    assert t.latency_s == 0.123
    assert t.loss_prob == 0.3


def test_transport_unknown_kind_raises():
    with pytest.raises(ValueError):
        SimulatedTransport({"a": (0, 0, 0)}, kind="lora")


def test_transport_drops_roughly_loss_prob_fraction_over_many_trials():
    # Two in-range devices; send many packets; the delivered fraction ~= 1 - loss_prob.
    positions = {"a": (0, 0, 0), "b": (10, 0, 0)}
    loss = 0.25
    n = 20000
    t = SimulatedTransport(positions, comm_range_m=100.0, loss_prob=loss, kind="wifi", rng=12345)
    delivered = sum(1 for _ in range(n) if t.send("a", "b", b"x"))
    frac_dropped = 1.0 - delivered / n
    assert frac_dropped == pytest.approx(loss, abs=0.02)
    # And every delivered packet is retrievable.
    assert len(t.deliver()) == delivered


def test_transport_out_of_range_send_always_fails():
    positions = {"a": (0, 0, 0), "b": (500, 0, 0)}
    t = SimulatedTransport(positions, comm_range_m=100.0, loss_prob=0.0, kind="wifi", rng=0)
    assert all(t.send("a", "b", b"x") is False for _ in range(100))
    assert t.deliver() == ()


def test_transport_seed_reproducible():
    pos = {"a": (0, 0, 0), "b": (10, 0, 0)}
    t1 = SimulatedTransport(pos, comm_range_m=100, loss_prob=0.5, rng=99)
    seq1 = [t1.send("a", "b", 1) for _ in range(50)]
    # Fresh transport, same seed -> identical drop pattern.
    t2 = SimulatedTransport(pos, comm_range_m=100, loss_prob=0.5, rng=99)
    seq2 = [t2.send("a", "b", 1) for _ in range(50)]
    assert seq1 == seq2
    # A different seed should generally diverge.
    t3 = SimulatedTransport(pos, comm_range_m=100, loss_prob=0.5, rng=100)
    seq3 = [t3.send("a", "b", 1) for _ in range(50)]
    assert seq3 != seq1


def test_transport_is_subclass_of_abc():
    assert issubclass(SimulatedTransport, Transport)
    with pytest.raises(TypeError):
        Transport()  # abstract, cannot instantiate


# --------------------------------------------------------------------------- #
# discover() + NetworkGraph on hand-built layouts
# --------------------------------------------------------------------------- #
def test_discover_neighbors_match_range():
    # a-b in range (50<=100), b-c in range (60<=100), a-c out of range (110>100).
    positions = {"a": (0, 0, 0), "b": (50, 0, 0), "c": (110, 0, 0)}
    reg = NodeRegistry([Node("a"), Node("b"), Node("c")])
    t = SimulatedTransport(positions, comm_range_m=100.0, loss_prob=0.0, kind="wifi", rng=0)
    g = discover(reg, t, rng=0)
    assert set(g.neighbors("a")) == {"b"}
    assert set(g.neighbors("b")) == {"a", "c"}
    assert set(g.neighbors("c")) == {"b"}
    # symmetric edge list
    assert g.edges() == (("a", "b"), ("b", "c"))


def test_discover_graph_connected_for_dense_layout():
    # Tight cluster, big range -> a clique -> connected.
    positions = {f"d{i}": (i * 10.0, 0.0, 0.0) for i in range(5)}
    reg = NodeRegistry([Node(f"d{i}") for i in range(5)])
    t = SimulatedTransport(positions, comm_range_m=1000.0, loss_prob=0.0, kind="wifi", rng=0)
    g = discover(reg, t, rng=0)
    assert g.is_connected() is True
    assert g.isolated() == ()
    assert len(g.components()) == 1


def test_discover_graph_partitioned_for_sparse_layout():
    # Two far-apart pairs, small range -> two components, graph NOT connected.
    positions = {
        "a": (0, 0, 0), "b": (20, 0, 0),          # cluster 1
        "c": (5000, 0, 0), "d": (5020, 0, 0),     # cluster 2 (far away)
    }
    reg = NodeRegistry([Node("a"), Node("b"), Node("c"), Node("d")])
    t = SimulatedTransport(positions, comm_range_m=100.0, loss_prob=0.0, kind="wifi", rng=0)
    g = discover(reg, t, rng=0)
    assert g.is_connected() is False
    comps = {frozenset(c) for c in g.components()}
    assert comps == {frozenset({"a", "b"}), frozenset({"c", "d"})}


def test_discover_isolated_node():
    positions = {"a": (0, 0, 0), "b": (10, 0, 0), "lonely": (9000, 0, 0)}
    reg = NodeRegistry([Node("a"), Node("b"), Node("lonely")])
    t = SimulatedTransport(positions, comm_range_m=100.0, loss_prob=0.0, kind="wifi", rng=0)
    g = discover(reg, t, rng=0)
    assert g.isolated() == ("lonely",)
    assert g.degree("lonely") == 0
    assert g.is_connected() is False


def test_discover_offline_node_has_no_edges():
    positions = {"a": (0, 0, 0), "b": (10, 0, 0), "c": (20, 0, 0)}
    reg = NodeRegistry([Node("a"), Node("b"), Node("c", online=False)])
    t = SimulatedTransport(positions, comm_range_m=100.0, loss_prob=0.0, kind="wifi", rng=0)
    g = discover(reg, t, rng=0)
    # c is offline -> never broadcasts or is reached.
    assert g.degree("c") == 0
    assert "c" not in g.neighbors("a")
    assert "c" not in g.neighbors("b")


# --------------------------------------------------------------------------- #
# NetworkGraph direct unit checks (hand-built)
# --------------------------------------------------------------------------- #
def test_graph_is_connected_handbuilt():
    g = NetworkGraph(node_ids=("a", "b", "c"))
    g.add_edge("a", "b", 0.9)
    g.add_edge("b", "c", 0.8)
    assert g.is_connected() is True
    assert g.mean_link_quality() == pytest.approx((0.9 + 0.8) / 2)

    g2 = NetworkGraph(node_ids=("a", "b", "c", "d"))
    g2.add_edge("a", "b")
    g2.add_edge("c", "d")
    assert g2.is_connected() is False


def test_graph_trivial_connectivity():
    assert NetworkGraph(node_ids=()).is_connected() is True
    assert NetworkGraph(node_ids=("solo",)).is_connected() is True


# --------------------------------------------------------------------------- #
# NetworkManager end-to-end via scenarios
# --------------------------------------------------------------------------- #
def test_manager_dense_scenario_connected_and_neighbors_in_range():
    devices = [
        _dev("dev0", (0.0, 0.0, 0.0), has_gps=True, battery_frac=0.9),
        _dev("dev1", (50.0, 0.0, 0.0), has_gps=True, battery_frac=0.8),
        _dev("dev2", (100.0, 0.0, 0.0), battery_frac=0.7),
        _dev("dev3", (50.0, 50.0, 0.0), battery_frac=0.6),
    ]
    scen = scenario_from_dict(_scenario_dict(
        devices, network={"comm_range_m": 200.0, "loss_prob": 0.0, "kind": "wifi"}
    ))
    mgr = NetworkManager(scen)
    g = mgr.form_network()
    assert mgr.is_connected() is True
    # dev0 reaches dev1 (50) and dev3 (~70.7) within 200; all within 200 actually.
    assert "dev1" in mgr.neighbors("dev0")
    assert g.isolated() == ()


def test_manager_sparse_scenario_partitioned():
    devices = [
        _dev("a", (0.0, 0.0, 0.0)),
        _dev("b", (30.0, 0.0, 0.0)),
        _dev("c", (10000.0, 0.0, 0.0)),  # unreachable
    ]
    scen = scenario_from_dict(_scenario_dict(
        devices, network={"comm_range_m": 100.0, "loss_prob": 0.0, "kind": "wifi"}
    ))
    mgr = NetworkManager(scen)
    mgr.form_network()
    assert mgr.is_connected() is False
    assert mgr.neighbors("c") == ()


def test_manager_health_summary_fields_and_values():
    devices = [
        _dev("a", (0.0, 0.0, 0.0), has_gps=True, battery_frac=1.0, has_mic=True, has_speaker=True),
        _dev("b", (40.0, 0.0, 0.0), battery_frac=0.5, has_mic=False, has_speaker=True),
        _dev("c", (10000.0, 0.0, 0.0), battery_frac=0.0),  # isolated
    ]
    scen = scenario_from_dict(_scenario_dict(
        devices, network={"comm_range_m": 100.0, "loss_prob": 0.0, "kind": "wifi"}
    ))
    mgr = NetworkManager(scen)
    mgr.form_network()
    h = mgr.health()

    assert h["online"] == 3
    assert h["total"] == 3
    assert h["mean_battery"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert h["connected"] is False
    assert h["n_components"] == 2
    assert "c" in h["isolated"]
    assert 0.0 <= h["mean_link_quality"] <= 1.0

    nodes_by_id = {n["id"]: n for n in h["nodes"]}
    assert nodes_by_id["b"]["capabilities"] == ["speaker"]   # no mic
    assert nodes_by_id["a"]["capabilities"] == ["gps", "mic", "speaker"]
    assert nodes_by_id["c"]["degree"] == 0
    assert set(nodes_by_id["a"]["neighbors"]) == {"b"}


def test_manager_lazy_forms_graph_on_query():
    devices = [_dev("a", (0.0, 0.0, 0.0)), _dev("b", (10.0, 0.0, 0.0))]
    scen = scenario_from_dict(_scenario_dict(
        devices, network={"comm_range_m": 100.0, "loss_prob": 0.0}
    ))
    mgr = NetworkManager(scen)
    assert mgr.graph is None
    # Querying without an explicit form_network() should still work.
    assert mgr.is_connected() is True
    assert mgr.graph is not None


def test_manager_reproducible_health_with_loss():
    # With packet loss, a fixed seed must give a reproducible graph/health.
    devices = [_dev(f"d{i}", (i * 60.0, 0.0, 0.0)) for i in range(5)]
    net = {"comm_range_m": 80.0, "loss_prob": 0.3, "kind": "wifi"}
    scen = scenario_from_dict(_scenario_dict(devices, network=net, seed=123))
    h1 = NetworkManager(scen).health()
    h2 = NetworkManager(scen).health()
    assert h1 == h2


# --------------------------------------------------------------------------- #
# the optional demo scenario loads and forms a connected network
# --------------------------------------------------------------------------- #
def test_network_demo_scenario_loads_and_forms_connected_network():
    path = REPO_ROOT / "scenarios" / "network_demo.yaml"
    if not path.exists():
        pytest.skip("network_demo.yaml not present")
    scen = load_scenario(path)
    mgr = NetworkManager(scen)
    mgr.form_network()
    assert mgr.is_connected() is True
    h = mgr.health()
    assert h["total"] == len(scen.devices)
    # Varied capabilities surfaced from the spec.
    caps = {n["id"]: n["capabilities"] for n in h["nodes"]}
    assert "mic" not in caps["dev3"]      # dev3 has_mic: false
    assert "speaker" not in caps["dev2"]  # dev2 has_speaker: false
