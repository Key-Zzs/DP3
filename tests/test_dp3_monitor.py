from __future__ import annotations

import sys
import threading
import time
import types

import numpy as np

from visualizer.monitor.client import MonitorClient
from visualizer.monitor.config import MonitorConfig, MonitorRates, TelemetryShapes
from visualizer.monitor.process import TelemetryProcess
from visualizer.monitor.schema import TelemetrySchema
from visualizer.monitor.shared_ring import SharedMemoryTelemetryBus


class _SlowSink:
    def start(self):
        return None

    def consume(self, snapshot):
        time.sleep(0.02)

    def close(self):
        return None


def _slow_sink_factory(**kwargs):
    return _SlowSink()


class _ReconnectOnceSink:
    def __init__(self):
        self.failed = False

    def start(self):
        return None

    def consume(self, snapshot):
        if not self.failed:
            self.failed = True
            raise RuntimeError("synthetic sink failure")

    def close(self):
        return None

    def try_reconnect(self):
        return True


def _reconnect_once_sink_factory(**kwargs):
    return _ReconnectOnceSink()


def _shapes() -> TelemetryShapes:
    return TelemetryShapes(
        camera_height=4,
        camera_width=5,
        point_count=3,
        point_dim=3,
        state_dim=4,
        action_dim=2,
        policy_horizon=3,
        depth_dtype="uint16",
        max_raw_points=6,
        max_cropped_points=5,
    )


def test_ring_latest_only_and_fixed_shapes() -> None:
    bus = SharedMemoryTelemetryBus.create(TelemetrySchema.from_shapes(_shapes()), capacity=2)
    try:
        for sequence in range(5):
            result = bus.publish(
                "sampled_pointcloud",
                {"points": np.full((3, 3), sequence, dtype=np.float32)},
                cycle_id=sequence,
            )
            assert result.committed
        snapshot = bus.consume_latest("sampled_pointcloud")
        assert snapshot is not None
        assert snapshot.sequence == 4
        assert snapshot.metadata["cycle_id"] == 4
        np.testing.assert_array_equal(snapshot.arrays["points"], 4.0)
        assert bus.stats()["sampled_pointcloud"]["overwritten"] == 3
    finally:
        bus.close()
        bus.unlink()


def test_ring_fast_consumer_does_not_report_false_overwrites_across_wraparound() -> None:
    bus = SharedMemoryTelemetryBus.create(TelemetrySchema.from_shapes(_shapes()), capacity=2)
    try:
        after = -1
        for sequence in range(8):
            result = bus.publish(
                "sampled_pointcloud",
                {"points": np.full((3, 3), sequence, dtype=np.float32)},
                cycle_id=sequence,
            )
            assert result.committed
            assert not result.overwritten
            snapshot = bus.consume_latest("sampled_pointcloud", after_sequence=after)
            assert snapshot is not None
            assert snapshot.sequence == sequence
            after = snapshot.sequence
        assert bus.stats()["sampled_pointcloud"] == {"published": 8, "dropped": 0, "overwritten": 0}
    finally:
        bus.close()
        bus.unlink()


def test_ring_channels_have_independent_sequences_flags_shapes_and_dtypes() -> None:
    shapes = _shapes()
    bus = SharedMemoryTelemetryBus.create(TelemetrySchema.from_shapes(shapes), capacity=3)
    try:
        camera = bus.publish(
            "camera",
            {
                "rgb": np.zeros((4, 5, 3), dtype=np.uint8),
                "depth": np.ones((4, 5), dtype=np.uint16),
            },
            valid_flags=0xA5,
        )
        sampled0 = bus.publish("sampled_pointcloud", {"points": np.zeros((3, 3), dtype=np.float32)})
        sampled1 = bus.publish("sampled_pointcloud", {"points": np.ones((3, 3), dtype=np.float32)})
        assert camera.sequence == sampled0.sequence == 0
        assert sampled1.sequence == 1
        camera_snapshot = bus.consume_latest("camera")
        sampled_snapshot = bus.consume_latest("sampled_pointcloud")
        assert camera_snapshot is not None and sampled_snapshot is not None
        assert camera_snapshot.metadata["valid_flags"] == 0xA5
        assert camera_snapshot.arrays["rgb"].shape == (4, 5, 3)
        assert camera_snapshot.arrays["rgb"].dtype == np.uint8
        assert camera_snapshot.arrays["depth"].dtype == np.uint16
        assert sampled_snapshot.arrays["points"].shape == (3, 3)
        assert sampled_snapshot.arrays["points"].dtype == np.float32
    finally:
        bus.close()
        bus.unlink()


def test_ring_xyzrgb_shape_and_torn_read_protection_under_concurrency() -> None:
    shapes = TelemetryShapes(
        camera_height=2,
        camera_width=2,
        point_count=256,
        point_dim=6,
        state_dim=4,
        action_dim=2,
        policy_horizon=3,
        max_raw_points=8,
        max_cropped_points=8,
    )
    bus = SharedMemoryTelemetryBus.create(TelemetrySchema.from_shapes(shapes), capacity=3)
    failures = []
    finished = threading.Event()

    def producer() -> None:
        for sequence in range(200):
            bus.publish(
                "sampled_pointcloud",
                {"points": np.full((256, 6), sequence, dtype=np.float32)},
                cycle_id=sequence,
            )
        finished.set()

    thread = threading.Thread(target=producer)
    thread.start()
    after = -1
    try:
        while not finished.is_set() or after < 199:
            snapshot = bus.consume_latest("sampled_pointcloud", after_sequence=after)
            if snapshot is None:
                time.sleep(0.0001)
                continue
            values = snapshot.arrays["points"]
            if values.shape != (256, 6) or not np.all(values == values[0, 0]):
                failures.append(snapshot.sequence)
            after = snapshot.sequence
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert not failures
        assert after == 199
    finally:
        thread.join(timeout=2.0)
        bus.close()
        bus.unlink()


def test_ring_producer_drops_without_waiting_when_slots_locked() -> None:
    bus = SharedMemoryTelemetryBus.create(TelemetrySchema.from_shapes(_shapes()), capacity=3)
    channel = bus._channels["sampled_pointcloud"]  # white-box lock contention test
    for lock in channel._locks:
        assert lock.acquire(block=False)
    started = time.perf_counter()
    try:
        result = bus.publish(
            "sampled_pointcloud",
            {"points": np.zeros((3, 3), dtype=np.float32)},
        )
    finally:
        for lock in channel._locks:
            lock.release()
        bus.close()
        bus.unlink()
    assert not result.committed
    assert result.lock_contention
    assert time.perf_counter() - started < 0.2


def test_ring_attach_and_close_are_idempotent() -> None:
    bus = SharedMemoryTelemetryBus.create(TelemetrySchema.from_shapes(_shapes()), capacity=3)
    descriptor = bus.descriptor()
    child_view = SharedMemoryTelemetryBus.attach(descriptor)
    try:
        bus.publish("camera", {"rgb": np.zeros((4, 5, 3), dtype=np.uint8), "depth": np.ones((4, 5), dtype=np.uint16)})
        snapshot = child_view.consume_latest("camera")
        assert snapshot is not None
        assert snapshot.arrays["depth"].dtype == np.uint16
    finally:
        child_view.close()
        child_view.close()
        bus.close()
        bus.close()
        bus.unlink()


def test_disabled_client_is_true_noop() -> None:
    sys.modules.pop("rerun", None)
    client = MonitorClient.disabled()
    plan = client.plan_cycle(0.0)
    result = client.publish_cycle(cycle_id=1, measured_state=np.zeros(4, dtype=np.float32), plan=plan)
    assert not plan.control_due
    assert result == {"enabled": False}
    assert "rerun" not in sys.modules
    client.close()
    client.close()


def test_client_rate_gates_bulk_before_copy() -> None:
    shapes = _shapes()
    config = MonitorConfig(
        enabled=True,
        backend="rerun",
        rates=MonitorRates(control_hz=10.0, camera_hz=1.0, sampled_pointcloud_hz=1.0, stage_pointcloud_hz=1.0, health_hz=1.0),
    )
    client = MonitorClient.create(config, shapes, sink_kind="null")
    try:
        now = 0.0
        first = client.plan_cycle(now)
        client.publish_cycle(
            cycle_id=0,
            measured_state=np.zeros(4, dtype=np.float32),
            rgb=np.zeros((4, 5, 3), dtype=np.uint8),
            depth=np.zeros((4, 5), dtype=np.uint16),
            sampled_pointcloud=np.zeros((3, 3), dtype=np.float32),
            plan=first,
        )
        second = client.plan_cycle(0.01)
        assert not second.control_due
        assert not second.camera_due
        assert not second.sampled_pointcloud_due
        stats_before = client.bus.stats()  # type: ignore[union-attr]

        class ExplodingBulkArray:
            def __array__(self, *args, **kwargs):
                raise AssertionError("bulk payload was converted despite rate gate")

        client.publish_cycle(
            cycle_id=1,
            measured_state=np.ones(4, dtype=np.float32),
            rgb=ExplodingBulkArray(),
            depth=ExplodingBulkArray(),
            sampled_pointcloud=ExplodingBulkArray(),
            plan=second,
        )
        stats_after = client.bus.stats()  # type: ignore[union-attr]
        assert stats_after["camera"]["published"] == stats_before["camera"]["published"]
        assert stats_after["sampled_pointcloud"]["published"] == stats_before["sampled_pointcloud"]["published"]
    finally:
        client.close()


def test_client_slack_gate_skips_bulk_but_still_commits_control() -> None:
    config = MonitorConfig(
        enabled=True,
        min_bulk_slack_ms=5.0,
        rates=MonitorRates(control_hz=10.0, camera_hz=5.0, sampled_pointcloud_hz=5.0, stage_pointcloud_hz=1.0, health_hz=1.0),
    )
    client = MonitorClient.create(config, _shapes(), sink_kind="null", start_process=False)
    try:
        plan = client.plan_cycle(0.0)
        result = client.publish_cycle(
            cycle_id=0,
            measured_state=np.ones(4, dtype=np.float32),
            rgb=np.ones((4, 5, 3), dtype=np.uint8),
            depth=np.ones((4, 5), dtype=np.uint16),
            sampled_pointcloud=np.ones((3, 3), dtype=np.float32),
            plan=plan,
            remaining_slack_ms=0.0,
        )
        assert result["control"]
        assert result["bulk"] == {}
        stats = client.stats()
        assert stats["bulk_skipped_slack"] == 1
        assert stats["bus"]["control"]["published"] == 1
        assert stats["bus"]["camera"]["published"] == 0
        assert stats["bus"]["sampled_pointcloud"]["published"] == 0
    finally:
        client.close()


def test_client_publish_remains_fail_open_after_child_crash() -> None:
    client = MonitorClient.create(MonitorConfig(enabled=True), _shapes(), sink_kind="null")
    try:
        assert client.process is not None
        client.process.process.terminate()
        client.process.process.join(timeout=2.0)
        assert not client.process.is_alive()
        plan = client.plan_cycle(0.0)
        result = client.publish_cycle(
            cycle_id=1,
            measured_state=np.ones(4, dtype=np.float32),
            plan=plan,
            remaining_slack_ms=100.0,
        )
        assert result["enabled"]
        assert result["control"]
    finally:
        client.close()


def test_spawned_telemetry_process_consumes_latest_and_stops() -> None:
    shapes = _shapes()
    bus = SharedMemoryTelemetryBus.create(TelemetrySchema.from_shapes(shapes), capacity=3)
    process = TelemetryProcess(bus.descriptor(), config=MonitorConfig(enabled=True), sink_kind="null")
    process.start(timeout=3.0)
    try:
        bus.publish("sampled_pointcloud", {"points": np.full((3, 3), 9, dtype=np.float32)}, cycle_id=9)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and process.is_alive():
            time.sleep(0.01)
        assert process.ready
        assert process.is_alive()
    finally:
        process.close(timeout=2.0)
        bus.close()
        bus.unlink()
    assert not process.is_alive()


def test_slow_sink_skips_stale_control_records_and_reports_latest_sequence() -> None:
    config = MonitorConfig(enabled=True, heartbeat_interval_sec=0.05)
    bus = SharedMemoryTelemetryBus.create(TelemetrySchema.from_shapes(_shapes()), capacity=3)
    process = TelemetryProcess(
        bus.descriptor(),
        config=config,
        sink_kind="fake",
        sink_factory=_slow_sink_factory,
    )
    process.start(timeout=3.0)
    try:
        for sequence in range(100):
            bus.publish("control", {"measured_state": np.full(4, sequence, dtype=np.float32)}, cycle_id=sequence)
        deadline = time.monotonic() + 2.0
        heartbeat = None
        while time.monotonic() < deadline:
            for message in process.poll_messages():
                if message.get("type") == "heartbeat":
                    heartbeat = message
            if heartbeat and heartbeat["latest_sequences"]["control"] == 99:
                break
            time.sleep(0.01)
        assert heartbeat is not None
        assert heartbeat["latest_sequences"]["control"] == 99
        assert heartbeat["processed_counts"]["control"] < 10
        assert process.is_alive()
    finally:
        process.close(timeout=2.0)
        bus.close()
        bus.unlink()


def test_sink_error_reconnects_without_propagating_to_producer() -> None:
    config = MonitorConfig(enabled=True, heartbeat_interval_sec=0.05)
    bus = SharedMemoryTelemetryBus.create(TelemetrySchema.from_shapes(_shapes()), capacity=3)
    process = TelemetryProcess(
        bus.descriptor(),
        config=config,
        sink_kind="fake",
        sink_factory=_reconnect_once_sink_factory,
    )
    process.start(timeout=3.0)
    try:
        bus.publish("control", {"measured_state": np.ones(4, dtype=np.float32)}, cycle_id=1)
        deadline = time.monotonic() + 2.0
        kinds = set()
        while time.monotonic() < deadline and "viewer_reconnected" not in kinds:
            kinds.update(message.get("type") for message in process.poll_messages())
            time.sleep(0.01)
        assert {"sink_error", "viewer_reconnected"} <= kinds
        assert process.is_alive()
        result = bus.publish("control", {"measured_state": np.full(4, 2, dtype=np.float32)}, cycle_id=2)
        assert result.committed
    finally:
        process.close(timeout=2.0)
        bus.close()
        bus.unlink()


def test_fail_open_startup_failure_cleans_owner_resources() -> None:
    client = MonitorClient.create(
        MonitorConfig(enabled=True, fail_open=True, startup_timeout_sec=1.0),
        _shapes(),
        sink_kind="unsupported-test-sink",
    )
    assert not client.enabled
    assert client.bus is None
    assert client.process is None
    client.close()


def test_rerun_sink_logs_fixed_entities_with_fake_recording(monkeypatch) -> None:
    from visualizer.monitor.config import MonitorPayloads, MonitorRecording, ViewerConfig
    from visualizer.monitor.rerun_sink import RerunSink
    from visualizer.monitor.schema import (
        FLAG_COMMANDED_ACTION,
        FLAG_DEPTH,
        FLAG_FILTERED_ACTION,
        FLAG_HORIZON,
        FLAG_RGB,
        FLAG_SELECTED_ACTION,
        FLAG_STATE,
        FLAG_TIMING,
    )

    class FakeRecording:
        def __init__(self):
            self.logs = []
            self.flushed = False
            self.disconnected = False

        def log(self, path, archetype):
            self.logs.append((path, archetype))

        def save(self, path):
            self.logs.append(("__save__", path))

        def set_time_sequence(self, name, value):
            self.logs.append((f"__time__/{name}", value))

        def flush(self, *, timeout_sec):
            assert timeout_sec == 1.0
            self.flushed = True

        def disconnect(self):
            self.disconnected = True

    class FakeArchetype:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    recording = FakeRecording()
    time_panel_calls = []
    blueprint_calls = []
    fake_rr = types.SimpleNamespace(
        init=lambda *args, **kwargs: recording,
        spawn=lambda **kwargs: None,
        send_blueprint=lambda *args, **kwargs: (
            blueprint_calls.append(kwargs),
            recording.logs.append(("__blueprint__", args[0])),
        )[-1],
        Image=FakeArchetype,
        DepthImage=FakeArchetype,
        Points3D=FakeArchetype,
        Scalar=FakeArchetype,
        TextLog=FakeArchetype,
        blueprint=types.SimpleNamespace(
            Spatial3DView=lambda **kwargs: ("spatial3d", kwargs),
            Spatial2DView=lambda **kwargs: ("spatial2d", kwargs),
            TimeSeriesView=lambda **kwargs: ("timeseries", kwargs),
            TextLogView=lambda **kwargs: ("textlog", kwargs),
            TimePanel=lambda **kwargs: time_panel_calls.append(kwargs) or ("timepanel", kwargs),
            Grid=lambda *args, **kwargs: ("grid", args, kwargs),
            Blueprint=lambda *args, **kwargs: ("blueprint", args, kwargs),
        ),
    )
    monkeypatch.setitem(sys.modules, "rerun", fake_rr)
    config = MonitorConfig(
        enabled=True,
        viewer=ViewerConfig(reconnect_attempts=1, detach_process=False),
        payloads=MonitorPayloads(raw_pointcloud=True, cropped_pointcloud=True),
        recording=MonitorRecording(save_rrd=True, rrd_path="synthetic.rrd"),
    )
    sink = RerunSink(
        config=config,
        static_metadata={
            "state_field_names": ["left_joint_1.pos", "right_joint_1.pos", "left_gripper_state_norm", "right_gripper_state_norm"],
            "action_field_names": ["left_delta_ee_pose.x", "right_gripper_cmd"],
        },
    )
    sink.start()
    schema = TelemetrySchema.from_shapes(_shapes())
    bus = SharedMemoryTelemetryBus.create(schema, capacity=2)
    try:
        bus.publish(
            "camera",
            {"rgb": np.zeros((4, 5, 3), dtype=np.uint8), "depth": np.ones((4, 5), dtype=np.uint16)},
            valid_flags=FLAG_RGB | FLAG_DEPTH,
            depth_scale=0.001,
        )
        sink.consume(bus.consume_latest("camera"))
        bus.publish(
            "sampled_pointcloud",
            {"points": np.zeros((3, 3), dtype=np.float32)},
            valid_flags=4,
            point_count=3,
        )
        sink.consume(bus.consume_latest("sampled_pointcloud"))
        bus.publish(
            "control",
            {
                "measured_state": np.ones((4,), dtype=np.float32),
                "policy_horizon": np.ones((3, 2), dtype=np.float32),
                "selected_raw_action": np.ones((2,), dtype=np.float32),
                "filtered_action": np.full((2,), 2, dtype=np.float32),
                "commanded_action": np.full((2,), 3, dtype=np.float32),
                "policy_predict_ms": np.array(1, dtype=np.float32),
            },
            valid_flags=(
                FLAG_STATE
                | FLAG_HORIZON
                | FLAG_SELECTED_ACTION
                | FLAG_FILTERED_ACTION
                | FLAG_COMMANDED_ACTION
                | FLAG_TIMING
            ),
            prediction_id=7,
            horizon_length=3,
            selected_index=1,
            commanded_valid=True,
        )
        control_snapshot = bus.consume_latest("control")
        control_snapshot.metadata["consumer_lag_cycles"] = 4
        sink.consume(control_snapshot)
        bus.publish("control", valid_flags=0, status=3, event_message="send error")
        sink.consume(bus.consume_latest("control"))
    finally:
        sink.close()
        bus.close()
        bus.unlink()
    paths = {path for path, _ in recording.logs if not path.startswith("__")}
    assert "/observation/camera/head/rgb" in paths
    assert "/observation/camera/head/depth" in paths
    assert "/observation/point_cloud/sampled" in paths
    assert "/observation/state/left_joint_1.pos" in paths
    assert "/policy/prediction/horizon/left_delta_ee_pose.x" in paths
    assert "/control/action_selected_raw/left_delta_ee_pose.x" in paths
    assert "/control/action_filtered/left_delta_ee_pose.x" in paths
    assert "/robot/action_commanded/left_delta_ee_pose.x" in paths
    assert "/timing/policy_predict_ms" in paths
    assert "/monitor/dropped/camera" in paths
    assert "/monitor/overwritten/sampled_pointcloud" in paths
    assert "/monitor/consumer_lag_cycles" in paths
    assert "/events" in paths
    assert "__blueprint__" in {path for path, _ in recording.logs}
    assert blueprint_calls[0]["make_active"] is True
    assert recording.logs[0] == ("__save__", "synthetic.rrd")
    assert time_panel_calls == [{"timeline": "log_time", "play_state": "following"}]
    blueprint = next(value for path, value in recording.logs if path == "__blueprint__")
    grid = blueprint[1][0]
    spatial_views = {
        view[1]["name"]: view[1]
        for view in grid[1]
        if view[0] == "spatial3d"
    }
    assert spatial_views["Raw point cloud (optional)"]["visible"] is True
    assert spatial_views["Cropped point cloud (optional)"]["visible"] is True
    view_names = {view[1]["name"] for view in grid[1] if view[0] == "timeseries"}
    assert "State | left joints [rad]" in view_names
    assert "State | left gripper [0..1]" in view_names
    assert "Actions | delta xyz [m]" in view_names
    assert "Actions | grippers [0..1]" in view_names
    assert "Timing [ms]" in view_names
    assert recording.flushed
    assert recording.disconnected


def test_monitor_config_requires_rrd_path_when_saving() -> None:
    from visualizer.monitor.config import load_monitor_config

    with np.testing.assert_raises_regex(ValueError, "rrd_path"):
        load_monitor_config(
            {
                "monitor": {
                    "enabled": True,
                    "recording": {"save_rrd": True, "rrd_path": None},
                }
            }
        )


def test_checked_in_monitor_profile_keeps_viewer_and_emits_all_visual_payloads() -> None:
    from pathlib import Path

    import yaml

    from visualizer.monitor.config import load_monitor_config

    config_path = (
        Path(__file__).resolve().parents[1]
        / "3D-Diffusion-Policy/diffusion_policy_3d/config/dp3_inference_config.yaml"
    )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # safe_load does not resolve the one OmegaConf interpolation used by the
    # launcher; resolve it exactly as the runtime/benchmark does.
    raw["monitor"]["rates"]["control_hz"] = float(raw["inference"]["rate_hz"])
    config = load_monitor_config(
        raw,
        inference_rate_hz=float(raw["inference"]["rate_hz"]),
    )

    assert config.viewer.detach_process is True
    assert config.viewer.activate_blueprint_on_start is True
    assert config.min_bulk_slack_ms == 0.0
    assert config.rates.camera_hz == 2.0
    assert config.rates.sampled_pointcloud_hz == 2.0
    assert config.rates.stage_pointcloud_hz == 1.0
    assert config.payloads.rgb and config.payloads.depth
    assert config.payloads.sampled_pointcloud
    assert config.payloads.raw_pointcloud and config.payloads.cropped_pointcloud
    assert config.display.max_raw_points == 5000
    assert config.display.max_cropped_points == 5000


def test_rerun_sink_uses_current_set_time_api() -> None:
    from visualizer.monitor.rerun_sink import RerunSink

    class CurrentRecording:
        def __init__(self):
            self.times = []
            self.logs = []

        def set_time(self, name, **kwargs):
            self.times.append((name, kwargs))

        def log(self, path, archetype):
            self.logs.append((path, archetype))

    sink = RerunSink(config=MonitorConfig(enabled=True))
    sink.recording = CurrentRecording()
    sink.rr = types.SimpleNamespace(Scalars=lambda value: value)
    sink._set_time(
        {
            "cycle_id": 12,
            "observation_timestamp": 1_784_393_435_304.6814,
            "wall_timestamp": 1_784_393_435.338,
        },
        sequence=12,
    )
    sink._log_scalar("/policy/prediction/horizon/left_x", 0.25, horizon_step=2, prediction_id=9)
    assert sink.recording.times == [
        ("cycle", {"sequence": 12}),
        ("observation", {"timestamp": 1_784_393_435.3046813}),
        ("wall_clock", {"timestamp": 1_784_393_435.338}),
        ("horizon_step", {"sequence": 2}),
        ("prediction_id", {"sequence": 9}),
    ]


def test_rerun_sink_spawned_viewer_client_is_closed() -> None:
    from visualizer.monitor.config import ViewerConfig
    from visualizer.monitor.rerun_sink import RerunSink

    calls = []

    class FakeRecording:
        def flush(self, *, timeout_sec):
            calls.append(("flush", timeout_sec))

        def disconnect(self):
            calls.append(("disconnect",))

    class FakeViewerClient:
        url = "rerun+http://127.0.0.1:9877/proxy"

        @classmethod
        def spawn(cls, **kwargs):
            calls.append(("spawn", kwargs))
            return cls()

        def close(self):
            calls.append(("close",))

    recording = FakeRecording()
    sink = RerunSink(
        config=MonitorConfig(
            enabled=True,
            viewer=ViewerConfig(port=9877, detach_process=False),
        )
    )
    sink.rr = types.SimpleNamespace(
        experimental=types.SimpleNamespace(ViewerClient=FakeViewerClient),
        connect_grpc=lambda url, **kwargs: calls.append(("connect", url, kwargs)),
    )
    sink.recording = recording
    sink._connect_viewer()
    sink.close()

    assert calls[0][0] == "spawn"
    assert calls[0][1]["port"] == 9877
    assert calls[1][0:2] == ("connect", FakeViewerClient.url)
    assert calls[-3:] == [("flush", 1.0), ("disconnect",), ("close",)]


def test_rerun_sink_detached_viewer_survives_close_and_existing_port_is_reused(
    monkeypatch,
) -> None:
    from visualizer.monitor.config import ViewerConfig
    from visualizer.monitor import rerun_sink as rerun_sink_module
    from visualizer.monitor.rerun_sink import RerunSink

    calls = []

    class FakeRecording:
        def flush(self, *, timeout_sec):
            calls.append(("flush", timeout_sec))

        def disconnect(self):
            calls.append(("disconnect",))

    class FakeViewerClient:
        url = "rerun+http://127.0.0.1:9876/proxy"

        @classmethod
        def connect(cls, url):
            calls.append(("reuse", url))
            return cls()

        @classmethod
        def spawn(cls, **kwargs):
            calls.append(("spawn", kwargs))
            return cls()

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(rerun_sink_module, "_local_port_open", lambda port: True)
    sink = RerunSink(
        config=MonitorConfig(
            enabled=True,
            viewer=ViewerConfig(port=9876, detach_process=True),
        )
    )
    sink.rr = types.SimpleNamespace(
        experimental=types.SimpleNamespace(ViewerClient=FakeViewerClient),
        connect_grpc=lambda url, **kwargs: calls.append(("connect", url, kwargs)),
    )
    sink.recording = FakeRecording()
    sink._connect_viewer()
    sink.close()

    assert calls[0] == ("reuse", FakeViewerClient.url)
    assert calls[1][0:2] == ("connect", FakeViewerClient.url)
    assert not any(call[0] == "spawn" for call in calls)
    assert not any(call[0] == "close" for call in calls)


def test_rerun_sink_new_detached_viewer_is_not_closed(monkeypatch) -> None:
    from visualizer.monitor.config import ViewerConfig
    from visualizer.monitor import rerun_sink as rerun_sink_module
    from visualizer.monitor.rerun_sink import RerunSink

    calls = []

    class FakeRecording:
        def flush(self, *, timeout_sec):
            calls.append(("flush", timeout_sec))

        def disconnect(self):
            calls.append(("disconnect",))

    class FakeViewerClient:
        url = "rerun+http://127.0.0.1:9876/proxy"

        @classmethod
        def spawn(cls, **kwargs):
            calls.append(("spawn", kwargs))
            return cls()

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(rerun_sink_module, "_local_port_open", lambda port: False)
    sink = RerunSink(
        config=MonitorConfig(
            enabled=True,
            viewer=ViewerConfig(port=9876, detach_process=True),
        )
    )
    sink.rr = types.SimpleNamespace(
        experimental=types.SimpleNamespace(ViewerClient=FakeViewerClient),
        connect_grpc=lambda url, **kwargs: calls.append(("connect", url, kwargs)),
    )
    sink.recording = FakeRecording()
    sink._connect_viewer()
    sink.close()

    assert calls[0][0] == "spawn"
    assert calls[0][1]["detach_process"] is True
    assert not any(call[0] == "close" for call in calls)
