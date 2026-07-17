from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_adapter():
    package_name = "_test_flexiv_control_trace_interface"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [  # type: ignore[attr-defined]
            str(ROOT / "third_party/real/dual_flexiv_rizon4s/interface")
        ]
        sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.flexiv_dual_arm")


def _load_plotter():
    path = ROOT / "tools/plot_flexiv_control_trace.py"
    spec = importlib.util.spec_from_file_location("test_flexiv_control_trace_plotter", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_html_plotter():
    tools_dir = ROOT / "tools"
    sys.path.insert(0, str(tools_dir))
    path = tools_dir / "plot_flexiv_control_trace_html.py"
    spec = importlib.util.spec_from_file_location(
        "test_flexiv_control_trace_html_plotter",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


adapter = _load_adapter()
plotter = _load_plotter()
html_plotter = _load_html_plotter()


def _identity_pose(x: float = 0.0) -> list[float]:
    return [x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def test_async_control_trace_records_policy_target_without_blocking_disk_io(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    config = adapter.FlexivDualArmConfig(
        debug=False,
        use_cartesian_servo_thread=True,
        control_debug_enabled=True,
        control_debug_log_path=str(trace_path),
        control_debug_queue_size=32,
    )
    robot = adapter.FlexivDualArm(config)
    robot.start_control_debug_trace()
    robot.set_control_debug_context({"step": 7, "chunk_index": 2})
    action = {
        **{f"left_delta_ee_pose.{axis}": 0.0 for axis in adapter.AXES},
        **{f"right_delta_ee_pose.{axis}": 0.0 for axis in adapter.AXES},
        "left_gripper_cmd": 1.0,
        "right_gripper_cmd": 0.5,
    }
    action["left_delta_ee_pose.x"] = 0.001
    trace = robot._send_cartesian_delta(action)
    robot._trace_policy_action(action, trace)
    robot._stop_control_debug_trace()

    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "trace_start",
        "policy_action",
        "trace_end",
    ]
    policy = records[1]
    np.testing.assert_allclose(policy["policy_delta_pose6"]["left"], [0.001, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(
        policy["accumulated_target_pose7"]["left"],
        _identity_pose(0.001),
    )
    assert policy["inference"] == {"step": 7, "chunk_index": 2}
    assert records[-1]["dropped_records"] == 0


def test_direct_nrt_control_trace_is_independent_of_servo_thread(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "direct_trace.jsonl"
    config = adapter.FlexivDualArmConfig(
        debug=False,
        use_cartesian_servo_thread=False,
        control_debug_enabled=True,
        control_debug_log_path=str(trace_path),
        control_debug_queue_size=32,
    )
    robot = adapter.FlexivDualArm(config)
    sent: list[tuple[np.ndarray, np.ndarray]] = []
    robot._send_cartesian_pose_targets = lambda left, right: sent.append(  # type: ignore[method-assign]
        (left.copy(), right.copy())
    )
    robot.start_control_debug_trace()
    action = {
        **{f"left_delta_ee_pose.{axis}": 0.0 for axis in adapter.AXES},
        **{f"right_delta_ee_pose.{axis}": 0.0 for axis in adapter.AXES},
    }
    action["right_delta_ee_pose.z"] = 0.002
    trace = robot._send_cartesian_delta(action)
    robot._trace_policy_action(action, trace)
    robot._stop_control_debug_trace()

    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "trace_start",
        "policy_action",
        "servo_command",
        "trace_end",
    ]
    assert len(sent) == 1
    assert records[0]["cartesian_command_mode"] == "direct_nrt"
    command = records[2]
    assert command["command_source"] == "direct_nrt"
    expected_right = np.asarray(_identity_pose())
    expected_right[2] = 0.002
    np.testing.assert_allclose(
        command["accumulated_target_pose7"]["right"],
        expected_right,
    )
    np.testing.assert_allclose(
        command["command_pose7"]["right"],
        command["accumulated_target_pose7"]["right"],
    )
    assert command["send_ms"] >= 0.0


def test_plotter_loads_complete_control_chain_and_writes_png(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    records = [
        {
            "event": "trace_start",
            "schema": plotter.TRACE_SCHEMA,
            "monotonic_ns": 1_000_000_000,
        }
    ]
    for index in range(2):
        x = 0.001 * index
        records.extend(
            [
                {
                    "event": "policy_action",
                    "schema": plotter.TRACE_SCHEMA,
                    "monotonic_ns": 1_000_000_000 + index * 100_000_000,
                    "policy_delta_pose6": {"left": [x, 0, 0, 0, 0, 0], "right": [0] * 6},
                    "applied_delta_pose6": {"left": [x, 0, 0, 0, 0, 0], "right": [0] * 6},
                    "gripper_command": {"left": 1.0, "right": 1.0},
                },
                {
                    "event": "servo_command",
                    "schema": plotter.TRACE_SCHEMA,
                    "monotonic_ns": 1_010_000_000 + index * 100_000_000,
                    "accumulated_target_pose7": {
                        "left": _identity_pose(x),
                        "right": _identity_pose(),
                    },
                    "smoothed_command_pose7": {
                        "left": _identity_pose(0.8 * x),
                        "right": _identity_pose(),
                    },
                    "send_ms": 0.3,
                    "loop_ms": 0.4,
                },
                {
                    "event": "tcp_feedback",
                    "schema": plotter.TRACE_SCHEMA,
                    "monotonic_ns": 1_020_000_000 + index * 100_000_000,
                    "actual_tcp_pose7": {
                        "left": _identity_pose(0.7 * x),
                        "right": _identity_pose(),
                    },
                    "read_ms": 0.2,
                },
            ]
        )
    records.append(
        {
            "event": "trace_end",
            "schema": plotter.TRACE_SCHEMA,
            "monotonic_ns": 1_300_000_000,
            "dropped_records": 0,
        }
    )
    trace_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    trace = plotter.load_trace(trace_path)
    output = tmp_path / "control.png"
    timing = tmp_path / "timing.png"
    plotter._plot_control_chain(trace, output, show=False)
    plotter._plot_timing(trace, timing, show=False)

    assert output.stat().st_size > 0
    assert timing.stat().st_size > 0

    interactive = tmp_path / "interactive.html"
    _, charts = html_plotter.write_html(
        trace_path,
        interactive,
        plotly_js="cdn",
    )
    document = interactive.read_text(encoding="utf-8")
    assert len(charts) == 17
    assert 'id="chart-select"' in document
    assert "event.shiftKey" in document
    assert "plotly_click" in document
    assert '"dragmode":"pan"' in document
    assert "hovermode" in document
