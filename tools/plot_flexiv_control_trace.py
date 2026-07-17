#!/usr/bin/env python3
"""Plot policy -> target -> servo command -> TCP feedback from a control trace."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


TRACE_SCHEMA = "flexiv_control_trace_v1"
SIDES = ("left", "right")
AXIS_COLORS = ("tab:red", "tab:green", "tab:blue")
AXIS_NAMES = ("x", "y", "z")


@dataclass
class TraceData:
    metadata: dict[str, Any]
    policy: list[dict[str, Any]]
    servo: list[dict[str, Any]]
    feedback: list[dict[str, Any]]
    footer: dict[str, Any]

    @property
    def all_events(self) -> list[dict[str, Any]]:
        return [*self.policy, *self.servo, *self.feedback]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline Flexiv control-chain plotter. Input must be the "
            "*_control_trace.jsonl sidecar produced by live inference."
        )
    )
    parser.add_argument("--log", type=Path, required=True, help="Control trace JSONL")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Control-chain PNG (default: <log-stem>_control_chain.png)",
    )
    parser.add_argument("--show", action="store_true", help="Also open matplotlib windows")
    parser.add_argument(
        "--no-timing",
        action="store_true",
        help="Do not create the separate timing PNG",
    )
    return parser.parse_args()


def load_trace(path: Path) -> TraceData:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Control trace does not exist: {path}")
    metadata: dict[str, Any] = {}
    footer: dict[str, Any] = {}
    policy: list[dict[str, Any]] = []
    servo: list[dict[str, Any]] = []
    feedback: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            event = record.get("event")
            if event == "trace_start":
                metadata = record
            elif event == "policy_action":
                policy.append(record)
            elif event == "servo_command":
                servo.append(record)
            elif event == "tcp_feedback":
                feedback.append(record)
            elif event == "trace_end":
                footer = record
    if not metadata or metadata.get("schema") != TRACE_SCHEMA:
        raise ValueError(
            f"{path} is not a {TRACE_SCHEMA} log. Pass the *_control_trace.jsonl "
            "sidecar, not the main inference audit JSONL."
        )
    missing = [
        name
        for name, records in (
            ("policy_action", policy),
            ("servo_command", servo),
            ("tcp_feedback", feedback),
        )
        if not records
    ]
    if missing:
        raise ValueError(f"Control trace is incomplete; missing events: {', '.join(missing)}")
    for records in (policy, servo, feedback):
        records.sort(key=lambda record: int(record["monotonic_ns"]))
    return TraceData(metadata, policy, servo, feedback, footer)


def _time_origin_ns(trace: TraceData) -> int:
    return min(int(record["monotonic_ns"]) for record in trace.all_events)


def _times(records: list[dict[str, Any]], origin_ns: int) -> np.ndarray:
    return np.asarray(
        [(int(record["monotonic_ns"]) - origin_ns) * 1e-9 for record in records],
        dtype=float,
    )


def _vectors(
    records: list[dict[str, Any]],
    key: str,
    side: str,
    width: int,
) -> np.ndarray:
    values = np.asarray([record[key][side] for record in records], dtype=float)
    if values.ndim != 2 or values.shape[1] != width or not np.isfinite(values).all():
        raise ValueError(f"Invalid {key}.{side} array shape/content: {values.shape}")
    return values


def _command_key(trace: TraceData) -> str:
    if trace.servo and "command_pose7" in trace.servo[0]:
        return "command_pose7"
    return "smoothed_command_pose7"


def _command_label(trace: TraceData) -> str:
    mode = str(trace.metadata.get("cartesian_command_mode", "servo_thread"))
    return "direct NRT command" if mode == "direct_nrt" else "servo-thread command"


def _rdk_rotations(pose7: np.ndarray) -> Rotation:
    return Rotation.from_quat(pose7[:, [4, 5, 6, 3]])


def _relative_rotvec(pose7: np.ndarray, reference_pose7: np.ndarray) -> np.ndarray:
    reference = Rotation.from_quat(reference_pose7[[4, 5, 6, 3]])
    return (_rdk_rotations(pose7) * reference.inv()).as_rotvec()


def _nearest_indices(source_t: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    upper = np.searchsorted(source_t, query_t, side="left")
    upper = np.clip(upper, 0, len(source_t) - 1)
    lower = np.clip(upper - 1, 0, len(source_t) - 1)
    choose_lower = np.abs(query_t - source_t[lower]) <= np.abs(source_t[upper] - query_t)
    return np.where(choose_lower, lower, upper)


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    delta = _rdk_rotations(first) * _rdk_rotations(second).inv()
    return np.rad2deg(delta.magnitude())


def _plot_control_chain(trace: TraceData, output: Path, *, show: bool) -> None:
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    origin_ns = _time_origin_ns(trace)
    policy_t = _times(trace.policy, origin_ns)
    servo_t = _times(trace.servo, origin_ns)
    feedback_t = _times(trace.feedback, origin_ns)
    figure, axes = plt.subplots(7, 2, figsize=(18, 25), sharex="col")

    for column, side in enumerate(SIDES):
        policy_delta = _vectors(trace.policy, "policy_delta_pose6", side, 6)
        applied_delta = _vectors(trace.policy, "applied_delta_pose6", side, 6)
        target = _vectors(trace.servo, "accumulated_target_pose7", side, 7)
        command = _vectors(trace.servo, _command_key(trace), side, 7)
        actual = _vectors(trace.feedback, "actual_tcp_pose7", side, 7)
        reference = actual[0]

        ax = axes[0, column]
        for axis, color in enumerate(AXIS_COLORS):
            ax.plot(
                policy_t,
                policy_delta[:, axis] * 1000.0,
                "--",
                color=color,
                alpha=0.45,
                label=f"policy {AXIS_NAMES[axis]}",
            )
            ax.plot(
                policy_t,
                applied_delta[:, axis] * 1000.0,
                color=color,
                label=f"applied {AXIS_NAMES[axis]}",
            )
        ax.set_title(f"{side.capitalize()} policy/applied translation delta")
        ax.set_ylabel("delta (mm)")

        ax = axes[1, column]
        for axis, color in enumerate(AXIS_COLORS):
            ax.plot(
                servo_t,
                (target[:, axis] - reference[axis]) * 1000.0,
                ":",
                color=color,
                label=f"target {AXIS_NAMES[axis]}",
            )
            ax.plot(
                servo_t,
                (command[:, axis] - reference[axis]) * 1000.0,
                color=color,
                label=f"command {AXIS_NAMES[axis]}",
            )
            ax.plot(
                feedback_t,
                (actual[:, axis] - reference[axis]) * 1000.0,
                "--",
                color=color,
                alpha=0.8,
                label=f"TCP {AXIS_NAMES[axis]}",
            )
        ax.set_title(f"{side.capitalize()} target / command / actual TCP position")
        ax.set_ylabel("relative position (mm)")

        ax = axes[2, column]
        for axis, color in enumerate(AXIS_COLORS):
            ax.plot(
                policy_t,
                policy_delta[:, axis + 3] * 1000.0,
                "--",
                color=color,
                alpha=0.45,
                label=f"policy r{AXIS_NAMES[axis]}",
            )
            ax.plot(
                policy_t,
                applied_delta[:, axis + 3] * 1000.0,
                color=color,
                label=f"applied r{AXIS_NAMES[axis]}",
            )
        ax.set_title(f"{side.capitalize()} policy/applied rotation delta")
        ax.set_ylabel("delta (mrad)")

        ax = axes[3, column]
        target_rot = np.rad2deg(_relative_rotvec(target, reference))
        command_rot = np.rad2deg(_relative_rotvec(command, reference))
        actual_rot = np.rad2deg(_relative_rotvec(actual, reference))
        for axis, color in enumerate(AXIS_COLORS):
            ax.plot(
                servo_t,
                target_rot[:, axis],
                ":",
                color=color,
                label=f"target r{AXIS_NAMES[axis]}",
            )
            ax.plot(
                servo_t,
                command_rot[:, axis],
                color=color,
                label=f"command r{AXIS_NAMES[axis]}",
            )
            ax.plot(
                feedback_t,
                actual_rot[:, axis],
                "--",
                color=color,
                alpha=0.8,
                label=f"TCP r{AXIS_NAMES[axis]}",
            )
        ax.set_title(f"{side.capitalize()} target / command / actual orientation")
        ax.set_ylabel("relative rotvec (deg)")

        nearest = _nearest_indices(servo_t, feedback_t)
        nearest_target = target[nearest]
        nearest_command = command[nearest]
        ax = axes[4, column]
        ax.plot(
            feedback_t,
            np.linalg.norm(nearest_target[:, :3] - nearest_command[:, :3], axis=1)
            * 1000.0,
            label="target-command",
        )
        ax.plot(
            feedback_t,
            np.linalg.norm(nearest_command[:, :3] - actual[:, :3], axis=1) * 1000.0,
            label="command-TCP",
        )
        ax.plot(
            feedback_t,
            np.linalg.norm(nearest_target[:, :3] - actual[:, :3], axis=1) * 1000.0,
            label="target-TCP",
        )
        ax.set_title(f"{side.capitalize()} translation tracking error")
        ax.set_ylabel("error (mm)")

        ax = axes[5, column]
        ax.plot(
            feedback_t,
            _rotation_error_deg(nearest_target, nearest_command),
            label="target-command",
        )
        ax.plot(
            feedback_t,
            _rotation_error_deg(nearest_command, actual),
            label="command-TCP",
        )
        ax.plot(
            feedback_t,
            _rotation_error_deg(nearest_target, actual),
            label="target-TCP",
        )
        ax.set_title(f"{side.capitalize()} rotation tracking error")
        ax.set_ylabel("error (deg)")

        gripper = np.asarray(
            [record["gripper_command"][side] for record in trace.policy],
            dtype=float,
        )
        axes[6, column].step(policy_t, gripper, where="post", label="gripper command")
        axes[6, column].set_title(f"{side.capitalize()} gripper command")
        axes[6, column].set_ylabel("normalized command")
        axes[6, column].set_xlabel("time since first control event (s)")

    for ax in axes.flat:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=3, loc="best")
    figure.suptitle(
        "Flexiv DP3 control chain: policy delta -> target -> "
        f"{_command_label(trace)} -> TCP",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    if show:
        plt.show()
    plt.close(figure)


def _plot_timing(trace: TraceData, output: Path, *, show: bool) -> None:
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    origin_ns = _time_origin_ns(trace)
    policy_t = _times(trace.policy, origin_ns)
    servo_t = _times(trace.servo, origin_ns)
    feedback_t = _times(trace.feedback, origin_ns)
    figure, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=False)

    for label, times in (
        ("policy action", policy_t),
        ("servo command", servo_t),
        ("TCP feedback", feedback_t),
    ):
        if len(times) > 1:
            axes[0].plot(times[1:], np.diff(times) * 1000.0, label=label)
    axes[0].set_title("Event intervals")
    axes[0].set_ylabel("interval (ms)")

    send_ms = np.asarray([record["send_ms"] for record in trace.servo], dtype=float)
    loop_ms = np.asarray([record["loop_ms"] for record in trace.servo], dtype=float)
    axes[1].plot(servo_t, send_ms, label="dual-arm send")
    axes[1].plot(servo_t, loop_ms, label="servo loop")
    axes[1].set_title("200 Hz servo cost")
    axes[1].set_ylabel("duration (ms)")

    read_ms = np.asarray([record["read_ms"] for record in trace.feedback], dtype=float)
    axes[2].plot(feedback_t, read_ms, label="dual-arm TCP feedback read")
    axes[2].set_title("TCP feedback sampler cost")
    axes[2].set_ylabel("duration (ms)")
    axes[2].set_xlabel("time since first control event (s)")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    if show:
        plt.show()
    plt.close(figure)


def _median_rate_hz(records: list[dict[str, Any]]) -> float:
    timestamps = np.asarray([record["monotonic_ns"] for record in records], dtype=np.int64)
    if len(timestamps) < 2:
        return float("nan")
    median_interval_s = float(np.median(np.diff(timestamps))) * 1e-9
    return 1.0 / median_interval_s if median_interval_s > 0.0 else float("nan")


def main() -> int:
    args = parse_args()
    trace_path = args.log.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else trace_path.with_name(f"{trace_path.stem}_control_chain.png")
    )
    timing_output = output.with_name(f"{output.stem}_timing.png")
    trace = load_trace(trace_path)
    _plot_control_chain(trace, output, show=args.show)
    if not args.no_timing:
        _plot_timing(trace, timing_output, show=args.show)

    dropped = int(trace.footer.get("dropped_records", 0))
    print(f"control plot: {output}")
    if not args.no_timing:
        print(f"timing plot:  {timing_output}")
    print(
        "records: "
        f"policy={len(trace.policy)} ({_median_rate_hz(trace.policy):.1f} Hz), "
        f"servo={len(trace.servo)} ({_median_rate_hz(trace.servo):.1f} Hz), "
        f"feedback={len(trace.feedback)} ({_median_rate_hz(trace.feedback):.1f} Hz), "
        f"dropped={dropped}"
    )
    if dropped:
        print(
            "WARNING: the trace dropped records; increase control_debug.queue_size "
            "before relying on high-frequency timing conclusions."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
