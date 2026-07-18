"""Synthetic resource benchmark for the DP3 telemetry transport."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

import numpy as np
import yaml

from .client import MonitorClient
from .config import MonitorConfig, TelemetryShapes, load_monitor_config


@dataclass
class ModeResult:
    mode: str
    duration_sec: float
    producer_cpu_mean: float | None
    producer_cpu_p95: float | None
    producer_rss_max: int | None
    producer_uss_max: int | None
    telemetry_cpu_mean: float | None
    telemetry_cpu_p95: float | None
    telemetry_rss_max: int | None
    telemetry_uss_max: int | None
    viewer_cpu_mean: float | None
    viewer_cpu_p95: float | None
    viewer_rss_max: int | None
    viewer_uss_max: int | None
    system_cpu_mean: float | None
    system_cpu_p95: float | None
    system_memory_used_max: int | None
    system_memory_available_min: int | None
    gpu_util_mean: float | None
    gpu_util_p95: float | None
    gpu_memory_used_max: int | None
    benchmark_process_gpu_memory_max: int | None
    publish_p50_ms: float | None
    publish_p95_ms: float | None
    publish_p99_ms: float | None
    publish_max_ms: float | None
    plan_p50_ms: float | None
    plan_p95_ms: float | None
    plan_p99_ms: float | None
    plan_max_ms: float | None
    control_publish_p99_ms: float | None
    camera_publish_p99_ms: float | None
    sampled_publish_p99_ms: float | None
    stage_publish_p99_ms: float | None
    channel_publish_latency_ms: dict[str, dict[str, float | None]]
    channel_frames: dict[str, dict[str, int]]
    cycle_work_p50_ms: float | None
    cycle_work_p95_ms: float | None
    cycle_work_p99_ms: float | None
    cycle_work_max_ms: float | None
    cycle_period_p50_ms: float | None
    cycle_period_p95_ms: float | None
    cycle_period_p99_ms: float | None
    cycle_period_max_ms: float | None
    cycle_jitter_p50_ms: float | None
    cycle_jitter_p95_ms: float | None
    cycle_jitter_p99_ms: float | None
    cycle_jitter_max_ms: float | None
    deadline_misses: int
    deadline_miss_ratio: float
    attempted_frames: int
    committed_frames: int
    dropped_frames: int
    overwritten_frames: int
    consumer_processed_frames: int
    consumer_lag_cycles: int
    effective_camera_rate_hz: float
    effective_pointcloud_rate_hz: float
    child_startup_ms: float | None
    shutdown_ms: float | None
    viewer_available: bool
    stage_payload_source: str
    error: str | None = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic DP3 monitor resource benchmark")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--warmup-sec", type=float, default=10.0)
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--stage-duration-sec", type=float, default=5.0)
    parser.add_argument("--output-root", type=Path, default=Path("logs/monitor_benchmark"))
    parser.add_argument("--modes", nargs="+", default=["baseline", "shared-memory/null-sink", "local-rerun-viewer"])
    parser.add_argument("--no-stages", action="store_true")
    parser.add_argument("--viewer-port", type=int, default=None, help="Override the spawned Viewer port for this benchmark")
    args = parser.parse_args(argv)
    if args.warmup_sec < 0 or args.duration_sec <= 0 or args.stage_duration_sec <= 0:
        parser.error("benchmark durations must be non-negative and duration-sec must be positive")
    config_path = args.config or _default_config()
    inference_cfg, monitor_cfg, shapes = _resolve_contract(config_path)
    # Deployment keeps its Viewer open by default. A benchmark must always own
    # and close its synthetic Viewer so repeated runs do not leak port/process
    # state into later measurements.
    monitor_cfg = replace(
        monitor_cfg,
        viewer=replace(monitor_cfg.viewer, detach_process=False),
    )
    if args.viewer_port is not None:
        if not 1 <= args.viewer_port <= 65535:
            parser.error("viewer-port must be in [1, 65535]")
        monitor_cfg = replace(monitor_cfg, viewer=replace(monitor_cfg.viewer, port=int(args.viewer_port)))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[ModeResult] = []
    all_samples: list[dict[str, Any]] = []
    for mode in args.modes:
        result, samples = _run_mode(
            mode,
            monitor_cfg,
            shapes,
            warmup_sec=float(args.warmup_sec),
            duration_sec=float(args.duration_sec),
            stages=False,
        )
        results.append(result)
        all_samples.extend(samples)
    if not args.no_stages:
        stage_cfg = _stage_config(monitor_cfg)
        result, samples = _run_mode(
            "shared-memory/null-sink; raw-cropped=on",
            stage_cfg,
            shapes,
            warmup_sec=0.0,
            duration_sec=float(args.stage_duration_sec),
            stages=True,
        )
        results.append(result)
        all_samples.extend(samples)
    system_info = _system_info(config_path, inference_cfg, monitor_cfg, shapes)
    (output_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(
            {"inference": inference_cfg, "monitor": monitor_cfg.as_dict(), "shapes": asdict(shapes)},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (output_dir / "system_info.json").write_text(json.dumps(system_info, indent=2, sort_keys=True), encoding="utf-8")
    with (output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        if all_samples:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in all_samples for key in row}))
            writer.writeheader()
            writer.writerows(all_samples)
    report = {"system": system_info, "modes": [asdict(result) for result in results]}
    (output_dir / "resource_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "resource_report.md").write_text(_markdown_report(report), encoding="utf-8")
    print(output_dir)
    print((output_dir / "resource_report.md").read_text(encoding="utf-8"))
    return 0 if all(result.error is None for result in results) else 1


def _default_config() -> Path:
    return Path(__file__).resolve().parents[3] / "3D-Diffusion-Policy/diffusion_policy_3d/config/dp3_inference_config.yaml"


def _resolve_contract(path: Path) -> tuple[dict[str, Any], MonitorConfig, TelemetryShapes]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"inference config must be a mapping: {path}")
    inference = dict(raw.get("inference", {}))
    monitor_raw = raw.get("monitor")
    if isinstance(monitor_raw, Mapping):
        rates_raw = monitor_raw.get("rates")
        if isinstance(rates_raw, Mapping) and rates_raw.get("control_hz") == "${inference.rate_hz}":
            # The launcher uses OmegaConf interpolation; this benchmark reads
            # the same public YAML with safe_load and resolves this one field.
            rates_raw = dict(rates_raw)
            rates_raw["control_hz"] = float(inference.get("rate_hz", 10.0))
            monitor_raw = dict(monitor_raw)
            monitor_raw["rates"] = rates_raw
            raw = dict(raw)
            raw["monitor"] = monitor_raw
    monitor = load_monitor_config(raw, inference_rate_hz=float(inference.get("rate_hz", 10.0)))
    shape_meta = raw.get("shape_meta", {})
    point_shape = shape_meta.get("obs", {}).get("point_cloud", {}).get("shape", [2048, 3])
    state_shape = shape_meta.get("obs", {}).get("agent_pos", {}).get("shape", [34])
    action_shape = shape_meta.get("action", {}).get("shape", [14])
    algorithm = raw.get("algorithm", "simple_dp3")
    horizon = raw.get("algorithm_profiles", {}).get(algorithm, {}).get("horizon", 8)
    pointcloud_path = path.parent.parent.parent / str(raw.get("pointcloud", {}).get("config", ""))
    camera_height, camera_width = 480, 640
    if pointcloud_path.is_file():
        try:
            with pointcloud_path.open("r", encoding="utf-8") as handle:
                pc_raw = yaml.safe_load(handle) or {}
            intrinsics = pc_raw.get("camera", {}).get("depth_intrinsics", {})
            camera_height = int(intrinsics.get("height", camera_height))
            camera_width = int(intrinsics.get("width", camera_width))
        except (OSError, TypeError, ValueError):
            pass
    shapes = TelemetryShapes(
        camera_height=camera_height,
        camera_width=camera_width,
        point_count=int(point_shape[0]),
        point_dim=int(point_shape[1]),
        state_dim=int(state_shape[0]),
        action_dim=int(action_shape[0]),
        policy_horizon=int(horizon),
        depth_dtype=monitor.depth_dtype,
        max_raw_points=monitor.display.max_raw_points,
        max_cropped_points=monitor.display.max_cropped_points,
    )
    return inference, monitor, shapes


def _stage_config(config: MonitorConfig) -> MonitorConfig:
    from dataclasses import replace

    return replace(
        config,
        enabled=True,
        payloads=replace(config.payloads, raw_pointcloud=True, cropped_pointcloud=True),
        rates=replace(config.rates, stage_pointcloud_hz=1.0),
    )


def _run_mode(
    mode: str,
    config: MonitorConfig,
    shapes: TelemetryShapes,
    *,
    warmup_sec: float,
    duration_sec: float,
    stages: bool,
) -> tuple[ModeResult, list[dict[str, Any]]]:
    import psutil

    arrays = {
        "state": np.zeros((shapes.state_dim,), dtype=np.float32),
        "rgb": np.zeros((shapes.camera_height, shapes.camera_width, 3), dtype=np.uint8),
        "depth": np.full((shapes.camera_height, shapes.camera_width), 1000, dtype=np.dtype(shapes.depth_dtype)),
        "points": np.zeros((shapes.point_count, shapes.point_dim), dtype=np.float32),
        "horizon": np.zeros((shapes.policy_horizon, shapes.action_dim), dtype=np.float32),
        "action": np.zeros((shapes.action_dim,), dtype=np.float32),
        "raw": np.zeros((shapes.max_raw_points, shapes.point_dim), dtype=np.float32),
        "cropped": np.zeros((shapes.max_cropped_points, shapes.point_dim), dtype=np.float32),
    }
    stage_payload_source = "disabled"
    if stages:
        # Exercise the real optional-stage D2H path with camera-scale tensors;
        # the shared-memory/display arrays remain capped by TelemetryShapes.
        source_points = max(int(shapes.camera_height * shapes.camera_width), int(shapes.max_raw_points), int(shapes.max_cropped_points))
        try:
            import torch

            if torch.cuda.is_available():
                arrays["raw"] = torch.zeros((source_points, shapes.point_dim), dtype=torch.float32, device="cuda")
                arrays["cropped"] = torch.zeros((source_points, shapes.point_dim), dtype=torch.float32, device="cuda")
                stage_payload_source = f"cuda:{source_points}x{shapes.point_dim}"
            else:
                stage_payload_source = f"cpu-fallback:{source_points}x{shapes.point_dim}"
        except Exception as exc:  # noqa: BLE001
            stage_payload_source = f"cpu-fallback:{source_points}x{shapes.point_dim} ({exc})"
        if stage_payload_source.startswith("cpu-fallback"):
            arrays["raw"] = np.zeros((source_points, shapes.point_dim), dtype=np.float32)
            arrays["cropped"] = np.zeros((source_points, shapes.point_dim), dtype=np.float32)
    process = psutil.Process(os.getpid())
    metric_processes: dict[int, Any] = {process.pid: process}
    process.cpu_percent(None)
    psutil.cpu_percent(None)
    gpu_sampler = _GpuSampler()
    telemetry_pid: int | None = None
    client: MonitorClient | None = None
    startup_ms: float | None = None
    viewer_available = mode != "local-rerun-viewer"
    error: str | None = None
    if mode != "baseline":
        sink_kind = "rerun" if mode == "local-rerun-viewer" else "null"
        started = time.perf_counter()
        try:
            client = MonitorClient.create(config, shapes, sink_kind=sink_kind)
            startup_ms = (time.perf_counter() - started) * 1000.0
            telemetry_pid = client.process.pid if client.process is not None else None
            if telemetry_pid is not None:
                telemetry_process = psutil.Process(telemetry_pid)
                metric_processes[telemetry_pid] = telemetry_process
                telemetry_process.cpu_percent(None)
                for child in telemetry_process.children(recursive=True):
                    metric_processes[child.pid] = child
                    child.cpu_percent(None)
            viewer_available = bool(client.process and client.process.ready) if mode == "local-rerun-viewer" else True
            if mode == "local-rerun-viewer" and not viewer_available:
                error = "Rerun child did not become ready; baseline/null-sink still completed"
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)
            viewer_available = False
    period = 1.0 / float(config.rates.control_hz)
    warmup_end = time.perf_counter() + warmup_sec
    end = warmup_end + duration_sec
    next_tick = time.perf_counter()
    next_sample = next_tick
    cycle_times: list[float] = []
    cycle_periods: list[float] = []
    jitters: list[float] = []
    publishes: list[float] = []
    plan_times: list[float] = []
    channel_times: dict[str, list[float]] = {name: [] for name in ("control", "camera", "sampled_pointcloud", "stage_pointcloud")}
    channel_attempted = {name: 0 for name in channel_times}
    channel_committed = {name: 0 for name in channel_times}
    deadline_misses = 0
    attempted = 0
    committed = 0
    camera_committed = 0
    pointcloud_committed = 0
    total_cycles = 0
    measurement_baseline: dict[str, dict[str, int]] | None = None
    measurement_baseline_processed: dict[str, int] = {}
    samples: list[dict[str, Any]] = []
    previous_measured_start: float | None = None
    while time.perf_counter() < end:
        cycle_start = time.perf_counter()
        if cycle_start < next_tick:
            time.sleep(next_tick - cycle_start)
            cycle_start = time.perf_counter()
        next_tick += period
        total_cycles += 1
        measuring = cycle_start >= warmup_end
        if measuring:
            attempted += 1
            if previous_measured_start is not None:
                cycle_periods.append((cycle_start - previous_measured_start) * 1000.0)
            previous_measured_start = cycle_start
            if measurement_baseline is None and client is not None:
                measurement_baseline = client.bus.stats() if client.bus is not None else {}
                heartbeat = client.process.last_heartbeat if client.process is not None else None
                measurement_baseline_processed = dict((heartbeat or {}).get("processed_counts", {}))
        plan_started = time.perf_counter()
        plan = client.plan_cycle(cycle_start) if client is not None else None
        if client is not None and measuring:
            plan_times.append((time.perf_counter() - plan_started) * 1000.0)
        if client is not None:
            publish_start = time.perf_counter()
            result = client.publish_cycle(
                cycle_id=total_cycles,
                measured_state=arrays["state"],
                rgb=arrays["rgb"],
                depth=arrays["depth"],
                depth_scale=0.001,
                sampled_pointcloud=arrays["points"],
                pointcloud_meta={"num_raw_points": shapes.max_raw_points, "num_cropped_points": shapes.max_cropped_points, "num_sampled_points": shapes.point_count},
                stages={"raw": arrays["raw"], "cropped": arrays["cropped"]} if stages else None,
                policy_horizon=arrays["horizon"],
                prediction_id=attempted,
                selected_raw_action=arrays["action"],
                filtered_action=arrays["action"],
                commanded_action=arrays["action"],
                commanded_valid=True,
                send_status="sent",
                plan=plan,
                remaining_slack_ms=1000.0,
            )
            if measuring:
                publishes.append((time.perf_counter() - publish_start) * 1000.0)
                for name, elapsed_ms in result.get("publish_timings_ms", {}).items():
                    if name in channel_times:
                        channel_times[name].append(float(elapsed_ms))
                        channel_attempted[name] += 1
                committed += int(bool(result.get("control")))
                camera_committed += int(bool(result.get("bulk", {}).get("camera")))
                pointcloud_committed += int(bool(result.get("bulk", {}).get("sampled_pointcloud")))
                channel_committed["control"] += int(bool(result.get("control")))
                for name in ("camera", "sampled_pointcloud", "stage_pointcloud"):
                    channel_committed[name] += int(bool(result.get("bulk", {}).get(name)))
        cycle_time = (time.perf_counter() - cycle_start) * 1000.0
        if measuring:
            cycle_times.append(cycle_time)
            jitters.append(abs((cycle_start - (next_tick - period)) * 1000.0))
            if cycle_time > period * 1000.0:
                deadline_misses += 1
        if measuring and cycle_start >= next_sample:
            sample = _sample_resources(process, telemetry_pid, mode, metric_processes, gpu_sampler)
            sample["mode"] = mode
            samples.append(sample)
            next_sample = cycle_start + 0.2
    shutdown_ms: float | None = None
    if client is not None:
        started = time.perf_counter()
        stats = client.stats()
        client.close()
        shutdown_ms = (time.perf_counter() - started) * 1000.0
        channel_stats = stats.get("bus", {})
        baseline_stats = measurement_baseline or {}
        dropped = sum(
            max(0, int(value.get("dropped", 0)) - int(baseline_stats.get(name, {}).get("dropped", 0)))
            for name, value in channel_stats.items()
        )
        overwritten = sum(
            max(0, int(value.get("overwritten", 0)) - int(baseline_stats.get(name, {}).get("overwritten", 0)))
            for name, value in channel_stats.items()
        )
        heartbeat = stats.get("telemetry_heartbeat") or {}
        processed_counts = heartbeat.get("processed_counts", {})
        consumer_processed = sum(
            max(0, int(value) - int(measurement_baseline_processed.get(name, 0)))
            for name, value in processed_counts.items()
        )
        control_published = int(channel_stats.get("control", {}).get("published", 0)) - int(baseline_stats.get("control", {}).get("published", 0))
        control_processed = int(processed_counts.get("control", 0)) - int(measurement_baseline_processed.get("control", 0))
        consumer_lag = max(0, control_published - control_processed)
        effective_camera_rate = camera_committed / duration_sec
        effective_pointcloud_rate = pointcloud_committed / duration_sec
    else:
        dropped = overwritten = consumer_processed = consumer_lag = 0
        effective_camera_rate = effective_pointcloud_rate = 0.0
    producer_samples = [row.get("producer_cpu") for row in samples if row.get("producer_cpu") is not None]
    telemetry_samples = [row.get("telemetry_cpu") for row in samples if row.get("telemetry_cpu") is not None]
    viewer_samples = [row.get("viewer_cpu") for row in samples if row.get("viewer_cpu") is not None]
    if mode == "local-rerun-viewer" and not viewer_samples:
        viewer_available = False
        error = error or "Rerun Viewer process metrics were unavailable"
    channel_stats_report = {
        name: {
            "attempted": int(channel_attempted[name]),
            "committed": int(channel_committed[name]),
            "dropped": max(
                0,
                int(channel_stats.get(name, {}).get("dropped", 0))
                - int(baseline_stats.get(name, {}).get("dropped", 0)),
            ) if client is not None else 0,
            "overwritten": max(
                0,
                int(channel_stats.get(name, {}).get("overwritten", 0))
                - int(baseline_stats.get(name, {}).get("overwritten", 0)),
            ) if client is not None else 0,
            "consumer_processed": max(
                0,
                int(processed_counts.get(name, 0))
                - int(measurement_baseline_processed.get(name, 0)),
            ) if client is not None else 0,
        }
        for name in channel_times
    }
    channel_latency_report = {name: _latency_stats(values) for name, values in channel_times.items()}
    result = ModeResult(
        mode=mode,
        duration_sec=duration_sec,
        producer_cpu_mean=_mean(producer_samples),
        producer_cpu_p95=_percentile(producer_samples, 95),
        producer_rss_max=_max_int([row.get("producer_rss") for row in samples]),
        producer_uss_max=_max_int([row.get("producer_uss") for row in samples]),
        telemetry_cpu_mean=_mean(telemetry_samples),
        telemetry_cpu_p95=_percentile(telemetry_samples, 95),
        telemetry_rss_max=_max_int([row.get("telemetry_rss") for row in samples]),
        telemetry_uss_max=_max_int([row.get("telemetry_uss") for row in samples]),
        viewer_cpu_mean=_mean(viewer_samples),
        viewer_cpu_p95=_percentile(viewer_samples, 95),
        viewer_rss_max=_max_int([row.get("viewer_rss") for row in samples]),
        viewer_uss_max=_max_int([row.get("viewer_uss") for row in samples]),
        system_cpu_mean=_mean([row["system_cpu"] for row in samples if row.get("system_cpu") is not None]),
        system_cpu_p95=_percentile([row["system_cpu"] for row in samples if row.get("system_cpu") is not None], 95),
        system_memory_used_max=_max_int([row.get("system_memory_used") for row in samples]),
        system_memory_available_min=_min_int([row.get("system_memory_available") for row in samples]),
        gpu_util_mean=_mean([row["gpu_util"] for row in samples if row.get("gpu_util") is not None]),
        gpu_util_p95=_percentile([row["gpu_util"] for row in samples if row.get("gpu_util") is not None], 95),
        gpu_memory_used_max=_max_int([row.get("gpu_memory_used") for row in samples]),
        benchmark_process_gpu_memory_max=_max_int([row.get("benchmark_process_gpu_memory") for row in samples]),
        publish_p50_ms=_percentile(publishes, 50),
        publish_p95_ms=_percentile(publishes, 95),
        publish_p99_ms=_percentile(publishes, 99),
        publish_max_ms=max(publishes) if publishes else None,
        plan_p50_ms=_percentile(plan_times, 50),
        plan_p95_ms=_percentile(plan_times, 95),
        plan_p99_ms=_percentile(plan_times, 99),
        plan_max_ms=max(plan_times) if plan_times else None,
        control_publish_p99_ms=_percentile(channel_times["control"], 99),
        camera_publish_p99_ms=_percentile(channel_times["camera"], 99),
        sampled_publish_p99_ms=_percentile(channel_times["sampled_pointcloud"], 99),
        stage_publish_p99_ms=_percentile(channel_times["stage_pointcloud"], 99),
        channel_publish_latency_ms=channel_latency_report,
        channel_frames=channel_stats_report,
        cycle_work_p50_ms=_percentile(cycle_times, 50),
        cycle_work_p95_ms=_percentile(cycle_times, 95),
        cycle_work_p99_ms=_percentile(cycle_times, 99),
        cycle_work_max_ms=max(cycle_times) if cycle_times else None,
        cycle_period_p50_ms=_percentile(cycle_periods, 50),
        cycle_period_p95_ms=_percentile(cycle_periods, 95),
        cycle_period_p99_ms=_percentile(cycle_periods, 99),
        cycle_period_max_ms=max(cycle_periods) if cycle_periods else None,
        cycle_jitter_p50_ms=_percentile(jitters, 50),
        cycle_jitter_p95_ms=_percentile(jitters, 95),
        cycle_jitter_p99_ms=_percentile(jitters, 99),
        cycle_jitter_max_ms=max(jitters) if jitters else None,
        deadline_misses=deadline_misses,
        deadline_miss_ratio=deadline_misses / attempted if attempted else 0.0,
        attempted_frames=attempted,
        committed_frames=committed,
        dropped_frames=dropped,
        overwritten_frames=overwritten,
        consumer_processed_frames=consumer_processed,
        consumer_lag_cycles=consumer_lag,
        effective_camera_rate_hz=effective_camera_rate,
        effective_pointcloud_rate_hz=effective_pointcloud_rate,
        child_startup_ms=startup_ms,
        shutdown_ms=shutdown_ms,
        viewer_available=viewer_available,
        stage_payload_source=stage_payload_source,
        error=error,
    )
    gpu_sampler.close()
    return result, samples


def _sample_resources(
    producer: Any,
    telemetry_pid: int | None,
    mode: str,
    process_cache: dict[int, Any],
    gpu_sampler: "_GpuSampler",
) -> dict[str, Any]:
    import psutil

    row: dict[str, Any] = {"timestamp": time.time(), "producer_pid": producer.pid}
    row.update(_process_metrics(producer, "producer"))
    row["system_cpu"] = float(psutil.cpu_percent(None))
    virtual_memory = psutil.virtual_memory()
    row["system_memory_used"] = int(virtual_memory.used)
    row["system_memory_available"] = int(virtual_memory.available)
    viewer_pids: list[int] = []
    if telemetry_pid:
        try:
            telemetry = process_cache.get(telemetry_pid)
            if telemetry is None:
                telemetry = producer.__class__(telemetry_pid)
                process_cache[telemetry_pid] = telemetry
                telemetry.cpu_percent(None)
            row.update(_process_metrics(telemetry, "telemetry"))
            descendants = telemetry.children(recursive=True)
            viewers = [child for child in descendants if "rerun" in child.name().lower() or "viewer" in child.name().lower()]
            if viewers:
                cached_viewers = []
                for child in viewers:
                    cached = process_cache.get(child.pid)
                    if cached is None:
                        cached = child
                        process_cache[child.pid] = cached
                        cached.cpu_percent(None)
                    cached_viewers.append(cached)
                viewer_details = []
                for child in cached_viewers:
                    if not child.is_running():
                        continue
                    metrics = _process_metrics(child, "viewer")
                    viewer_pids.append(int(child.pid))
                    viewer_details.append({"pid": int(child.pid), "name": child.name(), **metrics})
                if viewer_details:
                    row["viewer_cpu"] = sum(float(item.get("viewer_cpu", 0.0)) for item in viewer_details)
                    row["viewer_rss"] = sum(int(item.get("viewer_rss", 0)) for item in viewer_details)
                    uss_values = [item.get("viewer_uss") for item in viewer_details if item.get("viewer_uss") is not None]
                    row["viewer_uss"] = sum(int(value) for value in uss_values) if uss_values else None
                    row["viewer_threads"] = sum(int(item.get("viewer_threads", 0)) for item in viewer_details)
                    row["viewer_processes_json"] = json.dumps(viewer_details, sort_keys=True)
        except Exception:
            pass
    gpu_pids = [producer.pid, *([telemetry_pid] if telemetry_pid is not None else []), *viewer_pids]
    row.update(gpu_sampler.sample(gpu_pids))
    return row


def _process_metrics(process: Any, prefix: str) -> dict[str, Any]:
    try:
        memory = process.memory_info()
        values = {
            f"{prefix}_cpu": float(process.cpu_percent(None)),
            f"{prefix}_rss": int(memory.rss),
            f"{prefix}_threads": int(process.num_threads()),
        }
        try:
            values[f"{prefix}_uss"] = int(process.memory_full_info().uss)
        except Exception:
            values[f"{prefix}_uss"] = None
        try:
            io = process.io_counters()
            values[f"{prefix}_read_bytes"] = int(io.read_bytes)
            values[f"{prefix}_write_bytes"] = int(io.write_bytes)
        except Exception:
            values[f"{prefix}_read_bytes"] = None
            values[f"{prefix}_write_bytes"] = None
        return values
    except Exception:
        return {}


class _GpuSampler:
    """Best-effort low-overhead NVML sampler used at the 5 Hz resource rate."""

    def __init__(self) -> None:
        self.nvml: Any | None = None
        self.handle: Any | None = None
        self.error: str | None = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as exc:  # noqa: BLE001
            self.error = repr(exc)

    def sample(self, pids: list[int]) -> dict[str, Any]:
        if self.nvml is None or self.handle is None:
            return {"gpu_sample_error": self.error or "NVML unavailable"}
        try:
            utilization = self.nvml.nvmlDeviceGetUtilizationRates(self.handle)
            memory = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
            process_memory: dict[int, int] = {}
            for getter_name in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
                getter = getattr(self.nvml, getter_name, None)
                if not callable(getter):
                    continue
                try:
                    processes = getter(self.handle)
                except Exception:  # noqa: BLE001
                    continue
                for process in processes:
                    used = getattr(process, "usedGpuMemory", None)
                    if used is None or int(used) < 0:
                        continue
                    process_memory[int(process.pid)] = max(process_memory.get(int(process.pid), 0), int(used))
            return {
                "gpu_util": float(utilization.gpu),
                "gpu_memory_used": int(memory.used),
                "benchmark_process_gpu_memory": sum(process_memory.get(int(pid), 0) for pid in set(pids)),
                "gpu_process_memory_json": json.dumps(process_memory, sort_keys=True),
            }
        except Exception as exc:  # noqa: BLE001
            return {"gpu_sample_error": repr(exc)}

    def close(self) -> None:
        if self.nvml is not None:
            try:
                self.nvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass
        self.nvml = None
        self.handle = None


def _latency_stats(values: list[float]) -> dict[str, float | None]:
    return {
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": max(values) if values else None,
    }


def _system_info(config_path: Path, inference: Mapping[str, Any], monitor: MonitorConfig, shapes: TelemetryShapes) -> dict[str, Any]:
    try:
        rerun_version = __import__("importlib.metadata", fromlist=["version"]).version("rerun-sdk")
    except Exception:
        rerun_version = "N/A in benchmark interpreter"
    try:
        import psutil

        cpu = psutil.cpu_count(logical=True)
        memory = psutil.virtual_memory().total
    except Exception:
        cpu = memory = None
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3], text=True).strip()
    except Exception:
        git_sha = "unknown"
    return {
        "git_sha": git_sha,
        "python": sys.version,
        "rerun_sdk": rerun_version,
        "psutil": _module_version("psutil"),
        "os": platform.platform(),
        "cpu_model": _cpu_model(),
        "cpu_logical": cpu,
        "memory_bytes": memory,
        "gpu": _gpu_info(),
        "viewer_mode": monitor.viewer.mode,
        "viewer_port": monitor.viewer.port,
        "display_server": "wayland" if os.environ.get("WAYLAND_DISPLAY") else "x11" if os.environ.get("DISPLAY") else "headless",
        "headless": not bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")),
        "config_path": str(config_path.resolve()),
        "inference_rate_hz": inference.get("rate_hz"),
        "shapes": asdict(shapes),
        "monitor_memory_limit": monitor.viewer.memory_limit,
        "monitor_server_memory_limit": monitor.viewer.server_memory_limit,
    }


def _module_version(name: str) -> str:
    try:
        return __import__("importlib.metadata", fromlist=["version"]).version(name)
    except Exception:
        return "N/A"


def _gpu_info() -> str:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=2)
        return result.stdout.strip() or "N/A"
    except Exception as exc:
        return f"N/A ({exc})"


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _markdown_report(report: Mapping[str, Any]) -> str:
    system = report["system"]
    lines = [
        "# DP3 monitor benchmark",
        "",
        f"- Git SHA: `{system['git_sha']}`",
        f"- Python: `{system['python'].splitlines()[0]}`",
        f"- rerun-sdk: `{system['rerun_sdk']}`; psutil: `{system['psutil']}`",
        f"- OS: `{system['os']}`; CPU: `{system['cpu_model']}`; GPU: `{system['gpu']}`",
        f"- Viewer mode: `{system['viewer_mode']}`; headless: `{system['headless']}`",
        "",
        f"- Display server: `{system['display_server']}`; viewer rows use `{system['viewer_mode']}` on port `{system['viewer_port']}`",
        "",
        "| mode | duration s | producer CPU mean/p95 | producer RSS max | telemetry CPU mean/p95 | telemetry RSS max | Viewer CPU mean/p95 | Viewer RSS max | publish p50/p95/p99/max ms | jitter p50/p95/p99/max ms | deadline misses/ratio | dropped/overwritten | consumer processed/lag | effective camera/pointcloud Hz |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in report["modes"]:
        lines.append(
            "| {mode} | {duration_sec:.2f} | {producer_cpu_mean}/{producer_cpu_p95} | {producer_rss_max} | {telemetry_cpu_mean}/{telemetry_cpu_p95} | {telemetry_rss_max} | {viewer_cpu_mean}/{viewer_cpu_p95} | {viewer_rss_max} | {publish_p50_ms}/{publish_p95_ms}/{publish_p99_ms}/{publish_max_ms} | {cycle_jitter_p50_ms}/{cycle_jitter_p95_ms}/{cycle_jitter_p99_ms}/{cycle_jitter_max_ms} | {deadline_misses}/{deadline_miss_ratio:.4f} | {dropped_frames}/{overwritten_frames} | {consumer_processed_frames}/{consumer_lag_cycles} | {effective_camera_rate_hz:.2f}/{effective_pointcloud_rate_hz:.2f} |".format(**mode)
        )
        if mode.get("error"):
            lines.append(f"- `{mode['mode']}` viewer/child error: `{mode['error']}`")
    lines.extend(["", "## Detailed timing and resources", ""])
    for mode in report["modes"]:
        lines.extend(
            [
                f"### {mode['mode']}",
                "",
                f"- plan p50/p95/p99/max ms: `{mode['plan_p50_ms']}/{mode['plan_p95_ms']}/{mode['plan_p99_ms']}/{mode['plan_max_ms']}`",
                f"- cycle work p50/p95/p99/max ms: `{mode['cycle_work_p50_ms']}/{mode['cycle_work_p95_ms']}/{mode['cycle_work_p99_ms']}/{mode['cycle_work_max_ms']}`",
                f"- cycle period p50/p95/p99/max ms: `{mode['cycle_period_p50_ms']}/{mode['cycle_period_p95_ms']}/{mode['cycle_period_p99_ms']}/{mode['cycle_period_max_ms']}`",
                f"- channel publish latency ms: `{json.dumps(mode['channel_publish_latency_ms'], sort_keys=True)}`",
                f"- channel attempted/committed/dropped/overwritten/processed: `{json.dumps(mode['channel_frames'], sort_keys=True)}`",
                f"- producer/telemetry/Viewer USS max bytes: `{mode['producer_uss_max']}/{mode['telemetry_uss_max']}/{mode['viewer_uss_max']}`",
                f"- system CPU mean/p95 and memory used max/available min: `{mode['system_cpu_mean']}/{mode['system_cpu_p95']}`, `{mode['system_memory_used_max']}/{mode['system_memory_available_min']}`",
                f"- GPU util mean/p95, device memory max, benchmark-process memory max: `{mode['gpu_util_mean']}/{mode['gpu_util_p95']}`, `{mode['gpu_memory_used_max']}`, `{mode['benchmark_process_gpu_memory_max']}`",
                f"- child startup/shutdown ms: `{mode['child_startup_ms']}/{mode['shutdown_ms']}`; stage payload: `{mode['stage_payload_source']}`",
                "",
            ]
        )
    baseline = next((mode for mode in report["modes"] if mode["mode"] == "baseline"), None)
    if baseline is not None:
        lines.extend(["## Baseline comparisons", ""])
        for mode in report["modes"]:
            if mode is baseline:
                continue
            lines.append(
                f"- `{mode['mode']}` vs baseline: producer CPU mean delta "
                f"`{_difference(mode['producer_cpu_mean'], baseline['producer_cpu_mean'])}`, "
                f"producer RSS max delta `{_difference(mode['producer_rss_max'], baseline['producer_rss_max'])}` bytes."
            )
        lines.append("")
    lines.append(
        "The benchmark uses preallocated synthetic payloads (NumPy for normal channels and CUDA tensors for the optional stage D2H case); it never opens RealSense, connects Flexiv, or sends robot actions."
    )
    if system["headless"]:
        lines.append("If a local Viewer row is present, this is a headless viewer; not equivalent to visible native rendering.")
    return "\n".join(lines) + "\n"


def _percentile(values: list[float], percentile: float) -> float | None:
    return None if not values else float(np.percentile(np.asarray(values, dtype=float), percentile))


def _mean(values: list[float]) -> float | None:
    return None if not values else float(mean(values))


def _max_int(values: list[Any]) -> int | None:
    clean = [int(value) for value in values if value is not None]
    return max(clean) if clean else None


def _min_int(values: list[Any]) -> int | None:
    clean = [int(value) for value in values if value is not None]
    return min(clean) if clean else None


def _difference(left: Any, right: Any) -> float | int | None:
    if left is None or right is None:
        return None
    return left - right


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
