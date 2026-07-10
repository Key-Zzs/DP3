#!/usr/bin/env python3
"""Non-blocking live monitor process for DP3 perception stages."""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import queue
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


LOGGER = logging.getLogger("flexiv_dp3_live_viewer")


@dataclass(frozen=True)
class ViewerConfig:
    title: str = "DP3 Live Perception"
    width: int = 1280
    height: int = 960
    camera_width: int = 640
    camera_height: int = 480
    camera_fx: float = 640.0
    camera_fy: float = 640.0
    camera_cx: float = 319.5
    camera_cy: float = 239.5
    depth_scale: float = 0.001
    point_size: float = 3.0


class LiveVisualizationPublisher:
    """Publish latest-only visualization jobs without waiting in the control loop."""

    def __init__(
        self,
        *,
        rate_hz: float,
        max_raw_points: int,
        max_cropped_points: int,
        viewer_config: ViewerConfig,
    ) -> None:
        self.rate_hz = float(rate_hz)
        self.max_raw_points = int(max_raw_points)
        self.max_cropped_points = int(max_cropped_points)
        self.viewer_config = viewer_config
        self._period_s = 1.0 / self.rate_hz
        self._next_publish_at = 0.0
        self._pending: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._context = mp.get_context("spawn")
        self._frames: Any = self._context.Queue(maxsize=1)
        self._process = self._context.Process(
            target=run_live_viewer_process,
            args=(self._frames, viewer_config),
            name="flexiv_dp3_live_viewer",
            daemon=True,
        )
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="flexiv_dp3_visualization_publisher",
            daemon=True,
        )
        self.enqueued_frames = 0
        self.prepared_frames = 0
        self.dropped_frames = 0
        self._viewer_exit_reported = False
        self._started = False

    def start(self) -> None:
        self._process.start()
        self._started = True
        self._frames.cancel_join_thread()
        self._worker.start()

    @property
    def viewer_alive(self) -> bool:
        return bool(self._started and self._process.is_alive())

    @property
    def viewer_pid(self) -> int | None:
        return self._process.pid if self._started else None

    def maybe_publish(
        self,
        *,
        step_idx: int,
        depth: Any,
        stages: Mapping[str, Any],
        sampled_point_cloud: np.ndarray,
        pointcloud_meta: Mapping[str, Any],
    ) -> dict[str, Any]:
        now = time.monotonic()
        due = now >= self._next_publish_at
        alive = self.viewer_alive
        enqueued = False
        dropped = False

        if due:
            self._next_publish_at = now + self._period_s
            if alive:
                job = {
                    "step": int(step_idx),
                    "depth": depth,
                    "raw": stages["raw"],
                    "cropped": stages["cropped"],
                    "sampled": sampled_point_cloud,
                    "num_raw_points": int(pointcloud_meta["num_raw_points"]),
                    "num_cropped_points": int(pointcloud_meta["num_cropped_points"]),
                    "num_sampled_points": int(pointcloud_meta["num_sampled_points"]),
                }
                enqueued, dropped = _put_latest(self._pending, job)
                if enqueued:
                    self.enqueued_frames += 1
                if dropped:
                    self.dropped_frames += 1
            elif not self._viewer_exit_reported:
                LOGGER.warning(
                    "Live viewer exited with code %s; inference will continue without visualization",
                    self._process.exitcode,
                )
                self._viewer_exit_reported = True

        return {
            "enabled": True,
            "rate_hz": self.rate_hz,
            "due": due,
            "enqueued": enqueued,
            "viewer_alive": alive,
            "dropped_current": dropped,
            "enqueued_frames": self.enqueued_frames,
            "prepared_frames": self.prepared_frames,
            "dropped_frames": self.dropped_frames,
            "max_raw_display_points": self.max_raw_points,
            "max_cropped_display_points": self.max_cropped_points,
        }

    def close(self) -> None:
        if not self._started:
            return
        self._stop_event.set()
        _put_latest(self._pending, None)
        self._worker.join(timeout=2.0)
        _put_latest(self._frames, None)
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
        try:
            self._frames.close()
        except (OSError, ValueError):
            pass

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._pending.get(timeout=0.1)
            except queue.Empty:
                continue
            if job is None:
                break
            if not self.viewer_alive:
                continue
            try:
                frame = _prepare_frame(
                    job,
                    max_raw_points=self.max_raw_points,
                    max_cropped_points=self.max_cropped_points,
                )
            except Exception:  # noqa: BLE001
                LOGGER.exception("Failed to prepare a live visualization frame")
                continue
            queued, dropped = _put_latest(self._frames, frame)
            if queued:
                self.prepared_frames += 1
            if dropped:
                self.dropped_frames += 1


def _put_latest(target_queue: Any, item: Any) -> tuple[bool, bool]:
    """Insert without waiting, replacing one queued stale item when necessary."""

    try:
        target_queue.put_nowait(item)
        return True, False
    except queue.Full:
        pass

    try:
        target_queue.get_nowait()
    except queue.Empty:
        return False, False

    try:
        target_queue.put_nowait(item)
        return True, True
    except queue.Full:
        return False, True


def _prepare_frame(
    job: Mapping[str, Any],
    *,
    max_raw_points: int,
    max_cropped_points: int,
) -> dict[str, Any]:
    return {
        "step": int(job["step"]),
        "depth": _to_numpy(job["depth"]),
        "raw": _point_cloud_to_numpy(job["raw"], max_raw_points),
        "cropped": _point_cloud_to_numpy(job["cropped"], max_cropped_points),
        "sampled": _point_cloud_to_numpy(job["sampled"], None),
        "num_raw_points": int(job["num_raw_points"]),
        "num_cropped_points": int(job["num_cropped_points"]),
        "num_sampled_points": int(job["num_sampled_points"]),
        "prepared_at": time.time(),
    }


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value).copy()


def _point_cloud_to_numpy(value: Any, max_points: int | None) -> np.ndarray:
    if not hasattr(value, "shape") or len(value.shape) != 2:
        raise ValueError("Visualization point cloud must be a 2D array")
    point_count = int(value.shape[0])
    if max_points is not None and point_count > int(max_points):
        stride = max(1, math.ceil(point_count / int(max_points)))
        value = value[::stride][: int(max_points)]
    point_cloud = _to_numpy(value).astype(np.float32, copy=False)
    if point_cloud.ndim != 2 or point_cloud.shape[1] not in {3, 6}:
        raise ValueError(f"Visualization point cloud must be N x 3 or N x 6, got {point_cloud.shape}")
    if not np.isfinite(point_cloud).all():
        raise ValueError("Visualization point cloud contains NaN or Inf")
    return point_cloud


def run_live_viewer_process(frame_queue: Any, config: ViewerConfig) -> None:
    """Run the stable image monitor in a spawned process.

    The public name is retained so existing deployment commands remain valid.
    """

    try:
        import tkinter as tk
        from PIL import Image, ImageDraw, ImageFont, ImageTk

        root = tk.Tk()
        viewer = _TkQuadViewer(
            root=root,
            tk=tk,
            image_module=Image,
            image_draw_module=ImageDraw,
            image_font_module=ImageFont,
            image_tk_module=ImageTk,
            frame_queue=frame_queue,
            config=config,
        )
        viewer.run()
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        raise SystemExit(1) from None


class _TkQuadViewer:
    def __init__(
        self,
        *,
        root: Any,
        tk: Any,
        image_module: Any,
        image_draw_module: Any,
        image_font_module: Any,
        image_tk_module: Any,
        frame_queue: Any,
        config: ViewerConfig,
    ) -> None:
        self.root = root
        self.tk = tk
        self.image_module = image_module
        self.image_draw_module = image_draw_module
        self.image_font_module = image_font_module
        self.image_tk_module = image_tk_module
        self.frame_queue = frame_queue
        self.config = config
        self.done = False
        self._photo: Any = None
        try:
            self._font = image_font_module.truetype(
                "DejaVuSans.ttf",
                max(16, int(round(22 * config.width / 1280))),
            )
        except OSError:
            self._font = image_font_module.load_default()

        root.title(config.title)
        root.geometry(f"{int(config.width)}x{int(config.height)}")
        root.protocol("WM_DELETE_WINDOW", self.stop)
        root.bind("<Button>", lambda _event: "break")
        root.bind("<B1-Motion>", lambda _event: "break")
        self.image_label = tk.Label(
            root,
            background="#181a1e",
            borderwidth=0,
            highlightthickness=0,
        )
        self.image_label.pack(fill=tk.BOTH, expand=True)

    def run(self) -> None:
        self.root.after(0, self._poll)
        self.root.mainloop()

    def stop(self) -> None:
        if self.done:
            return
        self.done = True
        try:
            self.root.destroy()
        except self.tk.TclError:
            pass

    def _poll(self) -> None:
        if self.done:
            return
        payload: Mapping[str, Any] | None = None
        while True:
            try:
                newer = self.frame_queue.get_nowait()
            except queue.Empty:
                break
            if newer is None:
                self.stop()
                return
            payload = newer
        if payload is not None:
            self._update(payload)
        if not self.done:
            self.root.after(20, self._poll)

    def _update(self, payload: Mapping[str, Any]) -> None:
        step = int(payload["step"])
        depth = _colorize_depth(np.asarray(payload["depth"]), self.config.depth_scale)

        point_images: dict[str, np.ndarray] = {}
        for key in ("raw", "cropped", "sampled"):
            point_cloud = np.asarray(payload[key], dtype=np.float32)
            point_images[key] = _render_point_cloud_image(point_cloud, self.config)

        monitor = _compose_monitor_image(
            depth,
            point_images["raw"],
            point_images["cropped"],
            point_images["sampled"],
        )
        labels = (
            f"Depth | step {step} | {depth.shape[1]}x{depth.shape[0]}",
            (
                f"Raw point cloud | step {step} | display "
                f"{payload['raw'].shape[0]} / {int(payload['num_raw_points'])}"
            ),
            (
                f"Cropped point cloud | step {step} | display "
                f"{payload['cropped'].shape[0]} / {int(payload['num_cropped_points'])}"
            ),
            (
                f"Sampled policy input | step {step} | display "
                f"{payload['sampled'].shape[0]} / {int(payload['num_sampled_points'])}"
            ),
        )
        image = self.image_module.fromarray(monitor)
        draw = self.image_draw_module.Draw(image)
        pane_width = int(self.config.camera_width)
        pane_height = int(self.config.camera_height)
        for text_value, (x, y) in zip(
            labels,
            ((0, 0), (pane_width, 0), (0, pane_height), (pane_width, pane_height)),
        ):
            draw.rectangle((x, y, x + pane_width, y + 34), fill=(12, 13, 15))
            draw.text((x + 8, y + 5), text_value, font=self._font, fill=(235, 238, 242))
        if image.size != (int(self.config.width), int(self.config.height)):
            image = image.resize(
                (int(self.config.width), int(self.config.height)),
                self.image_module.Resampling.BILINEAR,
            )
        photo = self.image_tk_module.PhotoImage(image=image)
        self._photo = photo
        self.image_label.configure(image=photo)


def _colorize_depth(depth: np.ndarray, depth_scale: float) -> np.ndarray:
    depth = np.asarray(depth)
    depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError(f"Depth frame must be H x W, got {depth.shape}")
    meters = depth.astype(np.float32, copy=False) * float(depth_scale)
    valid = np.isfinite(meters) & (meters > 0.0)
    normalized = np.zeros_like(meters, dtype=np.float32)
    if bool(valid.any()):
        values = meters[valid]
        near, far = np.percentile(values, [1.0, 99.0])
        if far <= near:
            far = near + 1e-3
        normalized[valid] = np.clip((meters[valid] - near) / (far - near), 0.0, 1.0)
    colors = _turbo_like(normalized)
    colors[~valid] = 0
    return np.ascontiguousarray(np.round(colors * 255.0).astype(np.uint8))


def _render_point_cloud_image(
    point_cloud: np.ndarray,
    config: ViewerConfig,
) -> np.ndarray:
    """Render a fixed camera-space projection for a non-interactive monitor pane."""

    height = int(config.camera_height)
    width = int(config.camera_width)
    image = np.full((height, width, 3), (24, 26, 30), dtype=np.uint8)
    if point_cloud.shape[0] == 0:
        return image

    xyz = point_cloud[:, :3].astype(np.float32, copy=False)
    valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 1e-6)
    if not bool(valid.any()):
        return image

    xyz = xyz[valid]
    colors = _point_colors(point_cloud)[valid]
    z = xyz[:, 2]
    x = np.rint(float(config.camera_fx) * xyz[:, 0] / z + float(config.camera_cx)).astype(
        np.int32
    )
    y = np.rint(float(config.camera_fy) * xyz[:, 1] / z + float(config.camera_cy)).astype(
        np.int32
    )
    in_frame = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not bool(in_frame.any()):
        return image

    x = x[in_frame]
    y = y[in_frame]
    z = z[in_frame]
    rgb = np.round(np.clip(colors[in_frame], 0.0, 1.0) * 255.0).astype(np.uint8)
    order = np.argsort(z)[::-1]
    x = x[order]
    y = y[order]
    rgb = rgb[order]

    radius = max(0, min(3, int(round(float(config.point_size) / 2.0))))
    for offset_y in range(-radius, radius + 1):
        yy = y + offset_y
        valid_y = (yy >= 0) & (yy < height)
        for offset_x in range(-radius, radius + 1):
            xx = x + offset_x
            visible = valid_y & (xx >= 0) & (xx < width)
            image[yy[visible], xx[visible]] = rgb[visible]
    return np.ascontiguousarray(image)


def _compose_monitor_image(
    depth: np.ndarray,
    raw: np.ndarray,
    cropped: np.ndarray,
    sampled: np.ndarray,
) -> np.ndarray:
    if not (depth.shape == raw.shape == cropped.shape == sampled.shape):
        raise ValueError(
            "Visualization panes must have matching shapes: "
            f"depth={depth.shape}, raw={raw.shape}, cropped={cropped.shape}, "
            f"sampled={sampled.shape}"
        )
    height, width = depth.shape[:2]
    monitor = np.empty((height * 2, width * 2, 3), dtype=np.uint8)
    monitor[:height, :width] = depth
    monitor[:height, width:] = raw
    monitor[height:, :width] = cropped
    monitor[height:, width:] = sampled
    seam = max(1, min(4, min(height, width) // 160))
    monitor[height - seam : height + seam] = 0
    monitor[:, width - seam : width + seam] = 0
    return np.ascontiguousarray(monitor)


def _point_colors(point_cloud: np.ndarray) -> np.ndarray:
    if point_cloud.shape[1] == 6:
        colors = point_cloud[:, 3:6].astype(np.float64, copy=False)
        if colors.size and float(colors.max()) > 1.0:
            colors = colors / 255.0
        return np.clip(colors, 0.0, 1.0)
    z = point_cloud[:, 2]
    if z.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    low, high = np.percentile(z, [1.0, 99.0])
    if high <= low:
        high = low + 1e-6
    normalized = np.clip((z - low) / (high - low), 0.0, 1.0)
    return _turbo_like(normalized).astype(np.float64, copy=False)


def _turbo_like(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    anchors = np.asarray(
        [
            [0.10, 0.15, 0.75],
            [0.00, 0.75, 0.95],
            [0.15, 0.85, 0.35],
            [0.98, 0.82, 0.10],
            [0.85, 0.10, 0.10],
        ],
        dtype=np.float32,
    )
    scaled = np.clip(values, 0.0, 1.0) * float(len(anchors) - 1)
    lower = np.floor(scaled).astype(np.int64)
    upper = np.minimum(lower + 1, len(anchors) - 1)
    weight = (scaled - lower)[..., None]
    return anchors[lower] * (1.0 - weight) + anchors[upper] * weight


__all__ = [
    "LiveVisualizationPublisher",
    "ViewerConfig",
    "run_live_viewer_process",
]
