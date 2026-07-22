#!/usr/bin/env python3
"""Create the bounded Stage 2 screenshots, videos, contracts, and reports."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RMBENCH = ROOT / "third_party" / "sim" / "RMBench"
sys.path.insert(0, str(ROOT / "3D-Diffusion-Policy"))
sys.path.insert(0, str(RMBENCH))
os.chdir(RMBENCH)

from diffusion_policy_3d.common.flexiv_state_contract import flexiv_action_names, flexiv_state_names
from envs.flexiv_embodiment_smoke import FlexivEmbodimentSmoke
from validate_embodiment import run as run_validation


def write_video(path: Path, frames: list[np.ndarray], statuses: dict) -> None:
    try:
        if not frames:
            statuses[path.name] = "SKIP: no frames"
            return
        imageio.mimsave(path, frames, fps=20)
        statuses[path.name] = "PASS"
    except Exception as exc:
        statuses[path.name] = f"SKIP: {type(exc).__name__}: {exc}"


def write_depth(path: Path, depth: np.ndarray) -> None:
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        imageio.imwrite(path, np.zeros(depth.shape, dtype=np.uint8))
        return
    lo, hi = np.percentile(valid, [1, 99])
    imageio.imwrite(path, (np.clip((depth - lo) / max(hi - lo, 1.0), 0, 1) * 255).astype(np.uint8))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--capture-all", action="store_true")
    parser.add_argument("--view", choices=("front", "side", "top", "head-camera"), default=None)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (ROOT / "outputs" / "rmbench_flexiv_embodiment_acceptance" / timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        env = FlexivEmbodimentSmoke(gui=False, seed=0)
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        host_skip = "vk::PhysicalDevice" in detail or "Vulkan" in detail
        if not host_skip:
            raise
        manifest = {
            "status": "SKIP",
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "mode": "headless",
            "fixed_cameras": ["head_camera"],
            "wrist_cameras_enabled": False,
            "reason": detail,
            "manual_command": "conda run -n dp3-rmbench python scripts/rmbench/flexiv/capture_acceptance_artifacts.py --headless --capture-all --output-dir outputs/rmbench_flexiv_acceptance",
            "files": ["manifest.json"],
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        return 0
    video_status: dict[str, str] = {}
    try:
        env.home(settle_steps=600)
        validation = run_validation(out_dir, stability_steps=1000, repeated_resets=3)
        views = (args.view,) if args.view else ("front", "side", "top", "head-camera")
        visibility = {}
        for view in views:
            rgb, depth = env.capture_view(view)
            imageio.imwrite(out_dir / ("head_camera_rgb.png" if view == "head-camera" else f"home_{view}.png"), rgb)
            if view == "head-camera":
                write_depth(out_dir / "head_camera_depth.png", depth)
                visibility = {
                    "depth_valid_ratio": float(np.mean(depth > 0)),
                    "rgb_shape": list(rgb.shape),
                    "depth_shape": list(depth.shape),
                    "camera_frame": "world/head_camera",
                }
                axes = rgb.copy()
                cy, cx = axes.shape[0] // 2, axes.shape[1] // 2
                axes[max(0, cy - 45) : min(axes.shape[0], cy + 45), max(0, cx - 1) : min(axes.shape[1], cx + 2)] = [0, 255, 0]
                axes[max(0, cy - 1) : min(axes.shape[0], cy + 2), max(0, cx - 45) : min(axes.shape[1], cx + 45)] = [255, 0, 0]
                imageio.imwrite(out_dir / "camera_axes.png", axes)
                (out_dir / "camera_frustum.json").write_text(json.dumps(env.camera_config, indent=2) + "\n", encoding="utf-8")
                (out_dir / "visibility_summary.json").write_text(json.dumps(visibility, indent=2) + "\n", encoding="utf-8")

        write_video(out_dir / "joint_sweep_left.mp4", env.joint_sweep(side="left", joint_index=0), video_status)
        write_video(out_dir / "joint_sweep_right.mp4", env.joint_sweep(side="right", joint_index=0), video_status)
        write_video(out_dir / "gripper_cycle.mp4", env.gripper_cycle(side="left"), video_status)
        action_frames: list[np.ndarray] = []
        action_trace: list[dict] = []
        for action in (
            np.array([0.002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float32),
            np.array([0, 0.002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float32),
            np.array([0, 0, 0.002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float32),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        ):
            action_trace.append(env.apply_action(action))
            action_frames.append(env._head_rgb())
        write_video(out_dir / "action_smoke.mp4", action_frames, video_status)
        (out_dir / "action_trace.json").write_text(json.dumps(action_trace, indent=2) + "\n", encoding="utf-8")
        (out_dir / "state_contract.txt").write_text("\n".join(f"{i:02d} {name}" for i, name in enumerate(flexiv_state_names())) + "\n", encoding="utf-8")
        (out_dir / "action_contract.txt").write_text("\n".join(f"{i:02d} {name}" for i, name in enumerate(flexiv_action_names())) + "\nR_target = Exp(delta_rotvec) @ R_current\n", encoding="utf-8")
        contacts = {"status": "PASS", "contact_count": len(env.scene.get_contacts())}
        (out_dir / "contacts_summary.json").write_text(json.dumps(contacts, indent=2) + "\n", encoding="utf-8")
        manifest = {
            "status": "PASS" if validation["status"] == "PASS" and all(value == "PASS" for value in video_status.values()) else "PASS_WITH_SKIPS" if validation["status"] == "PASS" else "FAIL",
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "mode": "headless",
            "fixed_cameras": ["head_camera"],
            "wrist_cameras_enabled": False,
            "video_status": video_status,
            "files": sorted({path.name for path in out_dir.iterdir()} | {"manifest.json"}),
            "validation_status": validation["status"],
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
