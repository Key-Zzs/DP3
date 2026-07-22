#!/usr/bin/env python3
"""Interactive or bounded headless visualization entrypoint."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RMBENCH = ROOT / "third_party" / "sim" / "RMBench"
sys.path.insert(0, str(ROOT / "3D-Diffusion-Policy"))
sys.path.insert(0, str(RMBENCH))
os.chdir(RMBENCH)

import numpy as np

from envs.flexiv_embodiment_smoke import FlexivEmbodimentSmoke


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--mode", choices=("home", "joint-sweep", "gripper-cycle", "action-smoke"), default="home")
    parser.add_argument("--view", choices=("front", "side", "top", "head-camera"), default="head-camera")
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()
    if args.gui and args.headless:
        raise SystemExit("choose --gui or --headless")
    try:
        env = FlexivEmbodimentSmoke(gui=args.gui, seed=0, headless=not args.gui)
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        if "vk::PhysicalDevice" in detail or "Vulkan" in detail:
            print(f"SKIP: {detail}")
            print("manual: conda run -n dp3-rmbench python scripts/rmbench/flexiv/visualize_embodiment.py --gui --mode home --view head-camera")
            return 0
        raise
    try:
        env.home(settle_steps=600)
        print(env.state_adapter.describe())
        if args.mode == "joint-sweep":
            env.joint_sweep(side="left", joint_index=0)
            env.joint_sweep(side="right", joint_index=0)
        elif args.mode == "gripper-cycle":
            env.gripper_cycle(side="left")
            env.gripper_cycle(side="right")
        elif args.mode == "action-smoke":
            for action in (
                np.array([0.002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float32),
                np.array([0, 0.002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float32),
                np.array([0, 0, 0.002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float32),
            ):
                print(env.apply_action(action))
        if args.gui:
            deadline = time.monotonic() + max(0.0, args.seconds)
            while time.monotonic() < deadline:
                env.render()
                time.sleep(0.01)
        else:
            rgb, depth = env.capture_view(args.view)
            print(f"headless view={args.view} rgb={rgb.shape} depth={depth.shape} depth_finite={np.isfinite(depth).all()}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
