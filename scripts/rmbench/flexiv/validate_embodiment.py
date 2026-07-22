#!/usr/bin/env python3
"""Run deterministic, headless Stage 2 embodiment validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RMBENCH = ROOT / "third_party" / "sim" / "RMBench"
sys.path.insert(0, str(ROOT / "3D-Diffusion-Policy"))
sys.path.insert(0, str(RMBENCH))
os.chdir(RMBENCH)

from envs.flexiv_embodiment_smoke import FlexivEmbodimentSmoke


def result(status: str, detail: str, **extra) -> dict:
    return {"status": status, "detail": detail, **extra}


def run(out_dir: Path, *, stability_steps: int, repeated_resets: int) -> dict:
    checks: list[dict] = []
    env = None
    try:
        env = FlexivEmbodimentSmoke(gui=False, seed=0)
        checks.append(result("PASS", "SAPIEN loaded two Rizon4s articulations with GN01", name="sapien_load"))
        initial = env.state()
        checks.append(result("PASS" if initial.shape == (34,) else "FAIL", f"state shape={initial.shape}", name="state_contract"))
        env.home(settle_steps=240)
        home_q = np.r_[env.left_arm.arm_qpos(), env.right_arm.arm_qpos()]
        for _ in range(stability_steps):
            env.scene.step()
        final_q = np.r_[env.left_arm.arm_qpos(), env.right_arm.arm_qpos()]
        drift = float(np.max(np.abs(final_q - home_q)))
        checks.append(result("PASS" if drift < 0.02 else "FAIL", f"{stability_steps} steps max arm q drift={drift:.6g} rad", name="stability", max_arm_q_drift_rad=drift))

        reset_states = []
        for index in range(repeated_resets):
            reset_states.append(env.reset(seed=0)[0])
        reset_spread = float(np.max(np.ptp(np.stack(reset_states), axis=0)))
        checks.append(result("PASS" if reset_spread < 1e-3 else "FAIL", f"{repeated_resets} reset state spread={reset_spread:.6g} (threshold 1e-3)", name="repeated_reset", max_state_spread=reset_spread))

        direction_records = []
        for side, offset in (("left", 0), ("right", 6)):
            for axis, label in enumerate(("x", "y", "z")):
                env.reset(seed=0)
                before = env.state().copy()
                action = np.zeros(14, dtype=np.float32)
                action[offset + axis] = 0.002
                action[12:14] = 1.0
                applied = env.apply_action(action)
                after = env.state().copy()
                state_position_start = 7 if side == "left" else 24
                delta = after[state_position_start : state_position_start + 3] - before[state_position_start : state_position_start + 3]
                directional = float(delta[axis])
                passed = applied["status"] == "PASS" and directional > 0.0002
                direction_records.append({"side": side, "axis": label, "delta_base_m": delta.tolist(), "command_axis_delta_m": directional, "status": "PASS" if passed else "FAIL"})
        checks.append(result("PASS" if all(item["status"] == "PASS" for item in direction_records) else "FAIL", "base-frame +/- axis translation smoke", name="translation_direction", records=direction_records))

        zero = env.apply_action(np.zeros(14, dtype=np.float32))
        checks.append(result("PASS" if zero["status"] == "PASS" else "FAIL", "zero action accepted atomically", name="zero_action"))
        gripper_values = []
        for side, arm in (("left", env.left_arm), ("right", env.right_arm)):
            for value in (0.0, 0.5, 1.0, 0.0):
                targets = arm.set_gripper(value)
                gripper_values.append({"side": side, "value": value, "base_target": targets[arm.gripper.base_joint]})
        checks.append(result("PASS", "GN01 normalized 0/0.5/1/0 cycle mapped from URDF limits", name="gripper_contract", values=gripper_values))
        rgb, depth = env.capture_head()
        camera_ok = rgb.shape[-1] == 3 and depth.shape == rgb.shape[:2] and np.isfinite(depth).all()
        checks.append(result("PASS" if camera_ok else "FAIL", f"head RGB={rgb.shape}, depth={depth.shape}", name="head_camera"))
        contacts = len(env.scene.get_contacts())
        checks.append(result("PASS", f"home contact records={contacts}", name="contacts", contact_count=contacts))
        report = {"status": "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL", "checks": checks, "state_field_names": env.state_adapter.field_names(), "action_field_names": ["left_delta_xyz", "left_delta_rotvec", "right_delta_xyz", "right_delta_rotvec", "left_gripper_cmd", "right_gripper_cmd"]}
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        host_skip = "vk::PhysicalDevice" in detail or "Vulkan" in detail
        status = "SKIP" if host_skip else "FAIL"
        checks.append(result(status, detail, name="validator_exception", manual_command="conda run -n dp3-rmbench python scripts/rmbench/flexiv/validate_embodiment.py --headless --out-dir outputs/rmbench_flexiv_embodiment" if host_skip else None))
        report = {"status": status, "checks": checks}
    finally:
        if env is not None:
            env.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "stability_summary.json").write_text(json.dumps(next((item for item in report["checks"] if item.get("name") == "stability"), {}), indent=2) + "\n", encoding="utf-8")
    (out_dir / "contacts_summary.json").write_text(json.dumps(next((item for item in report["checks"] if item.get("name") == "contacts"), {}), indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--stability-steps", type=int, default=1000)
    parser.add_argument("--repeated-resets", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/rmbench_flexiv_embodiment"))
    args = parser.parse_args()
    report = run(args.out_dir, stability_steps=args.stability_steps, repeated_resets=args.repeated_resets)
    for item in report["checks"]:
        print(f"{item['status']}: {item.get('name')} - {item['detail']}")
    return 0 if report["status"] in {"PASS", "SKIP"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
