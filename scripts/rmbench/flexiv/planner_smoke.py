#!/usr/bin/env python3
"""Smoke-test the generated CuRobo configs and native MPlib URDF loading."""

from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import sapien.core as sapien
import yaml

ROOT = Path(__file__).resolve().parents[3]
RMBENCH = ROOT / "third_party" / "sim" / "RMBench"
sys.path.insert(0, str(ROOT / "3D-Diffusion-Policy"))
sys.path.insert(0, str(RMBENCH))
os.chdir(RMBENCH)

from envs.flexiv_embodiment_smoke import BUNDLE_ROOT


def run(out_dir: Path) -> dict:
    checks: list[dict] = []
    bundle = BUNDLE_ROOT
    for side in ("left", "right"):
        config_path = bundle / side / "curobo.yml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        kinematics = config["robot_cfg"]["kinematics"]
        cspace = kinematics["cspace"]
        relative_urdf = (config_path.parent / kinematics["urdf_path"]).resolve()
        structural = (
            relative_urdf.is_file()
            and len(cspace["joint_names"]) == len(cspace["retract_config"]) == 14
            and cspace["joint_names"][:7] == [f"SIM-RIZON4S-{side.upper()}_joint{i}" for i in range(1, 8)]
            and not any(Path(str(value)).is_absolute() for value in (kinematics.get("urdf_path"),))
        )
        checks.append({"name": f"{side}_curobo_structure", "status": "PASS" if structural else "FAIL", "detail": str(config_path)})
        try:
            from curobo.wrap.reacher.motion_gen import MotionGenConfig

            MotionGenConfig.load_from_robot_config(
                str(config_path),
                world_config={"cuboid": {"table": {"dims": [1.9, 1.4, 0.08], "pose": [0, 0.05, 0.70, 1, 0, 0, 0]}}},
                interpolation_dt=1 / 250,
                num_trajopt_seeds=1,
            )
            checks.append({"name": f"{side}_curobo_load", "status": "PASS", "detail": "MotionGenConfig.load_from_robot_config"})
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            status = "SKIP" if "CUDA GPUs are available" in detail else "FAIL"
            checks.append({"name": f"{side}_curobo_load", "status": status, "detail": detail})

    manifest = json.loads((bundle / "generation_manifest.json").read_text(encoding="utf-8"))
    for side in ("left", "right"):
        try:
            import mplib

            urdf_root = ET.parse(bundle / f"{side}.urdf").getroot()
            link_names = [node.get("name", "") for node in urdf_root.findall("link")]
            joint_names = [
                node.get("name", "")
                for node in urdf_root.findall("joint")
                if node.get("type") != "fixed"
            ]
            side_manifest = manifest["sides"][side]
            planner = mplib.Planner(
                urdf=str(bundle / f"{side}.urdf"),
                srdf=None,
                move_group=side_manifest["tcp_link"],
                user_link_names=link_names,
                user_joint_names=joint_names,
                use_convex=False,
            )
            planner.set_base_pose(sapien.Pose([0.0, 0.0, 0.0]))
            checks.append({"name": f"{side}_mplib_load", "status": "PASS", "detail": f"{len(planner.joint_limits)} joint limits"})
        except Exception as exc:
            checks.append({"name": f"{side}_mplib_load", "status": "FAIL", "detail": f"{type(exc).__name__}: {exc}"})

    has_fail = any(item["status"] == "FAIL" for item in checks)
    has_skip = any(item["status"] == "SKIP" for item in checks)
    report = {"status": "FAIL" if has_fail else "PASS_WITH_SKIPS" if has_skip else "PASS", "checks": checks}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "planner_smoke.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/rmbench_flexiv_embodiment"))
    args = parser.parse_args()
    report = run(args.out_dir)
    for item in report["checks"]:
        print(f"{item['status']}: {item['name']} - {item['detail']}")
    return 0 if report["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
