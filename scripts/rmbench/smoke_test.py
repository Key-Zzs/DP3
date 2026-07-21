#!/usr/bin/env python3
"""Run the bounded RMBench Stage 0 smoke levels."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RMBENCH = ROOT / "third_party" / "sim" / "RMBench"
PROJECT = ROOT / "3D-Diffusion-Policy"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(RMBENCH))


def result(status: str, detail: str) -> dict:
    return {"status": status, "detail": detail}


def _module_location(module) -> str:
    module_file = getattr(module, "__file__", None)
    if module_file is None and getattr(module, "__path__", None):
        module_file = next(iter(module.__path__))
    return str(module_file)


def _rmbench_import(module_name: str):
    old_cwd = Path.cwd()
    try:
        os.chdir(RMBENCH)
        return importlib.import_module(module_name)
    finally:
        os.chdir(old_cwd)


def level0() -> list[dict]:
    checks = [
        ("DP3", "diffusion_policy_3d"),
        ("PyTorch", "torch"),
        ("RMBench envs", "envs"),
        ("SAPIEN 3", "sapien"),
        ("Gymnasium", "gymnasium"),
        ("MPLib", "mplib"),
        ("PyTorch3D ops", "pytorch3d.ops"),
        ("CuRobo extensions", "curobo.curobolib.geom"),
    ]
    output = []
    for label, module_name in checks:
        try:
            module = _rmbench_import(module_name) if module_name == "envs" else importlib.import_module(module_name)
            output.append({"name": label, **result("PASS", _module_location(module))})
        except Exception as exc:
            output.append({"name": label, **result("FAIL", f"{type(exc).__name__}: {exc}")})
    return output


def _scene_smoke() -> str:
    import sapien

    engine = sapien.Engine()
    scene = engine.create_scene()
    scene.add_ground(0)
    scene.step()
    return f"SAPIEN {getattr(sapien, '__version__', 'unknown')} scene created"


def level1() -> list[dict]:
    output = []
    try:
        output.append({"name": "minimal SAPIEN 3 scene", **result("PASS", _scene_smoke())})
    except Exception as exc:
        output.append({"name": "minimal SAPIEN 3 scene", **result("FAIL", f"{type(exc).__name__}: {exc}")})
    try:
        module = _rmbench_import("envs.put_back_block")
        cls = getattr(module, "put_back_block")
        output.append({"name": "put_back_block import", **result("PASS", str(cls))})
    except Exception as exc:
        output.append({"name": "put_back_block import", **result("FAIL", f"{type(exc).__name__}: {exc}")})
    return output


def _asset_paths() -> dict[str, Path]:
    return {
        "aloha-agilex": RMBENCH / "assets" / "embodiments" / "aloha-agilex",
        "005_button": RMBENCH / "assets" / "objects" / "005_button",
    }


def level2() -> list[dict]:
    missing = [str(path) for path in _asset_paths().values() if not path.exists()]
    if missing:
        return [{"name": "put_back_block initialization", **result("SKIP", "missing assets: " + ", ".join(missing))}]

    task = None
    with tempfile.TemporaryDirectory(prefix="rmbench-stage0-") as save_dir:
        try:
            import yaml

            old_cwd = Path.cwd()
            os.chdir(RMBENCH)
            from envs.put_back_block import put_back_block

            with (RMBENCH / "task_config" / "demo_clean.yml").open(encoding="utf-8") as handle:
                args = yaml.safe_load(handle)
            robot_dir = _asset_paths()["aloha-agilex"]
            with (robot_dir / "config.yml").open(encoding="utf-8") as handle:
                robot_config = yaml.safe_load(handle)
            args.update(
                {
                    "task_name": "put_back_block",
                    "task_config": "demo_clean",
                    "save_path": save_dir,
                    "left_robot_file": str(robot_dir),
                    "right_robot_file": str(robot_dir),
                    "left_embodiment_config": robot_config,
                    "right_embodiment_config": robot_config,
                    "embodiment_name": "aloha-agilex",
                    "dual_arm_embodied": True,
                    "need_plan": False,
                    "save_data": False,
                    "eval_mode": False,
                    "render_freq": 0,
                }
            )
            task = put_back_block()
            task.setup_demo(now_ep_num=0, seed=0, **args)
            observation = task.get_obs()
            if not isinstance(observation, dict):
                raise TypeError(f"observation is {type(observation).__name__}, expected dict")
            required = {"observation", "joint_action", "endpose"}
            missing_keys = sorted(required - set(observation))
            if missing_keys:
                raise KeyError(f"observation missing keys: {missing_keys}")
            return [{"name": "put_back_block initialization", **result("PASS", f"observation keys={sorted(observation)}")}]
        except Exception as exc:
            return [{"name": "put_back_block initialization", **result("FAIL", f"{type(exc).__name__}: {exc}")}]
        finally:
            os.chdir(old_cwd) if "old_cwd" in locals() else None
            if task is not None:
                try:
                    task.close_env()
                except Exception:
                    pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    levels = {0: level0, 1: level1, 2: level2}
    checks = levels[args.level]()
    report = {"level": args.level, "results": checks}
    if args.json_out:
        output = args.json_out if args.json_out.is_absolute() else ROOT / args.json_out
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for item in checks:
        print(f"{item['status']}: {item['name']} - {item['detail']}")
    return 1 if any(item["status"] == "FAIL" for item in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
