#!/usr/bin/env python3
"""Read-only baseline checks for the existing dp3 environment."""

from __future__ import annotations

import importlib
import importlib.util
import argparse
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "3D-Diffusion-Policy"
SNAPSHOT = ROOT / "environments" / "snapshots" / "dp3_before_rmbench"
sys.path.insert(0, str(PROJECT))


def check(name: str, fn) -> dict:
    try:
        value = fn()
        return {"name": name, "status": "PASS", "detail": value}
    except Exception as exc:
        return {"name": name, "status": "BASELINE_EXISTING_FAILURE", "detail": f"{type(exc).__name__}: {exc}"}


def flexiv_import():
    interface = ROOT / "third_party" / "real" / "dual_flexiv_rizon4s" / "interface"
    if not (interface / "config_flexiv.py").is_file():
        raise FileNotFoundError(interface)
    package_name = "_rmbench_baseline_flexiv"
    package = types.ModuleType(package_name)
    package.__path__ = [str(interface)]
    package.__package__ = package_name
    sys.modules[package_name] = package
    importlib.import_module(f"{package_name}.config_flexiv")
    return str(interface)


def _sapien2_scene():
    sapien = importlib.import_module("sapien.core")
    engine = sapien.Engine()
    scene = engine.create_scene()
    scene.step()
    return getattr(sapien, "__version__", "imported")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-out",
        type=Path,
        help="optional output path; relative paths are resolved from the repository root",
    )
    args = parser.parse_args()
    checks = [
        ("diffusion_policy_3d", lambda: str(importlib.import_module("diffusion_policy_3d").__file__)),
        ("SimpleDP3", lambda: str(importlib.import_module("diffusion_policy_3d.policy.simple_dp3").__file__)),
        ("DP3", lambda: str(importlib.import_module("diffusion_policy_3d.policy.dp3").__file__)),
        ("Flexiv", flexiv_import),
        ("MetaWorld", lambda: str(importlib.import_module("diffusion_policy_3d.dataset.metaworld_dataset").__file__)),
        ("Adroit", lambda: str(importlib.import_module("diffusion_policy_3d.dataset.adroit_dataset").__file__)),
        ("DexArt", lambda: str(importlib.import_module("diffusion_policy_3d.dataset.dexart_dataset").__file__)),
        ("SAPIEN 2 scene", _sapien2_scene),
    ]
    results = [check(name, fn) for name, fn in checks]
    output = args.json_out or (SNAPSHOT / "sim-import-baseline.json")
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8")
    for item in results:
        print(f"{item['status']}: {item['name']} - {item['detail']}")
    print(f"JSON: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
