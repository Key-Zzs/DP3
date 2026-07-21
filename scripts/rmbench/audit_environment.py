#!/usr/bin/env python3
"""Report the active dp3-rmbench environment and RMBench Stage 0 contracts."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RMBENCH = ROOT / "third_party" / "sim" / "RMBench"
PROJECT = ROOT / "3D-Diffusion-Policy"
HF_REPO = "TianxingChen/RMBench"
HF_REVISION = os.environ.get(
    "RMBENCH_HF_REVISION", "d899d72b53270a89f71d216c08ecbd4d9a7004fd"
)


def _run(*args: str) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _module(name: str) -> dict:
    try:
        spec = importlib.util.find_spec(name)
        if spec is None:
            return {"status": "FAIL", "error": "module not found"}
        module = importlib.import_module(name)
        module_file = getattr(module, "__file__", None)
        if module_file is None and getattr(module, "__path__", None):
            module_file = next(iter(module.__path__))
        return {
            "status": "PASS",
            "file": str(module_file),
            "version": getattr(module, "__version__", None),
        }
    except Exception as exc:
        return {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}


def _assets() -> dict:
    expected = {
        "aloha-agilex": RMBENCH / "assets" / "embodiments" / "aloha-agilex",
        "franka-panda": RMBENCH / "assets" / "embodiments" / "franka-panda",
        "005_button": RMBENCH / "assets" / "objects" / "005_button",
    }
    return {name: {"exists": path.exists(), "path": str(path)} for name, path in expected.items()}


def collect() -> dict:
    modules = [
        "torch",
        "numpy",
        "scipy",
        "sapien",
        "gymnasium",
        "mplib",
        "pytorch3d.ops",
        "curobo.curobolib.geom",
        "warp",
    ]
    versions = {name: _module(name) for name in modules}
    torch_cuda = {}
    try:
        import torch

        torch_cuda = {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except Exception as exc:
        torch_cuda = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}

    dp3_spec = importlib.util.find_spec("diffusion_policy_3d")
    dp3_module = importlib.import_module("diffusion_policy_3d") if dp3_spec else None
    dp3_file = getattr(dp3_module, "__file__", None) if dp3_module else None
    if dp3_file is None and dp3_module is not None and getattr(dp3_module, "__path__", None):
        dp3_file = next(iter(dp3_module.__path__))
    return {
        "repository": str(ROOT),
        "branch": _run("git", "branch", "--show-current"),
        "commit": _run("git", "rev-parse", "HEAD"),
        "python": {"executable": sys.executable, "version": sys.version, "sys_path": sys.path},
        "conda": {"default_env": os.environ.get("CONDA_DEFAULT_ENV"), "prefix": os.environ.get("CONDA_PREFIX")},
        "torch_cuda": torch_cuda,
        "packages": versions,
        "dp3_import": {
            "file": str(dp3_file) if dp3_file else None,
            "spec_origin": str(dp3_spec.origin) if dp3_spec else None,
            "expected_root": str(PROJECT),
            "shadow_path": str(RMBENCH / "policy" / "DP3"),
        },
        "assets": _assets(),
        "asset_source": {"repo": HF_REPO, "revision": HF_REVISION},
        "subtree": {
            "path": str(RMBENCH),
            "license": (RMBENCH / "LICENSE").is_file(),
            "vendor_metadata": (RMBENCH / "README_VENDOR.md").is_file(),
        },
        "curobo": {
            "path": str(RMBENCH / "envs" / "curobo"),
            "git_head": _run("git", "-C", str(RMBENCH / "envs" / "curobo"), "rev-parse", "HEAD"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = collect()
    if args.json_out:
        output = args.json_out if args.json_out.is_absolute() else ROOT / args.json_out
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print("RMBench Stage 0 environment audit")
    print(f"repository: {report['repository']}")
    print(f"branch: {report['branch']}")
    print(f"python: {report['python']['executable']}")
    print(f"dp3 import: {report['dp3_import']['file']}")
    print(f"torch/cuda: {report['torch_cuda']}")
    for name, value in report["packages"].items():
        print(f"{name}: {value}")
    print(f"assets: {report['assets']}")
    print(f"asset source: {report['asset_source']}")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
