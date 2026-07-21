#!/usr/bin/env python3
"""Validate the repository and runtime contracts for RMBench Stage 0."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RMBENCH = ROOT / "third_party" / "sim" / "RMBench"
sys.path.insert(0, str(RMBENCH))


def run_git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def check(name: str, fn, failures: list[str]) -> None:
    try:
        detail = fn()
        print(f"PASS: {name} - {detail}")
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        failures.append(f"{name}: {message}")
        print(f"FAIL: {name} - {message}")


def check_branch():
    branch = run_git("branch", "--show-current")
    if branch != "develop/RMBench":
        raise RuntimeError(f"expected develop/RMBench, got {branch!r}")
    return branch


def check_subtree():
    for path in (RMBENCH, RMBENCH / "LICENSE", RMBENCH / "README.md", RMBENCH / "README_VENDOR.md"):
        if not path.exists():
            raise FileNotFoundError(path)
    metadata = (RMBENCH / "README_VENDOR.md").read_text(encoding="utf-8")
    if "87e0498891073d483d330195c0f160709bd92ff5" not in metadata:
        raise RuntimeError("vendor metadata does not contain the pinned SHA")
    return "subtree, LICENSE, README, and vendor metadata present"


def check_environment():
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(f"expected Python 3.10, got {sys.version}")
    if str(Path(sys.executable)).endswith("/envs/dp3/bin/python"):
        raise RuntimeError("doctor must not run in dp3")
    return sys.executable


def check_imports():
    names = (
        "torch",
        "diffusion_policy_3d",
        "sapien",
        "gymnasium",
        "mplib",
        "pytorch3d.ops",
        "curobo.curobolib.geom",
    )
    files = {}
    for name in names:
        module = importlib.import_module(name)
        module_file = getattr(module, "__file__", None)
        if module_file is None and getattr(module, "__path__", None):
            module_file = next(iter(module.__path__))
        files[name] = str(module_file)
    dp3_file = Path(files["diffusion_policy_3d"]).resolve()
    expected = (ROOT / "3D-Diffusion-Policy" / "diffusion_policy_3d").resolve()
    shadow = (RMBENCH / "policy" / "DP3").resolve()
    if dp3_file != expected and expected not in dp3_file.parents:
        raise RuntimeError(f"DP3 import is outside main project: {dp3_file}")
    if shadow in dp3_file.parents:
        raise RuntimeError(f"RMBench DP3 shadow import: {dp3_file}")
    return files


def check_assets():
    expected = (
        RMBENCH / "assets" / "embodiments" / "aloha-agilex",
        RMBENCH / "assets" / "embodiments" / "franka-panda",
        RMBENCH / "assets" / "objects" / "005_button",
    )
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    return "required embodiment and object assets present"


def check_sim():
    smoke = ROOT / "scripts" / "rmbench" / "smoke_test.py"
    completed = subprocess.run(
        [sys.executable, str(smoke), "--level", "1"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = (completed.stdout + completed.stderr).strip().replace("\n", " ")
        raise RuntimeError(f"Level 1 exit {completed.returncode}: {detail[-1000:]}")
    return "SAPIEN scene and put_back_block import pass"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--check-assets", action="store_true")
    parser.add_argument("--check-sim", action="store_true")
    args = parser.parse_args()
    failures: list[str] = []
    check("branch", check_branch, failures)
    check("subtree and vendor metadata", check_subtree, failures)
    check("dp3-rmbench environment", check_environment, failures)
    check("imports and DP3 source path", check_imports, failures)
    if args.check_assets:
        check("assets", check_assets, failures)
    else:
        print("SKIP: assets - use --check-assets")
    if args.check_sim:
        check("SAPIEN scene and put_back_block import", check_sim, failures)
    else:
        print("SKIP: simulation - use --check-sim")
    if failures:
        print(f"doctor: {len(failures)} failure(s)")
        return 1 if args.strict or failures else 0
    print("doctor: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
