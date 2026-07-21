#!/usr/bin/env python3
"""Apply the two RMBench upstream compatibility patches safely and once."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path


def _module_path(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"Cannot locate installed module {name!r}")
    return Path(spec.origin).resolve()


def _backup(path: Path) -> Path:
    backup = path.with_name(path.name + ".rmbench-stage0.orig")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def _replace_once(path: Path, old: str, new: str, label: str, dry_run: bool) -> dict:
    text = path.read_text(encoding="utf-8")
    if old in text:
        if not dry_run:
            backup = _backup(path)
            path.write_text(text.replace(old, new, 1), encoding="utf-8")
        else:
            backup = path.with_name(path.name + ".rmbench-stage0.orig")
        return {"label": label, "status": "would_patch" if dry_run else "patched", "file": str(path), "backup": str(backup)}
    if new in text:
        return {"label": label, "status": "already_patched", "file": str(path)}
    raise RuntimeError(f"Expected source text for {label} was not found in {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sapien_root = _module_path("sapien").parent
    mplib_root = _module_path("mplib").parent
    urdf_loader = sapien_root / "wrapper" / "urdf_loader.py"
    planner = mplib_root / "planner.py"
    if not urdf_loader.is_file():
        raise RuntimeError(f"SAPIEN URDF loader not found: {urdf_loader}")
    if not planner.is_file():
        raise RuntimeError(f"MPLib planner not found: {planner}")

    changes = []
    changes.append(_replace_once(urdf_loader, 'with open(urdf_file, "r") as f:', 'with open(urdf_file, "r", encoding="utf-8") as f:', "sapien urdf encoding", args.dry_run))
    changes.append(_replace_once(urdf_loader, 'urdf_file[:-4] + "srdf"', 'urdf_file[:-4] + ".srdf"', "sapien srdf suffix", args.dry_run))
    changes.append(_replace_once(urdf_loader, 'with open(srdf_file, "r") as f:', 'with open(srdf_file, "r", encoding="utf-8") as f:', "sapien srdf encoding", args.dry_run))
    changes.append(_replace_once(planner, "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:", "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:", "mplib collision compatibility", args.dry_run))
    print(json.dumps({"status": "dry_run" if args.dry_run else "complete", "changes": changes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
