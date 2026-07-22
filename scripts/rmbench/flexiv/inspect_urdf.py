#!/usr/bin/env python3
"""Strict structural and mesh audit for a generated Flexiv URDF."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def number(value: str | None, *, label: str) -> float | None:
    if value is None or value == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is NaN/Inf")
    return result


def audit_urdf(urdf_path: Path, manifest: dict | None = None, side: str | None = None, *, expected_arm_count: int = 7, expected_gripper_count: int = 1) -> dict:
    root = ET.parse(urdf_path).getroot()
    links = root.findall("link")
    joints = root.findall("joint")
    link_names = [link.get("name", "") for link in links]
    joint_names = [joint.get("name", "") for joint in joints]
    issues: list[str] = []
    if len(link_names) != len(set(link_names)):
        issues.append("duplicate link names")
    if len(joint_names) != len(set(joint_names)):
        issues.append("duplicate joint names")

    parent_by_child: dict[str, str] = {}
    joint_records = []
    mesh_records = []
    for joint in joints:
        name = joint.get("name", "")
        parent = joint.find("parent")
        child = joint.find("child")
        parent_name = parent.get("link", "") if parent is not None else ""
        child_name = child.get("link", "") if child is not None else ""
        if child_name in parent_by_child:
            issues.append(f"multiple parents for link {child_name}")
        parent_by_child[child_name] = name
        axis_node = joint.find("axis")
        limit_node = joint.find("limit")
        mimic_node = joint.find("mimic")
        limits = {}
        if limit_node is not None:
            for key in ("lower", "upper", "velocity", "effort"):
                if limit_node.get(key) is not None:
                    limits[key] = number(limit_node.get(key), label=f"{name}.{key}")
            if "lower" in limits and "upper" in limits and limits["lower"] > limits["upper"]:
                issues.append(f"invalid joint limits for {name}")
        joint_records.append(
            {
                "name": name,
                "type": joint.get("type"),
                "parent": parent_name,
                "child": child_name,
                "axis": (axis_node.get("xyz") if axis_node is not None else None),
                "limits": limits,
                "mimic": None
                if mimic_node is None
                else {
                    "joint": mimic_node.get("joint"),
                    "multiplier": number(mimic_node.get("multiplier", "1"), label=f"{name}.multiplier"),
                    "offset": number(mimic_node.get("offset", "0"), label=f"{name}.offset"),
                },
            }
        )

    for link in links:
        name = link.get("name", "")
        inertial = link.find("inertial")
        if inertial is not None:
            mass_node = inertial.find("mass")
            inertia_node = inertial.find("inertia")
            mass = number(mass_node.get("value") if mass_node is not None else None, label=f"{name}.mass")
            if mass is None or mass <= 0:
                issues.append(f"non-positive mass for {name}")
            if inertia_node is None:
                issues.append(f"missing inertia for {name}")
            else:
                vals = [number(inertia_node.get(key), label=f"{name}.{key}") for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
                matrix = np.array([[vals[0], vals[3], vals[4]], [vals[3], vals[1], vals[5]], [vals[4], vals[5], vals[2]]], dtype=float)
                if not np.all(np.linalg.eigvalsh(matrix) > 0):
                    issues.append(f"non-positive-definite inertia for {name}")
        link_meshes = []
        for mesh in link.findall(".//mesh"):
            filename = mesh.get("filename", "")
            resolved = None
            if filename.startswith("package://"):
                issues.append(f"package URI remains for {name}: {filename}")
            elif Path(filename).is_absolute():
                issues.append(f"absolute mesh path for {name}: {filename}")
            else:
                resolved = (urdf_path.parent / filename).resolve()
                if not resolved.is_file():
                    issues.append(f"missing mesh for {name}: {filename}")
            record = {"filename": filename, "resolved": str(resolved) if resolved else None, "exists": bool(resolved and resolved.is_file())}
            link_meshes.append(record)
            mesh_records.append({"link": name, **record})

    root_links = [name for name in link_names if name not in parent_by_child]
    arm_joints = [name for name in joint_names if re.search(r"joint[1-7]$", name)]
    gripper_base = [name for name in joint_names if name.endswith("finger_width_joint")]
    mimic_joints = [record["name"] for record in joint_records if record["mimic"] is not None]
    if len(arm_joints) != expected_arm_count:
        issues.append(f"expected exactly {expected_arm_count} arm joints, found {len(arm_joints)}")
    if len(gripper_base) != expected_gripper_count:
        issues.append(f"expected {expected_gripper_count} GN01 finger_width_joint(s), found {len(gripper_base)}")
    if not mimic_joints:
        issues.append("GN01 mimic joints are missing")

    expected = None
    if manifest is not None and side is not None:
        expected = manifest.get("sides", {}).get(side)
        if expected:
            if arm_joints != expected["arm_joints"]:
                issues.append(f"arm joint order differs from manifest for {side}")
            if expected["tcp_link"] not in link_names:
                issues.append(f"manifest TCP link missing: {expected['tcp_link']}")
            if expected["flange_link"] not in link_names:
                issues.append(f"manifest flange link missing: {expected['flange_link']}")

    return {
        "status": "PASS" if not issues else "FAIL",
        "urdf": str(urdf_path),
        "root_link": root_links[0] if len(root_links) == 1 else root_links,
        "links": [{"name": link.get("name"), "has_inertial": link.find("inertial") is not None, "mesh_count": len(link.findall(".//mesh"))} for link in links],
        "joints": joint_records,
        "arm_joints": arm_joints,
        "gripper_base_joints": gripper_base,
        "mimic_joints": mimic_joints,
        "mesh_files": mesh_records,
        "tcp_links": [name for name in link_names if name.endswith("grav_tcp")],
        "flange_links": [name for name in link_names if name.endswith("flange")],
        "issues": issues,
    }


def tree_text(report: dict) -> str:
    by_parent: dict[str, list[dict]] = {}
    for joint in report["joints"]:
        by_parent.setdefault(joint["parent"], []).append(joint)
    lines: list[str] = []

    def walk(link: str, depth: int) -> None:
        lines.append("  " * depth + link)
        for joint in by_parent.get(link, []):
            lines.append("  " * (depth + 1) + f"--[{joint['type']}] {joint['name']}")
            walk(joint["child"], depth + 2)

    root = report["root_link"]
    if isinstance(root, list):
        for item in root:
            walk(item, 0)
    else:
        walk(root, 0)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--side", choices=("left", "right"))
    parser.add_argument("--dual", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/rmbench_flexiv_embodiment"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else None
    report = audit_urdf(args.urdf.resolve(), manifest, args.side, expected_arm_count=14 if args.dual else 7, expected_gripper_count=2 if args.dual else 1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "urdf_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "urdf_tree.txt").write_text(tree_text(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "arm_joints": report["arm_joints"], "gripper_base_joints": report["gripper_base_joints"], "issues": report["issues"]}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
