#!/usr/bin/env python3
"""Generate the Flexiv Stage 2 runtime bundle from official xacro sources.

The official submodule is never written.  xacro is evaluated against the
pinned checkout, raw official XML is retained for provenance, and a small
deterministic postprocessor creates SAPIEN-compatible URDFs with relative
mesh paths and positive inertials for the official zero-mass marker links.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from xml.dom import minidom

import yaml


PINNED_SHA = "92ef7865d76585e6e08d291bdfe652d32f7740f4"
UPSTREAM_URL = "https://github.com/flexivrobotics/flexiv_description.git"
UPSTREAM_BRANCH = "humble-v1"
BUNDLE_NAME = "flexiv-rizon4s-dual-gn01"


def git_root() -> Path:
    return Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    ).resolve()


def git_output(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def check_environment(root: Path) -> None:
    if git_output(root, "branch", "--show-current") != "develop/RMBench":
        raise RuntimeError("generation is restricted to the develop/RMBench branch")
    env_name = os.environ.get("CONDA_DEFAULT_ENV", Path(sys.prefix).name)
    if env_name != "dp3-rmbench":
        raise RuntimeError(
            f"expected dp3-rmbench, got {env_name!r}; do not install SAPIEN into dp3"
        )


def check_source(root: Path) -> tuple[Path, str]:
    source = root / "third_party" / "sim" / "flexiv_description"
    if not (source / ".git").exists() and not (source / "urdf").is_dir():
        raise FileNotFoundError(f"official submodule is missing: {source}")
    actual = git_output(source, "rev-parse", "HEAD")
    if actual != PINNED_SHA:
        raise RuntimeError(f"official submodule pin mismatch: expected {PINNED_SHA}, got {actual}")
    if not (source / "LICENSE").is_file():
        raise FileNotFoundError("official Apache-2.0 LICENSE is missing")
    return source, actual


def load_xacro(source: Path):
    try:
        import xacro
        import xacro.substitution_args as substitution_args
    except ImportError as exc:
        raise RuntimeError(
            "xacro is required in dp3-rmbench; install xacro in that environment"
        ) from exc

    # xacro 2.x asks ament_index for $(find flexiv_description).  ROS 2 is not
    # required for this bounded local route, so resolve only this official
    # package name to the checked-out submodule.
    original_find = substitution_args._eval_find

    def resolve_find(package: str) -> str:
        if package == "flexiv_description":
            return str(source)
        return original_find(package)

    substitution_args._eval_find = resolve_find
    return xacro


def render_xacro(source: Path, xacro_path: str, mappings: dict[str, str]) -> str:
    xacro = load_xacro(source)
    document = xacro.process_file(str(source / xacro_path), mappings=mappings)
    return document.toxml()


def _float_attr(element, name: str) -> float:
    return float(element.getAttribute(name))


def postprocess_urdf(
    xml_text: str,
    *,
    source: Path,
    artifact_dir: Path,
) -> tuple[str, dict[str, object]]:
    document = minidom.parseString(xml_text)
    report: dict[str, object] = {
        "mesh_uri_rewrite": "package://flexiv_description -> relative path",
        "zero_mass_marker_links": [],
        "zero_inertia_marker_links": [],
    }

    for inertial in document.getElementsByTagName("inertial"):
        mass_nodes = inertial.getElementsByTagName("mass")
        if not mass_nodes:
            continue
        mass = _float_attr(mass_nodes[0], "value")
        if mass <= 0.0:
            link = inertial.parentNode
            link_name = link.getAttribute("name") if link is not None else "unknown"
            mass_nodes[0].setAttribute("value", "1e-6")
            report["zero_mass_marker_links"].append(link_name)
            inertia_nodes = inertial.getElementsByTagName("inertia")
            if inertia_nodes:
                inertia = inertia_nodes[0]
                values = [_float_attr(inertia, key) for key in ("ixx", "iyy", "izz")]
                if max(values) <= 0.0:
                    for key in ("ixx", "iyy", "izz"):
                        inertia.setAttribute(key, "1e-9")
                    for key in ("ixy", "ixz", "iyz"):
                        inertia.setAttribute(key, "0")
                    report["zero_inertia_marker_links"].append(link_name)

    relative_mesh_root = Path(os.path.relpath(source, artifact_dir)).as_posix()
    for mesh in document.getElementsByTagName("mesh"):
        filename = mesh.getAttribute("filename")
        if filename.startswith("package://flexiv_description/"):
            mesh.setAttribute(
                "filename",
                relative_mesh_root + "/" + filename.split("package://flexiv_description/", 1)[1],
            )
        elif filename.startswith("/"):
            raise RuntimeError(f"official output unexpectedly contains an absolute mesh path: {filename}")
    return document.toprettyxml(indent="  "), report


def _prefix(side: str) -> str:
    return f"SIM-RIZON4S-{side.upper()}_"


def _extract_gripper(xml_text: str, prefix: str) -> dict[str, object]:
    document = minidom.parseString(xml_text)
    joints = {}
    for joint in document.getElementsByTagName("joint"):
        name = joint.getAttribute("name")
        if name.startswith(prefix):
            limit_nodes = joint.getElementsByTagName("limit")
            limit = {}
            if limit_nodes:
                limit = {
                    key: float(limit_nodes[0].getAttribute(key))
                    for key in ("lower", "upper", "velocity", "effort")
                    if limit_nodes[0].hasAttribute(key)
                }
            mimic_nodes = joint.getElementsByTagName("mimic")
            mimic = None
            if mimic_nodes:
                node = mimic_nodes[0]
                mimic = {
                    "joint": node.getAttribute("joint"),
                    "multiplier": float(node.getAttribute("multiplier")),
                    "offset": float(node.getAttribute("offset")),
                }
            joints[name] = {"type": joint.getAttribute("type"), "limit": limit, "mimic": mimic}
    base_name = prefix + "finger_width_joint"
    base = joints[base_name]
    return {
        "base_joint": base_name,
        "closed_position": base["limit"]["lower"],
        "open_position": base["limit"]["upper"],
        "mimic_joints": {
            name: value["mimic"]
            for name, value in joints.items()
            if value["mimic"] is not None
        },
        "joint_names": sorted(
            [name for name, value in joints.items() if value["mimic"] is not None]
        ),
    }


def _gripper_retract_config(side_manifest: dict[str, object]) -> list[float]:
    """Return GN01 active-joint values at normalized-open home (g=1)."""

    gripper = side_manifest["gripper"]
    targets = {gripper["base_joint"]: float(gripper["open_position"])}
    pending = dict(gripper["mimic_joints"])
    while pending:
        progressed = False
        for name, relation in list(pending.items()):
            if relation["joint"] in targets:
                targets[name] = (
                    float(relation["multiplier"]) * targets[relation["joint"]]
                    + float(relation["offset"])
                )
                del pending[name]
                progressed = True
        if not progressed:
            raise ValueError(f"unresolved GN01 mimic graph: {sorted(pending)}")
    arm_names = set(side_manifest["arm_joints"])
    return [float(targets[name]) for name in side_manifest["active_joints"] if name not in arm_names]


def embodiment_config(manifest: dict[str, object]) -> dict[str, object]:
    sides = manifest["sides"]
    left = sides["left"]
    right = sides["right"]
    arm_names = [left["arm_joints"], right["arm_joints"]]
    gripper_names = [
        {
            "base": left["gripper"]["base_joint"],
            "mimic": [
                [n, left["gripper"]["mimic_joints"][n]["multiplier"], left["gripper"]["mimic_joints"][n]["offset"]]
                for n in left["gripper"]["joint_names"]
            ],
        },
        {
            "base": right["gripper"]["base_joint"],
            "mimic": [
                [n, right["gripper"]["mimic_joints"][n]["multiplier"], right["gripper"]["mimic_joints"][n]["offset"]]
                for n in right["gripper"]["joint_names"]
            ],
        },
    ]
    return {
        "urdf_path": "left.urdf",
        "srdf_path": None,
        "joint_stiffness": 1200,
        "joint_damping": 80,
        "gripper_stiffness": 300,
        "gripper_damping": 20,
        "move_group": [left["tcp_link"], right["tcp_link"]],
        "ee_joints": [left["flange_joint"], right["flange_joint"]],
        "arm_joints_name": arm_names,
        "gripper_name": gripper_names,
        "gripper_bias": 0.0,
        "gripper_scale": [left["gripper"]["closed_position"], left["gripper"]["open_position"]],
        "homestate": [manifest["home_pose"]["left"], manifest["home_pose"]["right"]],
        "delta_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "global_trans_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "robot_pose": [[0, 0, 0.74, 1, 0, 0, 0], [0, 0, 0.74, 1, 0, 0, 0]],
        "planner": "mplib_RRT",
        "dual_arm": False,
        "static_camera_list": [],
    }


def _prefixed_gripper(gripper: dict[str, object], prefix: str) -> dict[str, object]:
    return {
        "base": prefix + gripper["base_joint"],
        "mimic": [
            [prefix + name, relation["multiplier"], relation["offset"]]
            for name, relation in gripper["mimic_joints"].items()
        ],
    }


def combined_embodiment_config(manifest: dict[str, object]) -> dict[str, object]:
    """Config for the retained official-style combined dual URDF artifact."""

    sides = manifest["sides"]
    left = sides["left"]
    right = sides["right"]
    left_prefix = "left_"
    right_prefix = "right_"
    return {
        "urdf_path": "runtime_dual.urdf",
        "srdf_path": None,
        "joint_stiffness": 1200,
        "joint_damping": 80,
        "gripper_stiffness": 300,
        "gripper_damping": 20,
        "move_group": [left_prefix + left["tcp_link"], right_prefix + right["tcp_link"]],
        "ee_joints": [left_prefix + left["flange_joint"], right_prefix + right["flange_joint"]],
        "arm_joints_name": [
            [left_prefix + name for name in left["arm_joints"]],
            [right_prefix + name for name in right["arm_joints"]],
        ],
        "gripper_name": [
            _prefixed_gripper(left["gripper"], left_prefix),
            _prefixed_gripper(right["gripper"], right_prefix),
        ],
        "gripper_bias": 0.0,
        "gripper_scale": [left["gripper"]["closed_position"], left["gripper"]["open_position"]],
        "homestate": [manifest["home_pose"]["left"], manifest["home_pose"]["right"]],
        "delta_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "global_trans_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "robot_pose": [[0, 0, 0.74, 1, 0, 0, 0], [0, 0, 0.74, 1, 0, 0, 0]],
        "planner": "mplib_RRT",
        "dual_arm": True,
        "static_camera_list": [],
    }


def curobo_config(
    side_manifest: dict[str, object],
    *,
    home_pose: list[float],
    urdf_path: str,
    name_prefix: str = "",
) -> dict[str, object]:
    active_joints = [name_prefix + name for name in side_manifest["active_joints"]]
    arm_joints = [name_prefix + name for name in side_manifest["arm_joints"]]
    collision_links = [name_prefix + name for name in side_manifest["collision_links"]]
    gripper_targets = _gripper_retract_config(side_manifest)
    return {
        "robot_cfg": {
            "kinematics": {
                "use_usd_kinematics": False,
                "urdf_path": urdf_path,
                "asset_root_path": None,
                "base_link": name_prefix + side_manifest["base_link"],
                "ee_link": name_prefix + side_manifest["tcp_link"],
                "collision_link_names": collision_links,
                "mesh_link_names": collision_links,
                "cspace": {
                    "joint_names": active_joints,
                    "retract_config": [*home_pose, *gripper_targets],
                    "null_space_weight": [1.0] * len(active_joints),
                    "cspace_distance_weight": [1.0] * len(active_joints),
                    "max_acceleration": 15.0,
                    "max_jerk": 500.0,
                },
            }
        },
        "planner": {"frame_bias": [0.0, 0.0, 0.0]},
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_bundle(root: Path, source: Path, temp_dir: Path) -> dict[str, object]:
    package_uri = "package://flexiv_description"
    bundle = root / "third_party" / "sim" / "RMBench" / "assets" / "embodiments" / BUNDLE_NAME
    left_raw = render_xacro(
        source,
        "urdf/rizon.urdf.xacro",
        {"rizon_type": "Rizon4s", "robot_sn": "SIM-RIZON4S-LEFT", "load_gripper": "true", "gripper_name": "Flexiv-GN01"},
    )
    right_raw = render_xacro(
        source,
        "urdf/rizon.urdf.xacro",
        {"rizon_type": "Rizon4s", "robot_sn": "SIM-RIZON4S-RIGHT", "load_gripper": "true", "gripper_name": "Flexiv-GN01"},
    )
    dual_raw = render_xacro(
        source,
        "urdf/rizon_dual.urdf.xacro",
        {
            "rizon_type_left": "Rizon4s",
            "rizon_type_right": "Rizon4s",
            "robot_sn_left": "SIM-RIZON4S-LEFT",
            "robot_sn_right": "SIM-RIZON4S-RIGHT",
            "load_gripper_left": "true",
            "load_gripper_right": "true",
            "gripper_name_left": "Flexiv-GN01",
            "gripper_name_right": "Flexiv-GN01",
        },
    )
    # The generated files are staged in /tmp, but their final location is the
    # bundle root. Relative mesh paths must therefore be calculated from the
    # final runtime location, not from the staging directory.
    left_runtime, left_report = postprocess_urdf(left_raw, source=source, artifact_dir=bundle)
    right_runtime, right_report = postprocess_urdf(right_raw, source=source, artifact_dir=bundle)
    dual_runtime, dual_report = postprocess_urdf(dual_raw, source=source, artifact_dir=bundle)

    left_prefix = _prefix("left")
    right_prefix = _prefix("right")
    home = {
        "left": [-0.17, -0.89, 0.13, 1.50, 0.85, 0.56, -0.71],
        "right": [0.27, -0.91, 0.06, 1.82, -1.04, 0.56, 1.17],
    }
    manifest: dict[str, object] = {
        "schema": "rmbench_flexiv_embodiment_v1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "upstream": {"url": UPSTREAM_URL, "branch": UPSTREAM_BRANCH, "commit": PINNED_SHA, "license": "Apache-2.0"},
        "generation": {
            "route": "local_xacro",
            "official_command": "scripts/create_urdf.sh --dual --rizon_type_left Rizon4s --rizon_type_right Rizon4s --load_gripper_left --load_gripper_right --gripper_name_left Flexiv-GN01 --gripper_name_right Flexiv-GN01",
            "docker_available": False,
            "package_uri_rewrite": package_uri,
            "postprocess": "positive_inertials_for_zero_mass_markers_and_relative_mesh_paths",
        },
        "artifacts": {"official_generated_dual": "official_generated_dual.urdf", "runtime_dual": "runtime_dual.urdf", "left": "left.urdf", "right": "right.urdf"},
        "home_pose": home,
        "sides": {},
    }
    for side, prefix, raw, report, runtime_name in (
        ("left", left_prefix, left_raw, left_report, "left.urdf"),
        ("right", right_prefix, right_raw, right_report, "right.urdf"),
    ):
        gripper = _extract_gripper(raw, prefix)
        document = minidom.parseString(raw)
        active_joints_raw = [
            joint.getAttribute("name")
            for joint in document.getElementsByTagName("joint")
            if joint.getAttribute("type") != "fixed"
        ]
        arm_joints = [prefix + f"joint{i}" for i in range(1, 8)]
        gripper_joints = [name for name in active_joints_raw if name not in arm_joints]
        # CuRobo's cspace and retract_config must use the same ordering.  The
        # official xacro emits GN01 joints before the seven arm joints, while
        # the runtime contract is arm-first followed by gripper joints.
        active_joints = [*arm_joints, *gripper_joints]
        collision_links = [
            link.getAttribute("name")
            for link in document.getElementsByTagName("link")
            if link.getElementsByTagName("collision")
        ]
        manifest["sides"][side] = {
            "prefix": prefix,
            "base_link": prefix + "base_link",
            "flange_link": prefix + "flange",
            "flange_joint": prefix + "link7_to_flange",
            "tcp_link": prefix + "grav_tcp",
            "arm_joints": arm_joints,
            "active_joints": active_joints,
            "collision_links": collision_links,
            "mesh_links": collision_links,
            "gripper": gripper,
            "runtime_urdf": runtime_name,
            "postprocess": report,
        }
    manifest["postprocess"] = {"dual": dual_report, "left": left_report, "right": right_report}

    write_text(temp_dir / "official_generated_dual.urdf", dual_raw)
    write_text(temp_dir / "official_generated_left.urdf", left_raw)
    write_text(temp_dir / "official_generated_right.urdf", right_raw)
    write_text(temp_dir / "runtime_dual.urdf", dual_runtime)
    write_text(temp_dir / "left.urdf", left_runtime)
    write_text(temp_dir / "right.urdf", right_runtime)
    write_text(temp_dir / "generation_manifest.json", json.dumps(manifest, indent=2) + "\n")
    write_text(temp_dir / "postprocess_report.json", json.dumps(manifest["postprocess"], indent=2) + "\n")

    config = embodiment_config(manifest)
    combined_config = combined_embodiment_config(manifest)
    for side in ("left", "right"):
        side_config = dict(config)
        side_config["urdf_path"] = f"../{side}.urdf"
        write_text(temp_dir / side / "config.yml", yaml.safe_dump(side_config, sort_keys=False))
        side_manifest = manifest["sides"][side]
        curobo = curobo_config(
            side_manifest,
            home_pose=manifest["home_pose"][side],
            urdf_path=f"../{side}.urdf",
        )
        write_text(temp_dir / side / "curobo.yml", yaml.safe_dump(curobo, sort_keys=False))
        combined_curobo = curobo_config(
            side_manifest,
            home_pose=manifest["home_pose"][side],
            urdf_path="runtime_dual.urdf",
            name_prefix=f"{side}_",
        )
        write_text(temp_dir / f"curobo_{side}.yml", yaml.safe_dump(combined_curobo, sort_keys=False))
    write_text(temp_dir / "config.yml", yaml.safe_dump(combined_config, sort_keys=False))
    write_text(temp_dir / "curobo.yml", yaml.safe_dump(curobo_config(manifest["sides"]["left"], home_pose=manifest["home_pose"]["left"], urdf_path="runtime_dual.urdf", name_prefix="left_"), sort_keys=False))
    write_text(temp_dir / "README_RUNTIME.md", "Generated by scripts/rmbench/flexiv/generate_embodiment.py. Rebuild instead of editing generated files.\n")
    return manifest


def replace_bundle(bundle: Path, temp_dir: Path, *, force: bool) -> None:
    if bundle.exists() and not force:
        manifest = bundle / "generation_manifest.json"
        if manifest.is_file() and json.loads(manifest.read_text(encoding="utf-8")).get("upstream", {}).get("commit") == PINNED_SHA:
            print(f"Existing correct runtime bundle kept: {bundle}")
            return
        raise FileExistsError(f"{bundle} exists; pass --force to replace it")
    staging = bundle.with_name(f".{bundle.name}.staging-{uuid.uuid4().hex}")
    shutil.copytree(temp_dir, staging)
    old = None
    if bundle.exists():
        old = bundle.with_name(f".{bundle.name}.old-{uuid.uuid4().hex}")
        os.replace(bundle, old)
    os.replace(staging, bundle)
    if old is not None:
        shutil.rmtree(old)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = git_root()
    check_environment(root)
    source, commit = check_source(root)
    bundle = root / "third_party" / "sim" / "RMBench" / "assets" / "embodiments" / BUNDLE_NAME
    print(f"source={source}")
    print(f"commit={commit}")
    print("route=local_xacro (Docker unavailable)")
    print(f"bundle={bundle}")
    if args.dry_run:
        print("official dual command: scripts/create_urdf.sh --dual --rizon_type_left Rizon4s --rizon_type_right Rizon4s --load_gripper_left --load_gripper_right --gripper_name_left Flexiv-GN01 --gripper_name_right Flexiv-GN01")
        return 0
    with tempfile.TemporaryDirectory(prefix="flexiv-rmbench-") as temp:
        temp_dir = Path(temp)
        manifest = build_bundle(root, source, temp_dir)
        replace_bundle(bundle, temp_dir, force=args.force)
        print(json.dumps({"status": "generated", "bundle": str(bundle), "schema": manifest["schema"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
