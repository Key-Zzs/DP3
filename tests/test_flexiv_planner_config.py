from pathlib import Path
import json

import yaml


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "third_party/sim/RMBench/assets/embodiments/flexiv-rizon4s-dual-gn01"


def test_generated_planner_cspace_matches_arm_then_gn01_order():
    manifest = json.loads((BUNDLE / "generation_manifest.json").read_text())
    for side in ("left", "right"):
        config = yaml.safe_load((BUNDLE / side / "curobo.yml").read_text())
        cspace = config["robot_cfg"]["kinematics"]["cspace"]
        expected = manifest["sides"][side]["active_joints"]
        assert cspace["joint_names"] == expected
        assert len(cspace["retract_config"]) == len(expected) == 14
        assert cspace["joint_names"][:7] == manifest["sides"][side]["arm_joints"]
        assert not Path(config["robot_cfg"]["kinematics"]["urdf_path"]).is_absolute()


def test_combined_bundle_config_uses_prefixed_dual_runtime_artifact():
    config = yaml.safe_load((BUNDLE / "config.yml").read_text())
    assert config["urdf_path"] == "runtime_dual.urdf"
    assert config["dual_arm"] is True
    assert config["arm_joints_name"][0][0].startswith("left_")
    assert config["arm_joints_name"][1][0].startswith("right_")
    for side in ("left", "right"):
        curobo = yaml.safe_load((BUNDLE / f"curobo_{side}.yml").read_text())
        assert curobo["robot_cfg"]["kinematics"]["urdf_path"] == "runtime_dual.urdf"
