import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "3D-Diffusion-Policy"))
from diffusion_policy_3d.sim.flexiv.gripper_adapter import GripperMapping

BUNDLE = ROOT / "third_party/sim/RMBench/assets/embodiments/flexiv-rizon4s-dual-gn01"


def test_gn01_mapping_is_urdf_derived_and_symmetric():
    manifest = json.loads((BUNDLE / "generation_manifest.json").read_text())
    left = GripperMapping.from_manifest(manifest["sides"]["left"]["gripper"])
    right = GripperMapping.from_manifest(manifest["sides"]["right"]["gripper"])
    for mapping in (left, right):
        assert mapping.normalized_to_base(0.0) == 0.0
        assert mapping.normalized_to_base(0.5) == 0.05
        assert mapping.normalized_to_base(1.0) == 0.1
        assert mapping.base_to_normalized(0.1) == 1.0
        assert len(mapping.normalized_to_joint_targets(0.5)) == 7
    left_suffixes = {name.replace("SIM-RIZON4S-LEFT", "SIM-RIZON4S") for name in left.mimic_joints}
    right_suffixes = {name.replace("SIM-RIZON4S-RIGHT", "SIM-RIZON4S") for name in right.mimic_joints}
    assert left_suffixes == right_suffixes
