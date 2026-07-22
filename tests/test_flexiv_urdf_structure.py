import json
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "third_party/sim/RMBench/assets/embodiments/flexiv-rizon4s-dual-gn01"


def _load(side):
    manifest = json.loads((BUNDLE / "generation_manifest.json").read_text())
    root = ET.parse(BUNDLE / f"{side}.urdf").getroot()
    return manifest, root


def test_each_runtime_urdf_has_seven_unique_arm_joints_and_gn01():
    for side in ("left", "right"):
        manifest, root = _load(side)
        names = [node.get("name") for node in root.findall("joint")]
        expected = manifest["sides"][side]
        assert [name for name in names if name.endswith(tuple(f"joint{i}" for i in range(1, 8)))] == expected["arm_joints"]
        assert len([name for name in names if name.endswith("finger_width_joint")]) == 1
        assert expected["tcp_link"] in [node.get("name") for node in root.findall("link")]
        assert expected["flange_link"] in [node.get("name") for node in root.findall("link")]


def test_official_combined_runtime_has_two_prefixed_arms_and_grippers():
    root = ET.parse(BUNDLE / "runtime_dual.urdf").getroot()
    names = [node.get("name") for node in root.findall("joint")]
    assert sum(name.endswith(tuple(f"joint{i}" for i in range(1, 8))) for name in names) == 14
    assert sum(name.endswith("finger_width_joint") for name in names) == 2
    assert len(names) == len(set(names))
