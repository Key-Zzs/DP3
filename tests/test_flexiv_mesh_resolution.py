import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/rmbench/flexiv"))
from inspect_urdf import audit_urdf

BUNDLE = ROOT / "third_party/sim/RMBench/assets/embodiments/flexiv-rizon4s-dual-gn01"


def test_all_runtime_meshes_resolve_without_package_or_absolute_paths():
    manifest = json.loads((BUNDLE / "generation_manifest.json").read_text())
    for side in ("left", "right"):
        report = audit_urdf(BUNDLE / f"{side}.urdf", manifest, side)
        assert report["status"] == "PASS", report["issues"]
        assert all(item["exists"] for item in report["mesh_files"])
        assert all(not item["filename"].startswith("package://") for item in report["mesh_files"])
        assert all(not Path(item["filename"]).is_absolute() for item in report["mesh_files"])
