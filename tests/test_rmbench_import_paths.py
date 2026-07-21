import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "rmbench" / "smoke_test.py"


def test_smoke_script_prefers_main_dp3_project_and_vendor_root():
    source = SCRIPT.read_text(encoding="utf-8")
    ast.parse(source)
    assert "sys.path.insert(0, str(PROJECT))" in source
    assert "sys.path.insert(0, str(RMBENCH))" in source


def test_main_dp3_package_exists_outside_upstream_policy_reference():
    assert (ROOT / "3D-Diffusion-Policy" / "diffusion_policy_3d").is_dir()
    assert (ROOT / "third_party" / "sim" / "RMBench" / "policy" / "DP3").is_dir()
