import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "scripts" / "rmbench" / "smoke_test.py"


def test_smoke_exposes_bounded_levels_and_json_output():
    tree = ast.parse(SMOKE.read_text(encoding="utf-8"))
    source = SMOKE.read_text(encoding="utf-8")
    assert "choices=(0, 1, 2)" in source
    assert "--json-out" in source
    assert "def level0" in source
    assert "def level1" in source
    assert "def level2" in source
    assert any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body)


def test_level2_has_explicit_asset_skip_and_observation_contract():
    source = SMOKE.read_text(encoding="utf-8")
    assert 'result("SKIP", "missing assets:' in source
    assert 'required = {"observation", "joint_action", "endpose"}' in source
