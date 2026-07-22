from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_rmbench_registers_dual_bundle_and_two_single_arm_paths():
    config = yaml.safe_load((ROOT / "third_party/sim/RMBench/task_config/_embodiment_config.yml").read_text())
    assert config["flexiv-rizon4s-dual-gn01"]["file_path"].endswith("flexiv-rizon4s-dual-gn01/")
    assert config["flexiv-rizon4s-dual-gn01-left"]["file_path"].endswith("/left/")
    assert config["flexiv-rizon4s-dual-gn01-right"]["file_path"].endswith("/right/")
    task = yaml.safe_load((ROOT / "third_party/sim/RMBench/task_config/flexiv_embodiment_smoke.yml").read_text())
    assert task["embodiment"][:2] == ["flexiv-rizon4s-dual-gn01-left", "flexiv-rizon4s-dual-gn01-right"]
    assert task["camera"]["enabled"] == ["head_camera"]
    assert task["camera"]["collect_wrist_camera"] is False
