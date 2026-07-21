from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party" / "sim" / "RMBench"


def test_vendor_pin_and_metadata_are_present():
    metadata = (VENDOR / "README_VENDOR.md").read_text(encoding="utf-8")
    assert "https://github.com/RoboTwin-Platform/RMBench.git" in metadata
    assert "87e0498891073d483d330195c0f160709bd92ff5" in metadata
    assert "TianxingChen/RMBench" in metadata
    assert (VENDOR / "LICENSE").is_file()
    assert (VENDOR / "README.md").is_file()


def test_upstream_asset_downloader_remains_available():
    assert (VENDOR / "assets" / "_download.py").is_file()
    assert (VENDOR / "task_config" / "demo_clean.yml").is_file()
