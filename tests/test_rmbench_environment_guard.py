from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "scripts" / "rmbench" / "bootstrap_env.sh"
FETCH = ROOT / "scripts" / "rmbench" / "fetch_assets.sh"


def test_bootstrap_refuses_existing_dp3_environment():
    text = BOOTSTRAP.read_text(encoding="utf-8")
    assert "bootstrap_env.sh refuses to run from the existing dp3 environment" in text
    assert 'ENV_NAME="dp3-rmbench"' in text
    assert "python=3.10" in text
    assert "PYTHONNOUSERSITE=1" in text


def test_asset_fetch_is_scoped_and_does_not_download_full_dataset():
    text = FETCH.read_text(encoding="utf-8")
    downloader = (ROOT / "third_party" / "sim" / "RMBench" / "assets" / "_download.py").read_text(
        encoding="utf-8"
    )
    assert "embodiments/**" in text
    assert "objects/**" in text
    assert 'repo_type="dataset"' in downloader
    assert "snapshot_download" in downloader
    assert "data/**" not in text
    assert "d899d72b53270a89f71d216c08ecbd4d9a7004fd" in text
    assert "RMBENCH_HF_REVISION" in text
