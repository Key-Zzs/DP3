from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PIN = "92ef7865d76585e6e08d291bdfe652d32f7740f4"


def test_official_flexiv_description_is_pinned_to_humble_v1():
    source = ROOT / "third_party/sim/flexiv_description"
    assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() == PIN
    assert "https://github.com/flexivrobotics/flexiv_description.git" in (ROOT / ".gitmodules").read_text()
    vendor = (ROOT / "third_party/vendor/flexiv_description.md").read_text()
    assert "humble-v1" in vendor
    assert PIN in vendor
    assert "Apache-2.0" in vendor
    assert (source / "LICENSE").is_file()


def test_official_submodule_worktree_is_clean():
    source = ROOT / "third_party/sim/flexiv_description"
    assert subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True) == ""
