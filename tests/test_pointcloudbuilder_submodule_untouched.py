import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMODULE = ROOT / "PointCloudBuilder"
EXPECTED_GITLINK = "e19d89cb3e88a09db35eb5cdfbaa992ada5618d5"


def test_pointcloudbuilder_gitlink_and_nested_worktree_are_unchanged():
    entry = subprocess.check_output(
        ["git", "ls-files", "-s", "PointCloudBuilder"], cwd=ROOT, text=True
    ).strip()
    assert entry.startswith(f"160000 {EXPECTED_GITLINK} ")
    status = subprocess.check_output(
        ["git", "-C", str(SUBMODULE), "status", "--porcelain"], text=True
    )
    assert status == ""
