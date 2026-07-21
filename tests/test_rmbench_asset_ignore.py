import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def is_ignored(path: str) -> bool:
    return subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", path], cwd=ROOT
    ).returncode == 0


def test_scoped_assets_and_runtime_artifacts_are_ignored():
    paths = (
        "third_party/sim/RMBench/assets/embodiments/example/file.obj",
        "third_party/sim/RMBench/assets/objects/example/file.obj",
        "third_party/sim/RMBench/data/example.zarr",
        "third_party/sim/RMBench/envs/curobo/src/example.py",
        "third_party/sim/RMBench/.cache/example",
        "third_party/sim/RMBench/example.mp4",
        "third_party/sim/RMBench/example.log",
    )
    assert all(is_ignored(path) for path in paths)
