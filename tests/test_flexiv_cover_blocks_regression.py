import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RMBENCH = ROOT / "third_party/sim/RMBench"
sys.path.insert(0, str(RMBENCH))


def test_existing_cover_blocks_import_contract_is_unchanged():
    old = Path.cwd()
    try:
        os.chdir(RMBENCH)
        module = importlib.import_module("envs.cover_blocks")
        assert module.cover_blocks.__name__ == "cover_blocks"
        assert module.cover_blocks.__module__ == "envs.cover_blocks"
    finally:
        os.chdir(old)
