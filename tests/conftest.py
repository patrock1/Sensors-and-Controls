"""conftest.py"""

import pathlib
import sys
import yaml
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

@pytest.fixture(scope="session")
def cfg():
    with open(ROOT /  "configs" / "default.yaml") as f:
        return yaml.safe_load(f)
    
