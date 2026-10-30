"""Shared helpers for the scripts: path setup, headless GL, seed lists, sweep loading."""
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")      # headless rendering on Linux; Windows/macOS need nothing

import yaml  # noqa: E402


def seed_list(which):
    with open(ROOT / "configs" / "seeds.yaml") as f:
        s = yaml.safe_load(f)[which]
    return list(range(s["start"], s["start"] + s["count"]))


def load_sweep(name):
    with open(ROOT / "configs" / "sweeps" / f"{name}.yaml") as f:
        return yaml.safe_load(f)
