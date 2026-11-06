"""
test_sweeps.py
Every sweep config must only override keys that exist in default.yaml, so a typo in a sweep
(e.g. "lqr.q_err_mx") fails here instead of silently running the nominal case.
"""
import pathlib

import pytest
import yaml

from ur3pick.pipeline import load_cfg

SWEEPS = sorted((pathlib.Path(__file__).resolve().parents[1] / "configs" / "sweeps").glob("*.yaml"))
OPEN_GROUPS = {"disturbance", "calibration"}      # groups that are empty/absent by default and filled by sweeps


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.stem)
def test_sweep_overrides_exist(path):
    base = load_cfg()
    sweep = yaml.safe_load(open(path))
    assert sweep["levels"], f"{path.stem} has no levels"
    for level in sweep["levels"]:
        for key in level["overrides"]:
            *parents, leaf = key.split(".")
            if (parents or [leaf])[0] in OPEN_GROUPS:
                continue
            node = base
            for p in parents:
                assert p in node, f"{path.stem} / {level['name']}: unknown group '{p}' in '{key}'"
                node = node[p]
            assert leaf in node, f"{path.stem} / {level['name']}: unknown key '{key}'"