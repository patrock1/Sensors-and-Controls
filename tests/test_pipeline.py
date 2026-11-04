import json
import numpy as np
import pytest

from ur3pick.pipeline import load_cfg, run_trial
from ur3pick.evaluation.metrics import trial_metrics

REQUIRED = ["t", "q_meas", "x_hat", "P_x", "p_hat", "P_p", "p_meas", "meas_valid", "u", "phase","gt_q", "gt_qdot", "gt_p_obj", "gt_p_tool", "meta"]

@pytest.fixture(scope="module")
def run():
    cfg = load_cfg()
    return run_trial(0, cfg), cfg

def test_log_has_every_key(run):
    log, _ = run
    missing = [k for k in REQUIRED if k not in log]
    assert not missing, f"log is missing {missing}"

def test_log_shapes(run):
    log, _ = run
    T = len(log["t"])
    assert T > 0 
    expect = {"q_meas": (T, 6), "x_hat": (T, 12), "P_x": (T, 12, 12), "p_hat": (T, 3), "P_p": (T, 3, 3),
              "p_meas": (T, 3), "meas_valid": (T,), "u": (T, 6), "phase": (T,),
              "gt_q": (T, 6), "gt_qdot": (T, 6), "gt_p_obj": (T, 3), "gt_p_tool": (T, 3)}

    for k, shape in expect.items():
        assert log[k].shape == shape, f"{k}:{log[k].shape} != {shape} "

def test_meta_and_metrics(run):
    log, cfg = run
    meta = json.loads(str(log["meta"]))
    assert meta["seed"] == 0 and "failure_cause" in meta
    m = trial_metrics(log, cfg)
    assert m["final_phase"] in ("done", "failed")

def test_phase_are_known(run):
    log, _ = run
    known = {"search", "approach", "descend", "grasp", "lift", "done", "failed"}
    assert set(np.unique(log["phase"])) <= known

def test_load_cfg_overrides():
    cfg = load_cfg(overrides={"sim.gravcomp": 0.9, "disturbance": {"force": [0.0, 5.0, 0.0]}})
    assert cfg["sim"]["gravcomp"] == 0.9
    assert cfg["disturbance"]["force"] == [0.0, 5.0, 0.0]
    assert cfg["sim"]["dt_ctrl"] == load_cfg()["sim"]["dt_ctrl"]
