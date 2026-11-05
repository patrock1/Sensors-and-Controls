"""
test_metrics.py
Evaluation code checked on synthetic logs, so it can be trusted before the pipeline exists.
"""
import json
import numpy as np
from ur3pick.evaluation.metrics import trial_metrics, summarise, wilson


def _log(lifted=True, final="done", p_err=(0.003, -0.004, 0.0), T=300, dt=0.01):
    t = np.arange(T) * dt
    phase = np.array(["search"] * 50 + ["approach"] * 100 + ["descend"] * 50
                     + ["grasp"] * 30 + ["lift"] * (T - 231) + [final])
    cube = np.tile([0.30, 0.02, 0.02], (T, 1))
    if lifted:
        cube[230:, 2] = 0.17
    tool = cube.copy()
    p_hat = np.full((T, 3), np.nan)
    p_hat[50:] = cube[50:] + np.array(p_err)
    rng = np.random.default_rng(0)
    gt_q, gt_qd = rng.normal(0, 0.1, (T, 6)), rng.normal(0, 0.1, (T, 6))
    P = np.tile(np.eye(12) * 1e-4, (T, 1, 1))
    x_hat = np.hstack([gt_q, gt_qd]) + rng.normal(0, 1e-2, (T, 12))   # error sigma matches P
    return dict(t=t, phase=phase, p_hat=p_hat, gt_p_obj=cube, gt_p_tool=tool, x_hat=x_hat, P_x=P,
                gt_q=gt_q, gt_qdot=gt_qd, u=np.ones((T, 6)),
                p_meas=p_hat.copy(), meas_valid=np.isfinite(p_hat).all(axis=1),   # raw = filtered by default
                meta=json.dumps({"seed": 7, "failure_cause": "none"}))


def test_successful_trial(cfg):
    m = trial_metrics(_log(), cfg)
    assert m["success"] and m["claimed_success"] and not m["disagree"]
    assert abs(m["obj_err_freeze"] - 0.005) < 1e-9          # 3-4-5 triangle, measured at descend
    assert abs(m["obj_err_freeze_xy"] - 0.005) < 1e-9 and m["obj_err_freeze_z"] == 0.0
    assert m["t_grasp"] == 2.0 and m["seed"] == 7
    assert 9 < m["nees_joint"] < 15                          # consistent synthetic filter, n = 12


def test_controller_claims_done_but_cube_on_table(cfg):
    """The case truth-based success exists for."""
    m = trial_metrics(_log(lifted=False), cfg)
    assert not m["success"] and m["claimed_success"] and m["disagree"]


def test_top_face_bias_shows_up_in_z(cfg):
    m = trial_metrics(_log(p_err=(0.0, 0.0, 0.02)), cfg)
    assert abs(m["obj_err_freeze_z"] - 0.02) < 1e-9 and m["obj_err_freeze_xy"] == 0.0


def test_object_never_seen(cfg):
    log = _log(lifted=False, final="failed")
    log["p_hat"][:] = np.nan
    log["phase"][:] = "search"
    m = trial_metrics(log, cfg)
    assert not m["success"] and np.isnan(m["obj_err_freeze"])


def test_large_error_in_raw_measurements_is_sensing(cfg):
    """Hand-eye style failure: raw localisations and the estimate are both 25 mm off (> 20 mm capture radius)."""
    m = trial_metrics(_log(lifted=False, p_err=(0.025, 0.0, 0.0)), cfg)
    assert not m["success"] and m["failure_cause"] == "sensing"
    assert abs(m["meas_err_freeze_xy"] - 0.025) < 1e-9


def test_bias_just_under_radius_in_raw_is_still_sensing(cfg):
    """Raw 19.7 mm, estimate 20.5 mm: the same bias passing through, not a filter fault."""
    log = _log(lifted=False, p_err=(0.0205, 0.0, 0.0))
    log["p_meas"] = log["gt_p_obj"] + np.array([0.0197, 0.0, 0.0])
    assert trial_metrics(log, cfg)["failure_cause"] == "sensing"


def test_good_measurements_bad_estimate_is_estimator(cfg):
    log = _log(lifted=False, p_err=(0.025, 0.0, 0.0))
    log["p_meas"] = log["gt_p_obj"] + np.array([0.002, 0.0, 0.0])   # camera chain fine, filter wrong
    m = trial_metrics(log, cfg)
    assert m["failure_cause"] == "estimator"


def test_state_machine_cause_wins(cfg):
    log = _log(lifted=False, p_err=(0.025, 0.0, 0.0), final="failed")
    log["meta"] = json.dumps({"seed": 7, "failure_cause": "timeout"})
    assert trial_metrics(log, cfg)["failure_cause"] == "timeout"


def test_wilson_stays_inside_0_1():
    lo, hi = wilson(50, 50)
    assert 0.9 < lo < 1.0 and hi <= 1.0                       # normal approx would give [1, 1]
    lo, hi = wilson(0, 10)
    assert lo >= 0.0 and hi > 0.2


def test_summarise(cfg):
    rows = [trial_metrics(_log(), cfg) for _ in range(9)] + [trial_metrics(_log(lifted=False), cfg)]
    S = summarise(rows)
    assert S["n"] == 10 and S["success_rate"] == 0.9 and S["disagreements"] == 1
    assert S["ci_low"] < 0.9 < S["ci_high"] <= 1.0
    assert S["failure_causes"] == {"undiagnosed": 1}          # meta said none: a diagnosis is still owed
