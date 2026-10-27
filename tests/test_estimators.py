"""
test_estimators.py
"""

import numpy as np
from ur3pick.modelling import state_space as ss
from ur3pick.estimation.joint_observer import JointKF
from ur3pick.estimation.consistency import nees, verdict
from ur3pick.estimation.object_filter import ObjectFilter
from ur3pick.interfaces import ObjectMeasurement
 
IDX = [0, 6]     # joint 1: [q, qdot]
 
 
def _joint1(cfg):
    M = np.array(cfg["modelling"]["joint_inertia"])
    A, B, C, D = ss.build_model(M, cfg["modelling"]["damping"])
    dt = cfg["sim"]["dt_ctrl"]
    Ad, Bd, Cd, _ = ss.discretise(A, B, C, D, dt)
    qa = ss.q_acc_from_torque_sigma(M, cfg["observer"]["sigma_tau"], dt)
    Qd = ss.process_noise(dt, qa)
    sel = np.ix_(IDX, IDX)
    return Ad[sel], Bd[IDX][:, [0]], np.array([[1.0, 0.0]]), Qd[sel], np.array([[cfg["noise"]["sigma_q"] ** 2]])
 
 
def _mc_nees(cfg, qd_scale, runs=150, steps=300, seed=0):
    A, B, C, Qd_true, Rn = _joint1(cfg)
    rng = np.random.default_rng(seed)
    Lq = np.linalg.cholesky(Qd_true)
    vals = []
    for _ in range(runs):
        x = np.zeros(2)
        kf = JointKF(A, B, C, Qd_true * qd_scale, Rn, np.zeros(2), np.diag([1e-4, 1e-2]))
        for k in range(steps):
            x = A @ x + Lq @ rng.standard_normal(2)
            y = C @ x + rng.normal(0, np.sqrt(Rn[0, 0]), 1)
            kf.predict(np.zeros(1))
            kf.update(y)
            if k > 100:
                vals.append(nees(x - kf.x_hat, kf.P))
    return float(np.mean(vals))
 
 
def test_kf_consistency_three_tunings(cfg):
    """Week 7 D.3: truth consistent, Qd too small overconfident, Qd too large conservative."""
    n = 2
    good, bad_small, bad_big = (_mc_nees(cfg, s) for s in (1.0, 0.01, 100.0))
    assert verdict(good, n) == "consistent", good
    assert verdict(bad_small, n) == "OVERCONFIDENT", bad_small
    assert verdict(bad_big, n) == "conservative", bad_big
 
 
def test_object_filter_consistent_and_gates_outliers():
    rng = np.random.default_rng(1)
    p_true = np.array([0.3, 0.05, 0.04])
    R = np.diag([2e-5, 2e-5, 5e-5])
    f = ObjectFilter(q_obj=1e-6)
    vals = []
    for k in range(400):
        f.predict(0.05)
        y = p_true + rng.multivariate_normal(np.zeros(3), R)
        f.update(ObjectMeasurement(k * 0.05, True, y, R, np.zeros(3)))
        if k > 20:
            vals.append(nees(p_true - f.p_hat, f.P))
    assert verdict(float(np.mean(vals)), 3) in ("consistent", "conservative")   # q_obj > 0 keeps it honest
    used = f.update(ObjectMeasurement(0.0, True, p_true + [0.5, 0, 0], R, np.zeros(3)))
    assert used is False                                  # wrong blob rejected by the innovation gate
    assert f.update(ObjectMeasurement(0.0, False, np.full(3, np.nan), R, np.zeros(3))) is False