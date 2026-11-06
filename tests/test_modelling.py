"""
test_modelling.py
"""

import numpy as np
import control as ct
from ur3pick.modelling import state_space as ss, analysis as an
from ur3pick.estimation.joint_observer import steady_state
from ur3pick.control.lqr import bryson, design_dlqr
from ur3pick.modelling import ur3_params as P
 
 
def _model(cfg):
    M = np.array(cfg["modelling"]["joint_inertia"])
    A, B, C, D = ss.build_model(M, cfg["modelling"]["damping"])
    return M, (A, B, C, D)
 
 
def test_rank_tests(cfg):
    M, (A, B, C, D) = _model(cfg)
    assert A.shape == (12, 12) and B.shape == (12, 6) and C.shape == (6, 12)
    reports = an.per_joint_reports(A, B, C)
    assert len(reports) == 6                                # guards against an early return in the loop
    for r in reports:                                       # Week 5 C3/C4 per joint (2 states each)
        assert r["ctrb"]["rank"] == 2 and r["obsv"]["rank"] == 2
        assert r["ctrb"]["min_sv"] > 0 and r["obsv"]["min_sv"] > 0
    assert an.pbh_controllable(A, B) and an.pbh_observable(A, C)    # full 12-state model
 
 
def test_encoders_only_observes_velocity_but_velocity_only_does_not(cfg):
    """Week 5 C4: some sensor sets fail the rank test. Velocity-only cannot see joint position."""
    M, (A, B, C, D) = _model(cfg)
    sets = {"encoders (q)": C, "tachometers (qdot)": np.hstack([np.zeros((6, 6)), np.eye(6)])}
    rep = an.sensor_set_report(A, sets)
    assert rep["encoders (q)"]["observable"] is True
    assert rep["tachometers (qdot)"]["observable"] is False
 
 
def test_zoh_is_not_euler(cfg):
    M, (A, B, C, D) = _model(cfg)
    dt = cfg["sim"]["dt_ctrl"]
    Ad, Bd, _, _ = ss.discretise(A, B, C, D, dt)
    from scipy.linalg import expm
    assert np.allclose(Ad, expm(A * dt), atol=1e-10)                  # exact ZOH
    # forward Euler would give Bd = B*dt, i.e. zero position response to a torque in one step.
    # ZOH keeps the T^2/(2M) term (Week 5, C5):
    assert abs(Bd[0, 0] - dt**2 / (2 * M[0])) / (dt**2 / (2 * M[0])) < 0.05
    assert abs((B * dt)[0, 0]) == 0.0 and Bd[0, 0] > 0
 
 
def test_separation_principle(cfg):
    """Week 8 A.6: closed-loop eigenvalues are the union of controller and estimator eigenvalues."""
    M, (A, B, C, D) = _model(cfg)
    dt = cfg["sim"]["dt_ctrl"]
    Ad, Bd, Cd, Dd = ss.discretise(A, B, C, D, dt)
    Qw = bryson(np.r_[cfg["lqr"]["q_err_max"], cfg["lqr"]["qd_max"]])
    Rw = bryson(P.TORQUE_LIMIT)
    Kr, _, _ = design_dlqr(Ad, Bd, Qw, Rw)
    qa = ss.q_acc_from_torque_sigma(M, cfg["observer"]["sigma_tau"], dt)
    Qd = ss.process_noise(dt, qa)
    Rn = np.eye(6) * cfg["noise"]["sigma_q"] ** 2
    _, Kf, Ppost = steady_state(Ad, Cd, Qd, Rn)
    assert an.separation_check(Ad, Bd, Cd, Kr, Kf) < 1e-8
    assert np.all(np.linalg.eigvals(Ad - Bd @ Kr).__abs__() < 1)
    ratio = an.speed_ratio(Ad, Bd, Cd, Kr, Kf, dt)
    assert np.isfinite(ratio) and ratio > 0

def test_full_inertia_matrix_reduces_to_diagonal_model(cfg):
    """build_model accepts the full 6x6 M (coupled design, issue #4). A diagonal matrix must give
    exactly the decoupled model, so the two options differ only by the off-diagonal coupling."""
    d = np.array(cfg["modelling"]["joint_inertia"])
    for a, b in zip(ss.build_model(d, 1.0), ss.build_model(np.diag(d), 1.0)):
        assert np.allclose(a, b)


def test_config_inertia_matrix_is_consistent(cfg):
    """The full M in the config must be a valid inertia matrix whose diagonal is joint_inertia."""
    M = np.array(cfg["modelling"]["inertia_matrix"])
    assert M.shape == (6, 6) and np.allclose(M, M.T)
    assert np.all(np.linalg.eigvalsh(M) > 0)
    assert np.allclose(np.diag(M), cfg["modelling"]["joint_inertia"])
