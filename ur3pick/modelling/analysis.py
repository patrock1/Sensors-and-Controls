"""
analysis.py
Part A C D evidence rank tests with singular values , seperation check
"""

import numpy as np
import control as ct

def controllability_report(A, B):
    Co = ct.ctrb(A,B)
    sv = np.linalg.svd(Co, compute_uv=False)
    return dict(rank=int(np.linalg.matrix_rank(Co)), n=A.shape[0], min_sv=float(sv.min()), max_sv=float(sv.max()))

def observability_report(A,C):
    Ob = ct.obsv(A, C)
    sv=np.linalg.svd(Ob, compute_uv=False)
    return dict(rank=int(np.linalg.matrix_rank(Ob)), n=A.shape[0], min_sv=float(sv.min()), max_sv=float(sv.max()))


def pbh_controllable(A, B, tol=1e-8):
    """PBH test rank [lambda I - A, B] = n for every eigenvalue lambda of A."""
    n = A.shape[0]
    return all(np.linalg.matrix_rank(np.hstack([lam * np.eye(n) - A, B]), tol=tol) == n
            for lam in np.unique(np.round(np.linalg.eigvals(A), 9)))

def pbh_observable(A, C, tol=1e-8):
    n = A.shape[0]
    return all(np.linalg.matrix_rank(np.vstack([lam * np.eye(n) - A, C]), tol=tol) == n
            for lam in np.unique(np.round(np.linalg.eigvals(A), 9)))

def per_joint_reports(A, B, C, n_joints=6):
    """ctrb/obsv ranks and singular values for each joints's [q, qdot] block (2 states each)"""
    out = []
    for i in range(n_joints):
        idx = [i, n_joints + i]
        Ai, Bi, Ci = A[np.ix_(idx, idx)], B[idx][:, [i]], C[[i]][:, idx]
        out.append(dict(joint=i + 1, ctrb=controllability_report(Ai, Bi), obsv=observability_report(Ai, Ci)))
    return out

def sensor_set_report(A, sensor_sets):
    return {name: dict(observable=pbh_observable(A, C)) for name, C in sensor_sets.items()}

def separation_check(Ad, Bd, Cd, Kr, Kf):
    n = Ad.shape[0]
    Ae = (np.eye(n) - Kf @ Cd) @ Ad
    Acl = np.block([[Ad - Bd @ Kr, Bd @ Kr], [np.zeros((n, n)), Ae]])
    eig_cl = np.sort_complex(np.linalg.eigvals(Acl))
    union = np.sort_complex(np.concatenate([np.linalg.eigvals(Ad - Bd @ Kr), np.linalg.eigvals(Ae)]))
    return float(np.max(np.abs(eig_cl - union)))

def _decay_rates(z, dt):
    z = np.asarray(z, complex)
    s = np.log(np.maximum(np.abs(z), 1e-12)) / dt
    return -s

def speed_ratio(Ad, Bd, Cd, Kr, Kf, dt):
    n = Ad.shape[0]
    rc = _decay_rates(np.linalg.eigvals(Ad - Bd @ Kr), dt).min()
    re = _decay_rates(np.linalg.eigvals((np.eye(n) - Kf @ Cd) @ Ad), dt).min()
    return float(re/rc)


