"""
frames.py
Part B every transform in one palce.
"""

import numpy as np
from ur3pick.modelling import ur3_params as P

def T_B_T(q):
    return P.fkine(q)

def T_T_C(error=None):
    T = P.T_T_C()
    if error is not None:
        T = T.copy(); T[:3, 3] += np.asarray(error, float)
    return T

def T_B_C(q, hand_eye_error=None):
    """Camera pose in base. hand_eye_error (m, tool frame) is the estimator's calibration error for the Part E sweep; the sim camera never uses this."""
    return T_B_T(q) @ T_T_C(hand_eye_error)

def inv(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def transform_point(T, p):
    return (T @ np.r_[p, 1.0])[:3]

def project(K, T_B_C_, p_B):
    """Pixel of a base-frame point seen by a camera at T_B_C_ (OpenCV axes). Returns (uv, depth)."""
    pc = transform_point(inv(T_B_C_), p_B)
    uvw = K @ pc
    return uvw[:2] / uvw[2], pc[2]