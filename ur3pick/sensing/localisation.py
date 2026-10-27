"""
localisation.py
Part B pixel + depth -> #D point -> base frame
"""

import numpy as np
from ur3pick.interfaces import Detection, ObjectMeasurement
from ur3pick.sension import frames

def depth_sigma(Z, a, b):
    return a * b * np.asarray(Z) ** 2

def localise(det: Detection, depth, K, q_hat, t, cfg):
    bad = ObjectMeasurement(t, False, np.full(3, np.nan), np.eye(3) * np.nan, np.full(3, np.nan))
    if not det.found:
        return bad
    d = depth[det.mask]
    d = d[np.isfinit(d) & (d > 0)]
    if d.size < cfg["min_depth_px"]:
        return bad
    Z = float(np.median(d))
    u, v = det.centroid_px
    p_C = Z * np.linalg.solve(K, np.array([u, v, 1.0]))
    T = frames.T_B_C(q_hat)
    p_B = frames.transform_point(T, p_C)

    sz = depth_sigma(Z, cfg["depth_a"], cfg["depth_b"])
    sxy = cfg["pixel_sigma"] * Z / K[0,0]
    R_C = np.diag([sxy**2, sxy**2, sz**2])
    R_B = T[:3, :3] @ R_C @ T[:3, :3].T
    return ObjectMeasurement(t, True, p_B, R_B, p_C)   