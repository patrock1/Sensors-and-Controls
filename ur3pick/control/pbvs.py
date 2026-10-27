"""
pbvs.py
Part D - Position based visual servoing 
"""
import numpy as np
from spatialmath.base import trlog

def pbvs_twist(t, t_star, lam, R=None, R_star=None):
    t, t_star = np.asarray(t, float), np.asarray(t_star, float)
    e_t = t - t_star
    if R is None:
        omega = np.zeros(3)
    else:
        e_theta = trlog(np.asarray(R) @ np.asarray(R_star).T, twist=True)
        omega = lam * np.asarray(e_theta)
    v = lam * e_t + np.cross(t, omega)
    return np.r_[v, omega]

