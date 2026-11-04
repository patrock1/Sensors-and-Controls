"""
randomise.py
Sim side: random cube pose each trial (Track B constraint, object position randomised)
"""
import numpy as np


def sample_object_pose(rng: np.random.Generator, cfg: dict):
    """Returns (p_B_O (3,), yaw rad). Region from cfg object.region, cube resting on table."""
    reg = cfg["object"]["region"]
    x = rng.uniform(reg["x_min"], reg["x_max"])
    y = rng.uniform(reg["y_min"], reg["y_max"])
    z = cfg["object"]["size"] / 2          # cube centre when resting on table, table top z = 0
    yaw = rng.uniform(0.0, np.pi / 2)      # cube looks same every 90 deg, wider range adds nothing
    return np.array([x, y, z]), float(yaw)
