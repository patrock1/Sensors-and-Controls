"""
calibration.py
Part B: camera instrinsics
"""

import numpy as np

def intrinsics_from_fovy(fovy_deg, width, height):
    fy = (height / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    return np.array([[fy, 0.0, width / 2.0],
                     [0.0, fy, height / 2.0],
                     [0.0, 0.0, 1.0]])