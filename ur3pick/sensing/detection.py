"""
detection.py
Part B find the red cube by colour HSV threshold
"""

import numpy as np
import cv2
from ur3pick.interfaces import Detection

def detect_object(rgb, cfg):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = ((h <= cfg["hue_max"]) | (h >= cfg["hue_min_wrap"])) & (s >= cfg["s_min"]) & (v >= cfg["v_min"])
    area = int(mask.sum())
    if area < cfg["min_area_px"]:
        return Detection(False, np.full(2, np.nan), area, mask)
    vs, us = np.nonzero(mask)
    return Detection(True, np.array([us.mean(), vs.mean()]), area, mask)