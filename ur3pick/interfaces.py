from dataclasses import dataclass
import numpy as np

@dataclass
class SensorPacket:
    t: float
    q_meas: np.ndarray              # (6,) noisy encoder angles, rad
    rgb: "np.ndarray | None"        # (H, W, 3) uint8, None when no new frame
    depth: "np.ndarray | None"      # (H, W) metres along camera z, 0 = no return
    cam_stamp: "float | None"

@dataclass
class Detection:
    found: bool
    centroid_px: np.ndarray         # (2,) [u, v]; NaN if not found
    area_px: int
    mask: "np.ndarray | None" = None

@dataclass
class ObjectMeasurement:
    t: float
    valid: bool
    p_B: np.ndarray                 # (3,) object point in base frame, m
    R: np.ndarray                   # (3, 3) measurement covariance in base frame, m^2
    p_C: np.ndarray                 # (3,) object point in camera frame, m

@dataclass
class Estimate:
    t: float
    x_hat: np.ndarray               # (12,) [q, qdot]
    P_x: np.ndarray
    p_hat: np.ndarray               # (3,)
    P_p: np.ndarray
    p_seen: bool

@dataclass
class Command:
    u: np.ndarray                   # (6,) joint torques, N m
    gripper: float
    phase: str