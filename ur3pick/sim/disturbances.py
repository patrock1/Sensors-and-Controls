"""
disturbances.py
Sim side Part E disturbances, all read from cfg["disturbance"] (empty dict = nominal trial).
  force:        [fx, fy, fz] N in world frame, applied to body (default wrist3, end of arm) for t_start <= t < t_end
  blackout:     [[t0, t1], ...] s, camera gives no frame inside these windows
  encoder_bias: rad, constant offset on every encoder reading (scalar or 6 values)
Only applied inside sim/, so the pipeline sees a disturbance the same way it would see a real one.
"""
import numpy as np
import mujoco


class Disturbances:
    def __init__(self, model, cfg: dict):
        d = cfg.get("disturbance") or {}
        self.force = np.asarray(d.get("force", [0.0, 0.0, 0.0]), float)
        self.t_start = float(d.get("t_start", 0.0))
        self.t_end = float(d.get("t_end", 0.0))
        name = d.get("body", "wrist3")      # push the arm, a 5-10 N push on the 50 g cube would launch it
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if self.body < 0:
            raise KeyError(f"disturbance body '{name}' not in scene")
        self.blackout = [(float(a), float(b)) for a, b in d.get("blackout", [])]
        self.encoder_bias = np.broadcast_to(np.asarray(d.get("encoder_bias", 0.0), float), (6,)).copy()

    def apply(self, data) -> None:
        """Call before every mj_step. Force acts at body centre of mass, world frame."""
        on = self.t_start <= data.time < self.t_end
        data.xfrc_applied[self.body, :3] = self.force if on else 0.0

    def camera_blocked(self, t: float) -> bool:
        return any(a <= t < b for a, b in self.blackout)
