"""
sensors.py
Sim side: the ONLY pipeline-facing path to simulator state (interfaces.md 3.1).
Encoders: q only plus gaussian noise (+ Part E bias), no qdot (reason observer exists).
Wrist camera: rgb + depth every dt_cam, range dependent depth noise, random dropouts, Part E blackout windows.
"""
import numpy as np
import mujoco

from ur3pick.interfaces import SensorPacket

CAMERA = "wrist_cam"


class SensorSuite:
    def __init__(self, scene, cfg: dict, rng: np.random.Generator):
        self.scene, self.rng = scene, rng
        sim, noise = cfg["sim"], cfg["noise"]
        self.h, self.w = sim["image_height"], sim["image_width"]
        self.dt_cam = sim["dt_cam"]
        self.sigma_q = noise["sigma_q"]
        self.depth_a, self.depth_b = noise["depth_a"], noise["depth_b"]
        self.dropout = noise["dropout_prob"]
        # separate renderers for rgb and depth: toggling one renderer between modes
        # dropped the cube from later rgb frames on macOS
        self.rgb_r = mujoco.Renderer(scene.model, self.h, self.w)
        self.depth_r = mujoco.Renderer(scene.model, self.h, self.w)
        self.depth_r.enable_depth_rendering()
        self.next_cam_t = scene.t

    def read(self) -> SensorPacket:
        t = self.scene.t
        dist = self.scene.disturb
        q_meas = self.scene.data.qpos[:6] + self.rng.normal(0.0, self.sigma_q, 6) + dist.encoder_bias
        rgb = depth = stamp = None
        if t >= self.next_cam_t - 1e-6:                  # camera tick (every 5th control tick at 20 Hz)
            self.next_cam_t += self.dt_cam
            dropped = self.rng.random() < self.dropout     # drawn every tick so blackout does not shift the noise stream
            if not dropped and not dist.camera_blocked(t):
                rgb, depth = self._render()
                stamp = t
        return SensorPacket(t=t, q_meas=q_meas, rgb=rgb, depth=depth, cam_stamp=stamp)

    def _render(self):
        d = self.scene.data
        self.rgb_r.update_scene(d, camera=CAMERA)
        rgb = self.rgb_r.render().copy()
        self.depth_r.update_scene(d, camera=CAMERA)
        Z = self.depth_r.render().astype(np.float64)     # metres along camera z
        sigma = self.depth_a + self.depth_b * Z ** 2      # Week 4 RGB-D model, same one localise() uses for R
        depth = Z + self.rng.normal(0.0, 1.0, Z.shape) * sigma
        return rgb, depth.astype(np.float32)
