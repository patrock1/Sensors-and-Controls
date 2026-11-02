"""
scene.py
Sim side: loads models/scene.xml, resets a trial, applies commands, steps physics.
Only sim/ and evaluation/ touch MuJoCo state (interfaces.md section 4).
"""
import pathlib

import numpy as np
import mujoco

from ur3pick.interfaces import Command
from ur3pick.sim.randomise import sample_object_pose

SCENE_XML = pathlib.Path(__file__).resolve().parents[2] / "models" / "scene.xml"
ARM_BODIES = ["shoulder", "upper_arm", "forearm", "wrist1", "wrist2", "wrist3"]
GRIPPER_BODY = "wrist3"
WELD_NAME = "grasp"


def _id(model, kind, name):
    i = mujoco.mj_name2id(model, kind, name)
    if i < 0:
        raise KeyError(f"scene.xml is missing '{name}'")
    return i


class Scene:
    def __init__(self, cfg: dict, xml_path=SCENE_XML):
        self.cfg = cfg                                    # read_truth() looks for scene.cfg
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        m, sim = self.model, cfg["sim"]

        m.opt.timestep = sim["dt_sim"]
        self.n_sub = int(round(sim["dt_ctrl"] / sim["dt_sim"]))   # sim steps per control tick

        # gravity compensation level: 1 = perfect (model assumption 1), <1 = Part E mismatch test
        for name in ARM_BODIES:
            m.body_gravcomp[_id(m, mujoco.mjtObj.mjOBJ_BODY, name)] = sim.get("gravcomp", 1.0)

        self._obj = _id(m, mujoco.mjtObj.mjOBJ_BODY, "target")
        self._grip = _id(m, mujoco.mjtObj.mjOBJ_BODY, GRIPPER_BODY)
        self._tool = _id(m, mujoco.mjtObj.mjOBJ_SITE, "tool0")
        self._weld = _id(m, mujoco.mjtObj.mjOBJ_EQUALITY, WELD_NAME)
        j = _id(m, mujoco.mjtObj.mjOBJ_JOINT, "target_free")
        self._obj_qadr = m.jnt_qposadr[j]

        self.rng = None            # spare stream, e.g. disturbances
        self.sensor_rng = None     # pass to SensorSuite
        self.object_pose = None    # (p_B_O, yaw) set at reset, for logging only
        self.attached = False      # True while weld holds cube
        self._grip_closed = False

    def reset(self, seed: int) -> None:
        """Reload state, randomise cube, seed all RNGs from trial seed."""
        # separate streams so changing sensor noise does not move cube for same seed
        obj_ss, sens_ss, spare_ss = np.random.SeedSequence(seed).spawn(3)
        obj_rng = np.random.default_rng(obj_ss)
        self.sensor_rng = np.random.default_rng(sens_ss)
        self.rng = np.random.default_rng(spare_ss)

        d = self.data
        mujoco.mj_resetData(self.model, d)
        d.qpos[:6] = self.cfg["modelling"]["home_q"]

        p, yaw = sample_object_pose(obj_rng, self.cfg)
        a = self._obj_qadr
        d.qpos[a:a + 3] = p
        d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]   # yaw about z
        d.qvel[:] = 0.0
        d.eq_active[self._weld] = 0
        self.attached, self._grip_closed = False, False
        self.object_pose = (p.copy(), yaw)
        mujoco.mj_forward(self.model, d)

    def step(self, cmd: Command) -> None:
        """Apply torques and gripper, advance one control period."""
        self.data.ctrl[:6] = cmd.u                        # ctrlrange in ur3.xml clips to torque limits
        self._set_gripper(cmd.gripper >= 0.5)
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)

    @property
    def t(self) -> float:
        return float(self.data.time)

    # gripper

    def _cube_between_fingers(self) -> bool:
        """Cube centre in tool frame: within half a cube width of tool axis and a short way ahead of tool0."""
        d = self.data
        R = d.site_xmat[self._tool].reshape(3, 3)
        e = R.T @ (d.xpos[self._obj] - d.site_xpos[self._tool])
        lateral_ok = np.hypot(e[0], e[1]) <= self.cfg["object"]["size"] / 2
        depth_ok = 0.0 <= e[2] <= self.cfg["sim"].get("grasp_capture_depth", 0.05)
        return bool(lateral_ok and depth_ok)

    def _set_gripper(self, closed: bool):
        if closed and not self._grip_closed:              # closing edge only, no catching cube later
            if self._cube_between_fingers():
                self._attach()
        elif not closed and self._grip_closed:
            self.data.eq_active[self._weld] = 0
            self.attached = False
        self._grip_closed = closed

    def _attach(self):
        """Weld cube at its CURRENT pose relative to gripper so it does not jump when gripper closes."""
        d = self.data
        R1 = d.xmat[self._grip].reshape(3, 3)
        q1_inv = np.zeros(4)
        mujoco.mju_negQuat(q1_inv, d.xquat[self._grip])
        rel_q = np.zeros(4)
        mujoco.mju_mulQuat(rel_q, q1_inv, d.xquat[self._obj])
        eq = self.model.eq_data[self._weld]
        eq[0:3] = 0.0                                     # anchor at cube origin
        eq[3:6] = R1.T @ (d.xpos[self._obj] - d.xpos[self._grip])   # cube origin in gripper frame
        eq[6:10] = rel_q                                  # cube orientation in gripper frame
        d.eq_active[self._weld] = 1
        self.attached = True
