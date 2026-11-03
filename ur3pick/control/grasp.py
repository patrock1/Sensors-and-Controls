import numpy as np

from ur3pick.interfaces import Estimate
from ur3pick.modelling import ur3_params as P
from ur3pick.sensing import frames
from ur3pick.control.pbvs import pbvs_twist
from ur3pick.control.servo import resolved_rate, ReferenceIntegrator

class GraspStateMachine:
    PHASES = ["search", "approach", "descend", "grasp", "lift", "done", "failed"]

    def __init__(self, cfg:dict, dt:float, q0):
        self.g, self.s, self.dt = cfg["grasp"], cfg["servo"], dt
        self.robot = P.make_robot("tool")
        self.ref = ReferenceIntegrator(q0, dt, P.Q_LIMIT)
        self.phase, self.t_phase = "search", None
        self.gripper = 0.0
        self.p_target = None          # frozen cube estimate (descend onwards)
        self.p_lift = None            # tool0 goal in base frame for the lift
        self.failure_cause = "none"

    def _go(self, phase,t):
        self.phase, self.t_phase = phase, t

    def _fail(self, cause, t):
        self.failure_cause = cause
        self._go("failed", t)

    def _servo_to_cube(self, q, p_B, height):
        p_T = frames.transform_point(frames.inv(frames.T_B_T(q)), p_B)
        t_star = np.array([0, 0, height])
        nu = pbvs_twist(p_T, t_star, self.g["lam"])
        qd, _ = resolved_rate(self.robot, q, nu, self.s["qd_limit"])
        return qd, float(np.linalg.norm(p_T - t_star))

    def _servo_tool_to(self, q, p_goal_B):
        T = frames.T_B_T(q)
        e_B = p_goal_B - T[:3, 3]
        nu = np.r_[self.s["lam"] * (T[:3, :3].T @ e_B), np.zeros(3)]  
        qd, _ = resolved_rate(self.robot, q, nu, self.s["qd_limit"])
        return qd, float(np.linalg.norm(e_B))

    def step(self, est: Estimate):
        t, q = est.t, est.x_hat[:6]
        if self.t_phase is None:
            self.t_phase = t
        in_phase = t - self.t_phase
        qd = np.zeros(6)

        if self.phase == "search":
            if est.p_seen:
                self._go("approach", t)
            elif in_phase > self.g["search_timeout"]:
                self._fail("sensing", t)

        elif self.phase == "approach":
            qd, err = self._servo_to_cube(q, est.p_hat, self.s["pre_grasp_height"])
            if err < self.g["approach_tol"]:
                self.q_target = est.p_hat.copy()
                self._go("descend", t)

        elif self.phase == "descend":
            qd, err = self._servo_to_cube(q, self.q_target, self.s["descend_height"])
            if err < self.g["descend_tol"]:
                self._go("grasp", t)

        elif self.phase == "grasp":
            if in_phase >= self.g["settle_time"]:
                self.gripper = 1.0
                self.p_lift = frames.T_B_T(q)[:3, 3] + np.array([0, 0, self.g["lift_height"]])
                self._go("lift", t)

        elif self.phase == "lift":
            qd, err = self._servo_tool_to(q, self.p_lift)
            if err < self.g["lift_tol"]:
                self._go("done", t)

        if self.phase == "failed":
            self.gripper = 0.0
        elif self.phase not in ("search", "done") and in_phase > self.g["phase_timeout"]:
            self._fail("timeout", t)

        q_ref, qd_ref = self.ref.step(q, qd)    
        return np.r_[q_ref, qd_ref], self.gripper, self.phase

    @property
    def wants_object_updates(self) -> bool:
        """Camera measurements only feed the object filter before the estimate is frozen."""
        return self.phase in ("search", "approach")
 
    @property
    def finished(self) -> bool:
        return self.phase in ("done", "failed")