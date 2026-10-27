"""
servo.py

Visual servo plumbing camera/tool twist -> joint rates -> joint reference.

"""

import numpy as np

def resolved_rate(robot, q, nu, qd_max=None):
    """"""
    J = robot.jacobe(q)
    qd = np.linalg.pinv(J) @ nu
    if qd_max is not None:
        peak = np.max(np.abs(qd))
        if peak > qd_max:
            qd = qd * (qd_max / peak)
    return qd, float(robot.manipulability(q))

class ReferenceIntegrator:
    """Integrates the commanded joint rate intoa joint refence for the LDR to track.
    Initialised from the first ESTIMATE, then integrates the command not the noisy estimate so measurement noise does not leak int the refference"""

    def __init__(self, q0, dt, qlim=None):
        self.q_ref, self.dt = np.asarray(q0, float).copy(), dt
        self.qlim = qlim

    def step(self, qd):
        self.q_ref = self.q_ref + qd * self.dt
        if self.qlim is not None:
            self.q_ref = np.clip(self.q_ref, self.qlim[0], self.qlim[1])
        return self.q_ref.copy(), np.asarray(qd, float).copy()
    