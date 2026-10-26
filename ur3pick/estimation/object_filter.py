"""
object_filter.py
Part C: filter for the object position in the base frame (static object, camera updates)
"""

import numpy as np
from ur3pick.interfaces import ObjectMeasurement

CHI2_3DOF_99 = 11.34

class ObjectFilter:
    """Random-walk model p+ = p + w, w~ N(0, q_obj dt I); measurement y = p + v, v ~ N(0, R).
    q_obj is small because the cube is static, but NOT zero, a zero Qd makes the filter overconfident and unable to follow a cube that gets nudged
    """
    def __init__(self, q_obj=1e-6, gate=CHI2_3DOF_99):
        self.q_obj, self.gate = q_obj, gate
        self.p_hat = np.full(3, np.nan)
        self.P = np.eye(3) * 1e6
        self.seen = False
        self.last_nis = np.nan

    def predict(self, dt):
        if self.seen:
            self.P = self.P + self.q_obj * dt * np.eye(3)

    def update(self, meas: ObjectMeasurement) -> bool:
        """Returns True ig the measurement was used."""
        if not meas.valid:
            return False
        if not self.seen:
            self.p_hat, self.P, self.seen = meas.p_B.copy(), meas.R.copy(), True
            return True
        S = self.P + meas.R
        nu = meas.p_B - self.p_hat
        nis = float(nu @ np.linalg.solve(S, nu))
        self.last_nis = nis
        if nis > self.gate:
            return False
        K = self.P @ np.linalg.inv(S)
        self.p_hat = self.p_hat + K @ nu
        I = np.eye(3)
        self.P = (I - K) @ self.P @ (I - K).T + K @ meas.R @ K.T
        return True