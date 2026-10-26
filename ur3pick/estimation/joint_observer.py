"""
joint_observer.py
Part C kalman filter for the joint state
"""
import numpy as np
from scipy.linalg import solve_discrete_are

def kf_step(xh, P, u, y, A, B, C, Qd, Rn):
    """One predict + correct step """
    xh_pred = A @ xh + B @ np.atleast_1d(u)
    P_pred = A @ P @ A.T + Qd
    S = C @ P_pred @ C.T + Rn
    K = P_pred @ C.T @ np.linalg.inv(S)
    nu = np.atleast_1d(y) - C @ xh_pred
    xh_new = xh_pred + K @ nu
    I = np.eye(len(xh))
    P_new = (I - K @ C) @ P_pred @ (I - K @ C).T + K @ Rn @ K.T      # Joseph form (stays PSD)
    return xh_new, P_new, dict(x_pred=xh_pred, P_pred=P_pred, K=K, nu=nu, S=S)

def steady_state(Ad, Cd, Qd, Rn):
    """Steady-state predictor covariance, gain and posterior covariance ()
       P_post diagonal gives the 1-sigma error per state the filter will settle at, which is the
       'P tells hou how good the estimate will be'"""
    
    Pp = solve_discrete_are(Ad.T, Cd.T, Qd, Rn)
    K = Pp @ Cd.T @ np.linalg.inv(Cd @ Pp @ Cd.T + Rn)
    P_post = (np.eye(Ad.shape[0]) - K @ Cd) @ Pp
    return Pp, K, P_post

class JointKF:
    """Predict every control tick; correct whenever a measurement arrives"""
    def __init__(self, Ad, Bd, Cd, Qd, Rn, x0, P0):
        self.A, self.B, self.C, self.Qd, self.Rn = Ad, Bd, Cd, Qd, Rn
        self.x_hat = np.asarray(x0, float).copy()
        self.P = np.asarray(P0, float).copy()
        self.last = {}

    def predict(self, u):
        self.x_hat = self.A @ self.x_hat + self.B @ np.atleast_1d(u)
        self.P = self.A @ self.P @ self.A.T + self.Qd

    def update(self, y, C=None, Rn=None):
        C = self.C if C is None else C
        Rn = self.Rn if Rn is None else Rn
        S = C @ self.P @ C.T + Rn
        K = self.P @ C.T @ np.linalg.inv(S)
        nu = np.atleast_1d(y) - C @ self.x_hat
        self.x_hat = self.x_hat + K @ nu
        I = np.eye(len(self.x_hat))
        self.P = (I - K @ C) @ self.P @ (I - K @ C).T + K @ Rn @ K.T
        self.last = dict(nu=nu, S=S, K=K)