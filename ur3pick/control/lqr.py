"""
lqr.py

Part D inner loop: discrete LQR on the estimate state
Control law is LQG-form, u = -K (xhat - x_ref) - Ki xi, so it only ever sees the estimate
Integral action removes the offset left by the constat gravity or model error.
"""

import numpy as np
import control as ct

def bryson(max_vals):
    """Brysons rules: weight = 1 / largest acceptable value ^ 2"""
    return np.diag(1.0 / np.asarray(max_vals, float) ** 2)

def design_dlqr(Ad, Bd, Qw, Rw):
    K, P, E = ct.dlqr(Ad, Bd, Qw, Rw)
    return np.asarray(K), np.asarray(P), np.asarray(E)

class JointLQR:
    def __init__(self, Ad, Bd, dt, Qw, Rw, umax, integral=False, Qi=None):
        n, m = Bd.shape
        self.n, self.m, self.dt, self.umax, self.integral = n, m, dt, np.asarray(umax, float), integral
        if integral:
            Cq = np.hstack([np.eye(m), np.zeros((m, n - m))])          # integrate the position error
            Aa = np.block([[Ad, np.zeros((n, m))], [-dt * Cq, np.eye(m)]])
            Ba = np.vstack([Bd, np.zeros((m, m))])
            Qa = np.block([[Qw, np.zeros((n, m))], [np.zeros((m, n)), Qi]])
            Ka, _, _ = design_dlqr(Aa, Ba, Qa, Rw)
            self.K, self.Ki = Ka[:, :n], Ka[:, n:]
            self.Cq = Cq
        else:
            self.K, _, _ = design_dlqr(Ad, Bd, Qw, Rw)
            self.Ki = None
        self.xi = np.zeros(m)

    def reset(self):
        self.xi = np.zeros(self.m)

    def command(self, x_hat, x_ref, u_ff=0.0):
        e = x_hat - x_ref
        u = -self.K @ e + u_ff
        if self.integral:
            u = u - self.Ki @ self.xi
        u_sat = np.clip(u, -self.umax, self.umax)
        if self.integral:
            saturated = np.any(u != u_sat)
            if not saturated:
                self.xi = self.xi + self.dt * (-(self.Cq @ e))
        return u_sat
        
        