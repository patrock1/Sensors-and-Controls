"""
state_space.py

Part A, state space model of the ER3 joints, in refference to week 5 tut

per joint: M qddot + b qdot = u

assumptions ()
1. gravity is compensated (feedforward from the estimate) or removed by the integral action.
2. Inertia is the diagonal of the Mujoco mass matrix at the operatin pose. The coupling between the joints is neglected: sim.inertia.coupling_ratio() quantifies what is dropped.
3. Friction is linear viscous damping b.

Discretisation use the exact zero-order hold, NOT forward euler.
"""

import numpy as np
import control as ct

N = 6

def build_model(M, b=None):
    """Continous-time (A, B, C, D) for x [q, qdot], from M qdd + b qd = tau.

    M is either the 6 diagonal inertias (decoupled model: each joint its own double integrator)
    or the full 6x6 inertia matrix at the linearisation pose (coupled model). The coupled form is
    what the controller needs: at the home pose joints 2 and 3 are 0.64 coupled, so the fast
    joint-3 mode sees only about M33 (1 - 0.64^2) of its diagonal inertia and a decoupled design
    over-estimates it (see issue #4)."""
    M = np.asarray(M, float)
    Minv = np.linalg.inv(M) if M.ndim == 2 else np.diag(1.0 / M)
    b = np.zeros(N) if b is None else np.broadcast_to(np.asarray(b, float), (N,))
    Z, I = np.zeros((N, N)), np.eye(N)
    A = np.block([[Z, I], [Z, -Minv @ np.diag(b)]])
    B = np.vstack([Z, Minv])
    C = np.hstack([I, Z])
    D = np.zeros((N, N))
    return A, B, C, D

def discretise(A, B, C, D, dt):
    """Exact ZOH discretisation (same call as the Week 5 tutorial)."""
    d = ct.ss(A, B, C, D).sample(dt, method="zoh")
    return d.A, d.B, d.C, d.D

def process_noise(dt, q_acc):
    """ Discrete process covariance qd for a white-acceleration disturbance."""
    qa = np.broadcast_to(np.asarray(q_acc, float), (N,))
    return np.block([[np.diag(qa * dt**3 / 3), np.diag(qa * dt**2 / 2)],
                     [np.diag(qa * dt**2 / 2), np.diag(qa * dt)]])

def q_acc_from_torque_sigma(M, sigma_tau, dt):
    """ turn a unmodelled_torque standard deviation (N m) into a white-acceleration PSD."""
    return(np.asarray(sigma_tau, float) / np.asarray(M, float)) ** 2 * dt
