"""
ur3_params.py
single source of truth for the UR3 kinematic and intertial paramters.

Kinematics - standard distal DH parameters - publish UR# values
"""

from math import pi
import numpy as np
 
# Standard DH: T_i = Rz(q_i) Tz(d_i) Tx(a_i) Rx(alpha_i)
A = np.array([0.0, -0.24365, -0.21325, 0.0, 0.0, 0.0])
D = np.array([0.1519, 0.0, 0.0, 0.11235, 0.08535, 0.0819])
ALPHA = np.array([pi / 2, 0.0, 0.0, pi / 2, -pi / 2, 0.0])
 
# Mass of the moving body attached to joint i (kg). Joint 1 body = shoulder.
MASS = np.array([2.0, 3.42, 1.26, 0.8, 0.8, 0.35])
 
# Approximate torque limits (N m) for the actuators.
TORQUE_LIMIT = np.array([56.0, 56.0, 28.0, 12.0, 12.0, 12.0])
 
Q_LIMIT = (-2 * pi, 2 * pi)
 
# Wrist camera mount, expressed in the tool0 frame (m). The OpenCV camera axes
# (x right, y down, z forward) coincide with the tool axes, so the hand-eye
# transform is a pure translation. scripts/build_ur3_model.py uses this value.
CAM_OFFSET_IN_TOOL = np.array([0.0, 0.05, 0.07])
 
 
def dh_transform(q, d, a, alpha):
    """Standard DH homogeneous transform for one joint."""
    cq, sq = np.cos(q), np.sin(q)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [cq, -sq * ca,  sq * sa, a * cq],
        [sq,  cq * ca, -cq * sa, a * sq],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])
 
 
def fkine(q):
    """Forward kinematics of the tool0 frame in the base frame (4x4)."""
    T = np.eye(4)
    for i in range(6):
        T = T @ dh_transform(q[i], D[i], A[i], ALPHA[i])
    return T
 
 
def fixed_transform(i):
    """Tz(d_i) Tx(a_i) Rx(alpha_i): the constant part of DH link i."""
    return dh_transform(0.0, D[i], A[i], ALPHA[i])
 
 
def T_T_C():
    """Hand-eye transform: camera (OpenCV convention) pose in the tool frame."""
    T = np.eye(4)
    T[:3, 3] = CAM_OFFSET_IN_TOOL
    return T
 
 
def make_robot(controlled="tool"):
    """Robotics Toolbox DH model of the UR3 (same pattern as Week 9, C1).
 
    controlled="tool"   -> jacobe() is the tool0 Jacobian
    controlled="camera" -> robot.tool = hand-eye, so fkine() is the camera pose and
                           jacobe() maps joint rates to a CAMERA-frame twist (needed for IBVS).
    """
    import roboticstoolbox as rtb
    from spatialmath import SE3
    links = [rtb.RevoluteDH(d=D[i], a=A[i], alpha=ALPHA[i], qlim=list(Q_LIMIT)) for i in range(6)]
    robot = rtb.DHRobot(links, name="UR3")
    if controlled == "camera":
        robot.tool = SE3(T_T_C())
    return robot
 