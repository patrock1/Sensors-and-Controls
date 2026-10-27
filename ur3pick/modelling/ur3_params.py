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