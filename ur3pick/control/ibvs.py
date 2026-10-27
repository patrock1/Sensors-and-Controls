"""
ibvs.py
Image based visual servoing
Part A to C
"""

import numpy as np

def image_jacobian(uv, Z, K):
    uv = np.atleast_2d(np.asarray(uv, float))
    if uv.shape[0] != 2:
        uv = uv.T
    N = uv.shape[1]
    Z = np.broadcast_to(np.asarray(Z, float), (N,))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    J = np.zeros((2 * N, 6))
    for i in range(N):
        x, y = (uv[0, i] - cx) / fx, (uv[1, i] - cy) / fy
        z = Z[i]
        J[2 * i] = fx * np.array([-1 / z, 0, x / z, x * y, -(1 + x * x), y])
        J[2 * i + 1] = fy * np.array([0, -1 / z, y / z, 1 + y * y, -x * y, -x])
    return J

def ibvs_twist(uv, uv_star, Z, K, lam):
    J = image_jacobian(uv, Z, K)
    e = (np.asarray(uv_star, float) - np.asarray(uv, float)).flatten(order="F")
    return lam * np.linalg.pinv(J) @ e

