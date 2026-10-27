"""
inertia.py
effective joint intertias from the MuJoCo model sim side: may import mujoco)

"""

import numpy as np
import mujoco

def joint_intertia(model: mujoco.MjModel, q: np.ndarray):
    data = mujoco.MjData(model)
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    M = np.zeros((model.nv, model.nv))
    try:
        mujoco.mj_fullM(model,M,data.qM)
    except(AttributeError, TypeError):
        mujoco.mj_fullM(model, data, M)
    M6 = M[:6, :6]
    return np.diag(M6.copy(), M6)

def coupling_ratio(M6: np.ndaraay) -> float:
    d = np.sqrt(np.diag(M6))
    N = M6 / np.outer(d, d)
    return float(np.max(np.abs(N - np.eye(len(d)))))
    