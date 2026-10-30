"""Check the MuJoCo UR3 against DH forward kinematics (Part A evidence).
 
Run from the repo root:  python -m scripts.check_fk
"""
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # make `ur3pick` importable when run as scripts/xxx.py
import numpy as np
import mujoco
from ur3pick.modelling import ur3_params as P
 
 
def main(n=50, seed=0):
    model = mujoco.MjModel.from_xml_path("models/scene.xml")
    data = mujoco.MjData(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool0")
    rng = np.random.default_rng(seed)
    pos_err, rot_err = [], []
    for _ in range(n):
        q = rng.uniform(-np.pi, np.pi, 6)
        data.qpos[:6] = q
        mujoco.mj_forward(model, data)
        T = P.fkine(q)
        R_mj = data.site_xmat[sid].reshape(3, 3)
        pos_err.append(np.linalg.norm(data.site_xpos[sid] - T[:3, 3]))
        rot_err.append(np.linalg.norm(R_mj - T[:3, :3]))
    print(f"{n} random configs: max position error {max(pos_err):.2e} m, "
          f"max rotation error {max(rot_err):.2e}")
    assert max(pos_err) < 1e-6 and max(rot_err) < 1e-6, "MuJoCo model does not match DH"
    print("PASS")
 
 
if __name__ == "__main__":
    main()