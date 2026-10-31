"""
ground_truth.py
The ONLY module in the pipeline that reads true simulator state (interfaces.md section 3.6).

Why it exists: the brief fails any submission whose controller uses ground truth. Putting every
truth read in one file makes that rule visible in the code. pipeline.py passes the dict returned
here straight into the log and nowhere else, and tests/test_no_ground_truth.py stops sensing,
estimation, control and modelling from importing this package.
"""

import numpy as np
import mujoco


def _ids(model):
    """Look up ids by NAME, not by index, so the code survives changes to the XML ordering."""
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool0")
    if body < 0 or site < 0:
        raise KeyError("scene.xml must define body 'target' and site 'tool0'")
    return body, site


def read_truth(scene) -> dict:
    """True state at the current sim time.

    Keys
    ----
    q, qdot : (6,)  true joint angles and rates (the arm is the first 6 dofs in the model)
    p_B_O   : (3,)  cube CENTRE in the base frame. Base frame = MuJoCo world frame (interfaces.md 1).
    p_B_T   : (3,)  tool0 origin in the base frame. Needed for the 'cube held by the tool' check.
    object_lifted : bool, cube centre above the success height (falls back to 0.05 m if the
                    scene has no cfg attached)
    """
    m, d = scene.model, scene.data
    body, site = _ids(m)
    p_obj = d.xpos[body].copy()
    h = 0.05
    cfg = getattr(scene, "cfg", None)
    if cfg is not None:
        h = cfg.get("grasp", {}).get("success_height", h)
    return dict(
        q=d.qpos[:6].copy(),
        qdot=d.qvel[:6].copy(),
        p_B_O=p_obj,
        p_B_T=d.site_xpos[site].copy(),
        object_lifted=bool(p_obj[2] >= h),
    )
