"""
test_frames.py

Frame and sensing chain against the MuJoCo scene (needs mujoco and an offscreen renderer)."""
import numpy as np
import pytest
 
mujoco = pytest.importorskip("mujoco")
from ur3pick.sensing import frames
from ur3pick.sensing.calibration import intrinsics_from_fovy
from ur3pick.sensing.detection import detect_object
from ur3pick.sensing.localisation import localise
from ur3pick.modelling import ur3_params as P
from tests.conftest import ROOT
 
 
@pytest.fixture(scope="module")
def world(cfg):
    model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "scene.xml"))
    data = mujoco.MjData(model)
    q = np.array(cfg["modelling"]["home_q"])
    data.qpos[:6] = q
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_free")
    a = model.jnt_qposadr[j]
    cube = np.array([0.27, 0.03, 0.02])
    data.qpos[a:a + 7] = [*cube, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    try:
        r = mujoco.Renderer(model, cfg["sim"]["image_height"], cfg["sim"]["image_width"])
    except Exception as e:                                   # no GL on this machine
        pytest.skip(f"no offscreen renderer: {e}")
    r.update_scene(data, camera="wrist_cam")
    rgb = r.render().copy()
    r.enable_depth_rendering()
    r.update_scene(data, camera="wrist_cam")
    depth = r.render().copy()
    return dict(model=model, data=data, q=q, cube=cube, rgb=rgb, depth=depth)
 
 
def test_camera_matches_hand_eye(world, cfg):
    """The camera in the MJCF sits where T_T_C says (MuJoCo world pose vs DH fkine @ T_T_C)."""
    m, d = world["model"], world["data"]
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    T_mj = np.eye(4)
    R_mj = d.cam_xmat[cid].reshape(3, 3)                    # MuJoCo camera axes: x right, y up, z backwards
    T_mj[:3, :3] = R_mj @ np.diag([1, -1, -1])             # convert to OpenCV axes
    T_mj[:3, 3] = d.cam_xpos[cid]
    assert np.allclose(T_mj, frames.T_B_C(world["q"]), atol=1e-9)
 
 
def test_projection_hits_the_red_blob(world, cfg):
    K = intrinsics_from_fovy(cfg["camera"]["fovy_deg"], cfg["sim"]["image_width"], cfg["sim"]["image_height"])
    det = detect_object(world["rgb"], cfg["detection"])
    assert det.found
    top = world["cube"] + np.array([0, 0, 0.02])           # centre of the visible top face
    uv, z = frames.project(K, frames.T_B_C(world["q"]), top)
    assert np.linalg.norm(uv - det.centroid_px) < 3.0       # px; side faces bias the centroid slightly
 
 
def test_localisation_error_against_truth(world, cfg):
    K = intrinsics_from_fovy(cfg["camera"]["fovy_deg"], cfg["sim"]["image_width"], cfg["sim"]["image_height"])
    det = detect_object(world["rgb"], cfg["detection"])
    meas = localise(det, world["depth"], K, world["q"], 0.0, {**cfg["localisation"], **cfg["noise"]})
    assert meas.valid
    err = np.linalg.norm(meas.p_B - world["cube"])
    assert err < 0.008, f"localisation error {1000 * err:.1f} mm"
    assert np.all(np.linalg.eigvalsh(meas.R) > 0)