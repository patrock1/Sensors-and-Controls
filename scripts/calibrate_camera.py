"""Part B camera calibration from a known target.

The cube is placed at known poses (the calibration target; ground truth is allowed here because this
is an offline calibration step, not the running pipeline). For each pose the red blob centroid is
detected and the pinhole model u = fx x/z + cx, v = fy y/z + cy is fitted by linear least squares.
The fitted K is compared with the K implied by the MJCF fovy, and the reprojection error is reported.

    python scripts/calibrate_camera.py
"""
import _common as c
import numpy as np
import mujoco

from ur3pick.pipeline import load_cfg
from ur3pick.sim.scene import Scene
from ur3pick.sim.sensors import SensorSuite
from ur3pick.sensing.detection import detect_object
from ur3pick.sensing.calibration import intrinsics_from_fovy
from ur3pick.sensing import frames
from ur3pick.interfaces import Command


def main(n=40):
    cfg = load_cfg(overrides={"noise.dropout_prob": 0.0, "noise.sigma_q": 0.0})
    scene = Scene(cfg)
    rows, pts = [], []
    for s in range(n):
        scene.reset(1000 + s)
        mujoco.mj_forward(scene.model, scene.data)
        sensors = SensorSuite(scene, cfg, scene.rng)
        pkt = sensors.read(); sensors.rgb_r.close(); sensors.depth_r.close()
        det = detect_object(pkt.rgb, cfg["detection"])
        if not det.found:
            continue
        top = scene.data.xpos[mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "target")] + np.array([0, 0, cfg["object"]["size"] / 2])
        p_C = frames.transform_point(frames.inv(frames.T_B_C(scene.data.qpos[:6])), top)
        pts.append((p_C, det.centroid_px))
    X = np.array([p for p, _ in pts]); uv = np.array([q for _, q in pts])
    ax = np.c_[X[:, 0] / X[:, 2], np.ones(len(X))]; ay = np.c_[X[:, 1] / X[:, 2], np.ones(len(X))]
    fx, cx = np.linalg.lstsq(ax, uv[:, 0], rcond=None)[0]
    fy, cy = np.linalg.lstsq(ay, uv[:, 1], rcond=None)[0]
    K = intrinsics_from_fovy(cfg["camera"]["fovy_deg"], cfg["sim"]["image_width"], cfg["sim"]["image_height"])
    pred = np.c_[fx * X[:, 0] / X[:, 2] + cx, fy * X[:, 1] / X[:, 2] + cy]
    rms = float(np.sqrt(np.mean(np.sum((pred - uv) ** 2, axis=1))))
    print(f"{len(pts)} views")
    print(f"fitted   fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    print(f"from MJCF fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    print(f"reprojection RMS {rms:.2f} px")


if __name__ == "__main__":
    main()
