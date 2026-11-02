"""Quick check of scene + sensors: reset, zero torque for 1 s, count frames, detect cube, save wrist image.
    python scripts/check_sim.py --seed 0
"""
import argparse

import _common as c
import numpy as np
import cv2
import yaml

from ur3pick.interfaces import Command
from ur3pick.sim.scene import Scene
from ur3pick.sim.sensors import SensorSuite
from ur3pick.sensing.detection import detect_object


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    with open(c.ROOT / "configs" / "default.yaml") as f:
        cfg = yaml.safe_load(f)

    scene = Scene(cfg)
    scene.reset(a.seed)
    sens = SensorSuite(scene, cfg, scene.sensor_rng)
    print("cube", np.round(scene.object_pose[0], 3), "yaw", round(scene.object_pose[1], 3))

    zero = Command(np.zeros(6), 0.0, "test")
    frames, last = 0, None
    for _ in range(100):                                  # 1 s at dt_ctrl
        pkt = sens.read()
        if pkt.rgb is not None:
            frames, last = frames + 1, pkt
        scene.step(zero)

    drift = np.abs(scene.data.qpos[:6] - cfg["modelling"]["home_q"]).max()
    print(f"t={scene.t:.2f} s  frames={frames} (expect ~20)  max drift from home={drift:.4f} rad")
    det = detect_object(last.rgb, cfg["detection"])
    print("detected", det.found, "area", det.area_px, "px  centroid", np.round(det.centroid_px, 1))
    if det.found:
        print(f"median depth on cube {np.median(last.depth[det.mask]):.3f} m")
    out = c.ROOT / "results" / "check_sim_wrist.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(last.rgb, cv2.COLOR_RGB2BGR))
    print("wrote", out)


if __name__ == "__main__":
    main()
