"""Localisation accuracy over many seeds: first camera frame from home pose, error of p_B against the true cube centre.
    python scripts/check_localise.py --n 20
Before the top-face correction: mean z +20.0 mm, median lateral 0.7 mm (20 seeds).
After  (localisation.surface_to_centre = 0.02): mean z 0.0 mm, median lateral 0.7 mm.
"""
import argparse

import _common as c
import numpy as np
import yaml

from ur3pick.sim.scene import Scene
from ur3pick.sim.sensors import SensorSuite
from ur3pick.sensing.detection import detect_object
from ur3pick.sensing.localisation import localise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    with open(c.ROOT / "configs" / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    W, H = cfg["sim"]["image_width"], cfg["sim"]["image_height"]
    f = 0.5 * H / np.tan(np.deg2rad(cfg["camera"]["fovy_deg"]) / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    lcfg = {**cfg["localisation"], **cfg["noise"]}
    errs = []
    for seed in range(a.n):
        s = Scene(cfg); s.reset(seed); sens = SensorSuite(s, cfg, s.sensor_rng)
        pkt = sens.read()
        det = detect_object(pkt.rgb, cfg["detection"])
        meas = localise(det, pkt.depth, K, pkt.q_meas, pkt.t, lcfg)   # noisy encoders stand in for q_hat
        if not meas.valid:
            print(f"seed {seed:2d}  not localised"); continue
        e = (meas.p_B - s.object_pose[0]) * 1000                       # truth used for evaluation only
        errs.append(e)
        print(f"seed {seed:2d}  err x {e[0]:6.1f}  y {e[1]:6.1f}  z {e[2]:6.1f} mm")
    E = np.array(errs)
    print(f"\nmean   x {E[:,0].mean():6.1f}  y {E[:,1].mean():6.1f}  z {E[:,2].mean():6.1f} mm")
    print(f"median lateral {np.median(np.hypot(E[:,0], E[:,1])):.1f} mm   ({len(errs)}/{a.n} localised)")


if __name__ == "__main__":
    main()
