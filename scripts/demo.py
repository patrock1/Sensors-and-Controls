"""Run one pick and write a video: wrist camera with detection (left), overview camera (right).
 
    python scripts/demo.py --seed 3 --out results/demo.mp4
Also writes a PNG of the trial plots next to the video.
"""
import argparse
import pathlib
 
import _common as c
import numpy as np
import cv2
import imageio
import mujoco
 
from ur3pick.pipeline import load_cfg, run_trial
from ur3pick.evaluation.metrics import trial_metrics
from ur3pick.evaluation import plots
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(c.ROOT / "results" / "demo.mp4"))
    a = ap.parse_args()
 
    cfg = load_cfg()
    h, w = cfg["sim"]["image_height"], cfg["sim"]["image_width"]
    renderer = {}
    out_frames = []
 
    def cb(pkt, det, phase, scene):
        if "r" not in renderer:
            renderer["r"] = mujoco.Renderer(scene.model, h, w)      # sim-side rendering, video only
        left = pkt.rgb.copy()
        if det.found:
            u, v = det.centroid_px.astype(int)
            cv2.drawMarker(left, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(left, "wrist camera", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        renderer["r"].update_scene(scene.data, camera="overview")
        right = renderer["r"].render().copy()
        cv2.putText(right, phase, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        out_frames.append(np.hstack([left, right]))
 
    log = run_trial(a.seed, cfg, frame_cb=cb)
    m = trial_metrics(log, cfg)
    print(f"seed {a.seed}: {'success' if m['success'] else 'FAIL'} ({m['final_phase']})")
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out), out_frames, fps=int(round(1 / cfg["sim"]["dt_cam"])) * 2)    # 2x speed
    # a still of the moment of grasp, handy for the README
    i = int(len(out_frames) * 0.2)
    imageio.imwrite(str(out.with_name(out.stem + "_frame.png")), out_frames[i])
    plots.plot_trial(log, str(out.with_suffix(".png")))
    print("wrote", out)
 
 
if __name__ == "__main__":
    main()
