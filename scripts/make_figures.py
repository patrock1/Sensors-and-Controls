"""Regenerate every figure used in the README / report into docs/figures/.

    python scripts/make_figures.py
Run scripts/run_sweep.py first if you also want the sweep figures refreshed (they are written there).

frames.png and pipeline.png are drawn from the code itself (frames.py transforms, the real scene
render, the module layout), so they stay correct if the geometry or the pipeline changes.
This is a script, not pipeline code, so reading simulator state here is allowed.
"""
import _common as c
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import mujoco

from ur3pick.pipeline import load_cfg, run_trial
from ur3pick.evaluation import plots
from ur3pick.sim.scene import Scene
from ur3pick.sensing import frames
from ur3pick.sensing.calibration import intrinsics_from_fovy
from ur3pick.sensing.detection import detect_object

AXIS_LEN = 0.06                       # m, length of each drawn frame axis
AXIS_COLOURS = ("#d62728", "#2ca02c", "#1f77b4")   # x red, y green, z blue (the usual convention)


def _mujoco_cam_projector(model, data, cam_name, W, H):
    """Pixel projection for a MuJoCo camera (looks along its own -z, y up)."""
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    pos, R = data.cam_xpos[cid].copy(), data.cam_xmat[cid].reshape(3, 3).copy()
    f = (H / 2) / np.tan(np.deg2rad(model.cam_fovy[cid]) / 2)

    def project(p_B):
        x, y, z = R.T @ (np.asarray(p_B) - pos)
        return np.array([W / 2 + f * x / -z, H / 2 - f * y / -z])
    return project


def _draw_triad(ax, project, T, label, offset=(6, -6)):
    o = project(T[:3, 3])
    for k in range(3):
        e = project(T[:3, 3] + AXIS_LEN * T[:3, k])
        ax.annotate("", xy=e, xytext=o, arrowprops=dict(arrowstyle="-|>", color=AXIS_COLOURS[k], lw=2))
    ax.text(o[0] + offset[0], o[1] + offset[1], label, fontsize=12, weight="bold",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5))


def frames_figure(path, seed=0):
    """Left: overview render with frames B, T, C, O drawn from the pipeline's own transforms.
    Right: what the wrist camera sees, with the detected centroid and the projected cube centre."""
    cfg = load_cfg()
    sc = Scene(cfg); sc.reset(seed)
    m, d = sc.model, sc.data
    q = d.qpos[:6].copy()
    p_O, yaw = sc.object_pose

    T_B = np.eye(4)
    T_BT = frames.T_B_T(q)
    T_BC = frames.T_B_C(q)
    T_BO = np.eye(4); T_BO[:3, 3] = p_O
    T_BO[:3, :3] = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

    W, H = 640, 480
    r = mujoco.Renderer(m, H, W)
    r.update_scene(d, camera="overview")
    over = r.render().copy(); r.close()
    proj = _mujoco_cam_projector(m, d, "overview", W, H)

    Wc, Hc = cfg["sim"]["image_width"], cfg["sim"]["image_height"]
    rc = mujoco.Renderer(m, Hc, Wc)
    rc.update_scene(d, camera="wrist_cam")
    wrist = rc.render().copy(); rc.close()
    K = intrinsics_from_fovy(cfg["camera"]["fovy_deg"], Wc, Hc)
    det = detect_object(wrist, cfg["detection"])
    uv_centre, _ = frames.project(K, T_BC, p_O)
    uv_top, _ = frames.project(K, T_BC, p_O + np.array([0, 0, cfg["object"]["size"] / 2]))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
    a1.imshow(over); a1.set_axis_off()
    _draw_triad(a1, proj, T_B, "B  base / world")
    _draw_triad(a1, proj, T_BT, "T  tool0", offset=(-70, -14))
    _draw_triad(a1, proj, T_BC, "C  camera", offset=(10, 18))
    _draw_triad(a1, proj, T_BO, "O  cube", offset=(10, 22))
    a1.set_title("Frames at the home pose (seed %d). x red, y green, z blue, %.0f mm axes" % (seed, AXIS_LEN * 1000))
    chain = (r"$p_B = T_{BT}(\hat q)\,T_{TC}\,p_C$" "\n"
             r"$T_{BT}$: DH forward kinematics (ur3_params.py)" "\n"
             r"$T_{TC}$: hand-eye, translation [0, 0.05, 0.07] m in T" "\n"
             r"$p_C = Z\,K^{-1}[u, v, 1]^T$, then $-$%d mm along $z_B$ (top face to centre)"
             % round(cfg["localisation"]["surface_to_centre"] * 1000))
    a1.text(0.01, 0.02, chain, transform=a1.transAxes, fontsize=10, va="bottom",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.6"))

    a2.imshow(wrist); a2.set_axis_off()
    if det.found:
        a2.plot(*det.centroid_px, "x", color="yellow", ms=12, mew=2.5, label="detected centroid (mask)")
    a2.plot(*uv_centre, "+", color="cyan", ms=14, mew=2.5, label="cube centre projected")
    a2.plot(*uv_top, "o", mfc="none", color="white", ms=10, mew=2, label="top-face centre projected")
    a2.annotate("", xy=(55, 12), xytext=(10, 12), arrowprops=dict(arrowstyle="-|>", color=AXIS_COLOURS[0], lw=2))
    a2.annotate("", xy=(10, 57), xytext=(10, 12), arrowprops=dict(arrowstyle="-|>", color=AXIS_COLOURS[1], lw=2))
    a2.text(58, 16, "u = x_C", color="white", fontsize=10)
    a2.text(14, 66, "v = y_C", color="white", fontsize=10)
    a2.legend(loc="lower right", fontsize=9, framealpha=0.85)
    a2.set_title("Wrist camera C (OpenCV: x right, y down, z forward)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def pipeline_figure(path, cfg=None):
    """Block diagram of one control tick. Solid arrows: data the pipeline uses.
    Dashed: ground truth, which only goes to the log and evaluation."""
    cfg = cfg or load_cfg()
    hz = lambda dt: f"{1 / dt:.0f} Hz"
    ctrl, cam = hz(cfg["sim"]["dt_ctrl"]), hz(cfg["sim"]["dt_cam"])
    # three rows: camera branch (top), encoder branch (middle), control + evaluation (bottom)
    X = dict(c0=0.01, c1=0.205, c2=0.40, c3=0.595, c4=0.79)
    Y = dict(top=0.70, mid=0.40, bot=0.08)
    boxes = {   # name: (x, y, text, colour)
        "sim":   (X["c0"], Y["mid"], "MuJoCo scene\nUR3 + randomised cube\nsim/scene.py", "#e8e8e8"),
        "cam":   (X["c1"], Y["top"], "Wrist RGB-D camera\n%s, 320x240\nsim/sensors.py" % cam, "#e8e8e8"),
        "enc":   (X["c1"], Y["mid"], "Joint encoders\n%s, q only\nsim/sensors.py" % ctrl, "#e8e8e8"),
        "det":   (X["c2"], Y["top"], "Detection\nHSV threshold\nsensing/detection.py", "#cfe2f3"),
        "loc":   (X["c3"], Y["top"], "Localisation\ndepth + K + T_B_C(q̂)\nsensing/localisation.py", "#cfe2f3"),
        "objf":  (X["c4"], Y["top"], "Object filter\nstatic-point KF, χ² gate\nestimation/object_filter.py", "#d9ead3"),
        "obs":   (X["c2"], Y["mid"], "Joint observer\nKF on [q, q̇]\nestimation/joint_observer.py", "#d9ead3"),
        "grasp": (X["c4"], Y["mid"], "PBVS + grasp\nstate machine\ncontrol/grasp.py", "#fff2cc"),
        "lqr":   (X["c3"], Y["bot"], "LQR + integral action\non the estimated state\ncontrol/lqr.py", "#fff2cc"),
        "eval":  (X["c0"], Y["bot"], "Evaluation only\nevaluation/ground_truth.py\nmetrics.py, plots.py", "#f4cccc"),
    }
    w, h = 0.16, 0.17
    fig, ax = plt.subplots(figsize=(14, 6.4)); ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for x, y, txt, col in boxes.values():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008", fc=col, ec="0.3"))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9)

    def pt(n, side, f=0.5):
        """Point on a box edge: side l/r/t/b, f = fraction along that edge (0..1)."""
        x, y = boxes[n][:2]
        return {"l": (x, y + f * h), "r": (x + w, y + f * h), "t": (x + f * w, y + h), "b": (x + f * w, y)}[side]

    def arrow(p0, p1, label="", style="-", col="0.15", lab_off=(0, 0.02), ha="center"):
        ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6, ls=style))
        if label:
            ax.text((p0[0] + p1[0]) / 2 + lab_off[0], (p0[1] + p1[1]) / 2 + lab_off[1], label,
                    ha=ha, fontsize=8.5, color=col, bbox=dict(facecolor="white", edgecolor="none", pad=0.5))

    arrow(pt("sim", "t", 0.7), pt("cam", "l", 0.4), "render", lab_off=(-0.02, 0.02))
    arrow(pt("sim", "r"), pt("enc", "l"))
    arrow(pt("cam", "r", 0.65), pt("det", "l", 0.65), "RGB")
    # depth goes over the top of the detection box: up, across, down into localisation
    x0, x1, yt = pt("cam", "t", 0.8)[0], pt("loc", "t", 0.3)[0], pt("cam", "t")[1] + 0.06
    ax.plot([x0, x0, x1], [pt("cam", "t")[1], yt, yt], color="0.15", lw=1.6)
    ax.annotate("", xy=pt("loc", "t", 0.3), xytext=(x1, yt), arrowprops=dict(arrowstyle="-|>", color="0.15", lw=1.6))
    ax.text((x0 + x1) / 2, yt + 0.012, "depth", ha="center", fontsize=8.5)
    arrow(pt("det", "r"), pt("loc", "l"), "centroid")
    arrow(pt("loc", "r"), pt("objf", "l"), "p_B, R")
    arrow(pt("enc", "r"), pt("obs", "l"), "q_meas")
    arrow(pt("obs", "t", 0.8), pt("loc", "b", 0.2), "q̂", lab_off=(0.012, 0), ha="left")
    arrow(pt("objf", "b"), pt("grasp", "t"), "p̂_B, P", lab_off=(0.012, 0), ha="left")
    arrow(pt("obs", "r"), pt("grasp", "l"), "x̂ (q̂ for PBVS)")
    arrow(pt("grasp", "b"), pt("lqr", "r", 0.65), "x_ref, phase", lab_off=(0.03, 0.0), ha="left")
    arrow(pt("obs", "b", 0.75), pt("lqr", "t", 0.3), "x̂", lab_off=(0.015, 0), ha="left")
    arrow(pt("lqr", "t", 0.7), pt("obs", "b", 0.95), "u (predict step)", col="0.5", lab_off=(0.06, -0.01), ha="left")
    arrow(pt("lqr", "l"), pt("sim", "b", 0.85), "torque u", lab_off=(0.02, -0.04))
    arrow(pt("sim", "b", 0.3), pt("eval", "t", 0.3), "ground truth\n(log only)", style="--", col="#cc0000",
          lab_off=(-0.005, 0), ha="right")
    ax.set_title(f"One control tick ({ctrl}). Only estimates reach the controller; ground truth goes to the log and evaluation only",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main(seed=0):
    out = c.ROOT / "docs" / "figures"; out.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg()
    plots.plot_trial(run_trial(seed, cfg), str(out / "nominal_trial.png"))
    frames_figure(str(out / "frames.png"), seed)
    pipeline_figure(str(out / "pipeline.png"), cfg)
    for name in ("nominal_trial.png", "frames.png", "pipeline.png"):
        print("wrote", out / name)


if __name__ == "__main__":
    main()
