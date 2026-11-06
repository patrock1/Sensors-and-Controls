import copy
import json
import pathlib
import numpy as np
import yaml
from scipy.linalg import block_diag

from ur3pick.interfaces import Command, Estimate
from ur3pick.modelling import ur3_params as P
from ur3pick.modelling.state_space import build_model, discretise, process_noise, q_acc_from_torque_sigma
from ur3pick.sensing.calibration import intrinsics_from_fovy
from ur3pick.sensing.detection import detect_object
from ur3pick.sensing.localisation import localise
from ur3pick.estimation.joint_observer import JointKF, steady_state
from ur3pick.estimation.object_filter import ObjectFilter
from ur3pick.control.lqr import JointLQR, bryson
from ur3pick.control.grasp import GraspStateMachine
from ur3pick.sim.scene import Scene
from ur3pick.sim.sensors import SensorSuite
from ur3pick.evaluation.ground_truth import read_truth

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / "configs" / "default.yaml"

#config

def _deep_merge(base: dict, new:dict) -> dict:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base

def load_cfg(overrides=None, path=DEFAULT_CFG) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key, val in (overrides or {}).items():
        *parents, leaf = key.split(".")
        node = cfg
        for p in parents:
            node = node.setdefault(p, {})
        if isinstance(val, dict) and isinstance(node.get(leaf), dict):
            _deep_merge(node[leaf], val)
        else:
            node[leaf] = copy.deepcopy(val)
    return cfg

def config_hash(cfg) -> str:
    import hashlib
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


# Build the estimator and controller from the config without ground truth

def build_stack(cfg, q0):
    dt = cfg["sim"]["dt_ctrl"]
    mod = cfg["modelling"]
    M = mod["joint_inertia"]                                        # diagonal: used to scale torque noise
    M_design = mod["inertia_matrix"] if mod.get("coupled", False) else M
    Ad, Bd, Cd, _ = discretise(*build_model(M_design, mod["damping"]), dt)

    sigma_q = cfg["noise"]["sigma_q"]
    Qd = process_noise(dt, q_acc_from_torque_sigma(M, cfg["observer"]["sigma_tau"], dt))
    Rn = np.eye(6) * sigma_q ** 2
    _, _, P0 = steady_state(Ad, Cd, Qd, Rn)
    obs = JointKF(Ad, Bd, Cd, Qd, Rn, np.r_[q0, np.zeros(6)], P0)

    objf = ObjectFilter(cfg["object_filter"]["q_obj"])

    L = cfg["lqr"]
    Qw = block_diag(bryson(L["q_err_max"]), bryson(L["qd_max"]))
    Rw = bryson(P.TORQUE_LIMIT)
    Qi = bryson(L["qi_max"]) if L["integral"] else None
    lqr = JointLQR(Ad, Bd, dt, Qw, Rw, P.TORQUE_LIMIT, integral=L["integral"], Qi=Qi)

    grasp = GraspStateMachine(cfg, dt, q0)
    return obs, objf, lqr, grasp


# one trial

def run_trial(seed: int, cfg:dict, frame_cb=None) -> dict:
    dt = cfg["sim"]["dt_ctrl"]
    W, H = cfg["sim"]["image_width"], cfg["sim"]["image_height"]

    fovy = cfg["camera"]["fovy_deg"] + cfg.get("calibration", {}).get("fovy_error_deg", 0.0)
    K = intrinsics_from_fovy(fovy, W, H)
    lcfg = {**cfg["localisation"], ** cfg["noise"],
            "hand_eye_error": cfg.get("calibration", {}).get("hand_eye_error")}   # estimator-side only, like fovy_error_deg above
    g = cfg["grasp"]
    t_max = g["search_timeout"] + 4 * g["phase_timeout"]        # hard stop in a normal trial

    scene = Scene(cfg)
    scene.reset(seed)
    sens = SensorSuite(scene, cfg, scene.sensor_rng)

    pkt = sens.read()
    obs, objf, lqr, grasp = build_stack(cfg, pkt.q_meas)

    u = np.zeros(6)
    log = {k: [] for k in ["t", "q_meas", "x_hat", "P_x", "p_hat", "P_p", "p_meas", "meas_valid", "u", "x_ref", "phase", "gripper", "gt_q", "gt_qdot", "gt_p_obj", "gt_p_tool"]}
    t_done = None

    try:
        while True:
            #estimation
            obs.predict(u)
            obs.update(pkt.q_meas)
            p_meas, valid = np.full(3, np.nan), False
            if pkt.rgb is not None:
                det = detect_object(pkt.rgb, cfg["detection"])
                meas = localise(det, pkt.depth, K, obs.x_hat[:6], pkt.cam_stamp, lcfg)
                p_meas, valid = meas.p_B, meas.valid
                if grasp.wants_object_updates:
                    objf.update(meas)
                if frame_cb is not None:
                    frame_cb(pkt, det, grasp.phase, scene)
            objf.predict(dt)
            est = Estimate(pkt.t, obs.x_hat.copy(), obs.P.copy(), objf.p_hat.copy(), objf.P.copy(), objf.seen)
            x_ref, grip, phase = grasp.step(est)
            u = lqr.command(est.x_hat, x_ref)

            truth = read_truth(scene)
            for k, v in [("t", pkt.t), ("q_meas", pkt.q_meas), ("x_hat", est.x_hat), ("P_x", est.P_x), ("p_hat", est.p_hat), ("P_p", est.P_p), ("p_meas", p_meas), ("meas_valid", valid), ("u", u), ("x_ref", x_ref), ("phase", phase), ("gripper", grip), ("gt_q", truth ["q"]), ("gt_qdot", truth["qdot"]), ("gt_p_obj", truth["p_B_O"]), ("gt_p_tool", truth["p_B_T"])]:
                log[k].append(v)

            # stop: hold_time after done/failed, or the third hard stop
            if grasp.finished and t_done is None:
                t_done = pkt.t
            if(t_done is not None and pkt.t - t_done >= g["hold_time"]) or pkt.t >= t_max:
                break
            scene.step(Command(u, grip, phase))
            pkt = sens.read()

    finally:
        for r in (sens.rgb_r, sens.depth_r): # free GL contexts fo the next trial renders correctly
            r.close()

    out = {k: np.asarray(v) for k, v in log.items()}
    out["phase"] = out["phase"].astype(str)
    cause = grasp.failure_cause if grasp.phase != "done" else "none"
    if pkt.t >= t_max and not grasp.finished:
        cause = "timeout"
    out["meta"] = json.dumps(dict(seed=int(seed), config_hash=config_hash(cfg), failure_cause=cause, final_phase=grasp.phase))
    return out