"""Prints the numbers for the modelling, estimation and control sections of the README.
 
Run from the repo root:  python -m scripts.design_report
"""
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # make `ur3pick` importable when run as scripts/xxx.py
import numpy as np
import yaml
import mujoco
from ur3pick.modelling import state_space as ss, analysis as an, ur3_params as P
from ur3pick.estimation.joint_observer import steady_state
from ur3pick.control.lqr import bryson, design_dlqr
from ur3pick.sim.inertia import joint_inertia, coupling_ratio
 
 
def main():
    cfg = yaml.safe_load(open("configs/default.yaml"))
    dt = cfg["sim"]["dt_ctrl"]
    q = np.array(cfg["modelling"]["home_q"])
    model = mujoco.MjModel.from_xml_path("models/scene.xml")
    M_diag, M6 = joint_inertia(model, q)
    print("diag(M) from MuJoCo        :", np.round(M_diag, 4), "(config:", cfg["modelling"]["joint_inertia"], ")")
    print(f"coupling neglected         : {coupling_ratio(M6):.2f} (0 = decoupled)  <- state this assumption")
 
    M = np.array(cfg["modelling"]["joint_inertia"])
    A, B, C, D = ss.build_model(M, cfg["modelling"]["damping"])
    print("\nPart A, per joint [q, qdot] block (ctrb / obsv):")
    for r in an.per_joint_reports(A, B, C):
        print(f"  joint {r['joint']}: rank {r['ctrb']['rank']}/2, min sv {r['ctrb']['min_sv']:.3g} | "
              f"rank {r['obsv']['rank']}/2, min sv {r['obsv']['min_sv']:.3g}")
    print("  full 12-state PBH controllable:", an.pbh_controllable(A, B), " observable:", an.pbh_observable(A, C))
    sets = {"encoders (q)": C, "tachometers (qdot)": np.hstack([np.zeros((6, 6)), np.eye(6)])}
    print("  sensor sets:", {k: v["observable"] for k, v in an.sensor_set_report(A, sets).items()})
 
    Ad, Bd, Cd, Dd = ss.discretise(A, B, C, D, dt)
    Qw = bryson(np.r_[cfg["lqr"]["q_err_max"], cfg["lqr"]["qd_max"]])
    Rw = bryson(P.TORQUE_LIMIT)
    Kr, _, Ec = design_dlqr(Ad, Bd, Qw, Rw)
    qa = ss.q_acc_from_torque_sigma(M, cfg["observer"]["sigma_tau"], dt)
    Qd, Rn = ss.process_noise(dt, qa), np.eye(6) * cfg["noise"]["sigma_q"] ** 2
    _, Kf, Ppost = steady_state(Ad, Cd, Qd, Rn)
    sig = np.sqrt(np.diag(Ppost))
    print("\nPart C, steady-state 1-sigma error  q [rad]   :", np.round(sig[:6], 5))
    print("                                     qdot [rad/s]:", np.round(sig[6:], 4))
    print("Part D, |eig(Ad - Bd Kr)| max:", np.abs(Ec).max().round(4))
    print("Separation check (should be ~0):", f"{an.separation_check(Ad, Bd, Cd, Kr, Kf):.1e}")
    print("Estimator/controller speed ratio (Week 8 rule 2 to 5):", np.round(an.speed_ratio(Ad, Bd, Cd, Kr, Kf, dt), 2))
 
 
if __name__ == "__main__":
    main()