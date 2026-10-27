
"""
test_control.py
"""
import numpy as np
from ur3pick.modelling import state_space as ss, ur3_params as P
from ur3pick.control.lqr import JointLQR, bryson
from ur3pick.control.pbvs import pbvs_twist
from ur3pick.control.ibvs import image_jacobian, ibvs_twist
from ur3pick.control.servo import resolved_rate
from ur3pick.sensing.calibration import intrinsics_from_fovy
from ur3pick.sensing import frames
 
 
def _plant(cfg):
    M = np.array(cfg["modelling"]["joint_inertia"])
    A, B, C, D = ss.build_model(M, cfg["modelling"]["damping"])
    dt = cfg["sim"]["dt_ctrl"]
    Ad, Bd, Cd, _ = ss.discretise(A, B, C, D, dt)
    return M, Ad, Bd, dt
 
 
def _lqr(cfg, integral):
    M, Ad, Bd, dt = _plant(cfg)
    Qw = bryson(np.r_[cfg["lqr"]["q_err_max"], cfg["lqr"]["qd_max"]])
    Rw = bryson(P.TORQUE_LIMIT)
    Qi = bryson(cfg["lqr"]["qi_max"])
    return JointLQR(Ad, Bd, dt, Qw, Rw, P.TORQUE_LIMIT, integral=integral, Qi=Qi), Ad, Bd, dt
 
 
def _run(ctrl, Ad, Bd, x_ref, tau_dist, steps=1500):
    x = np.zeros(12)
    peak = 0.0
    for _ in range(steps):
        u = ctrl.command(x, x_ref)
        peak = max(peak, float(np.max(np.abs(u) / P.TORQUE_LIMIT)))
        x = Ad @ x + Bd @ (u + tau_dist)
    return x, peak
 
 
def test_integral_action_removes_gravity_offset(cfg):
    """Week 6 A.7: constant disturbance leaves an offset; the integral state cancels it."""
    x_ref = np.r_[0.3, np.zeros(11)]
    tau = np.array([3.0, 3.0, 1.0, 0.1, 0.1, 0.1])        # gravity-like constant torque
    plain, Ad, Bd, dt = _lqr(cfg, integral=False)
    integ, *_ = _lqr(cfg, integral=True)
    x_p, _ = _run(plain, Ad, Bd, x_ref, tau)
    x_i, peak = _run(integ, Ad, Bd, x_ref, tau, steps=4000)
    assert abs(x_p[0] - 0.3) > 1e-3
    assert abs(x_i[0] - 0.3) < 1e-3 and abs(x_i[2] - 0.0) < 1e-3
    assert peak <= 1.0 + 1e-9                               # torque limits respected
 
 
def test_pbvs_converges_on_ur3(cfg):
    """Week 9 Part C/D pattern: resolved-rate PBVS toward a pre-grasp pose above the cube."""
    robot = P.make_robot("tool")
    q = np.array(cfg["modelling"]["home_q"], float)
    p_obj = np.array([0.30, 0.04, 0.02])
    t_star = np.array([0.0, 0.0, cfg["servo"]["pre_grasp_height"]])
    lam, dt, peak = cfg["servo"]["lam"], 0.01, 0.0
    for _ in range(1500):
        T = frames.T_B_T(q)
        t_T = frames.transform_point(frames.inv(T), p_obj)
        nu = pbvs_twist(t_T, t_star, lam)
        qd, _ = resolved_rate(robot, q, nu, qd_max=cfg["servo"]["qd_limit"])
        peak = max(peak, np.abs(qd).max())
        q = q + qd * dt
        if np.linalg.norm(t_T - t_star) < 5e-4:
            break
    T = frames.T_B_T(q)
    err = np.linalg.norm(frames.transform_point(frames.inv(T), p_obj) - t_star)
    assert err < 1e-3 and peak <= cfg["servo"]["qd_limit"] + 1e-9
 
 
def test_image_jacobian_matches_finite_difference_and_toolbox(cfg):
    K = intrinsics_from_fovy(cfg["camera"]["fovy_deg"], cfg["sim"]["image_width"], cfg["sim"]["image_height"])
    pc = np.array([0.03, -0.02, 0.25])
    uvw = K @ pc
    uv = (uvw[:2] / uvw[2]).reshape(2, 1)
    nu = np.array([0.1, 0.05, -0.05, 0.02, -0.03, 0.05])
    h = 1e-6
    from spatialmath import SE3
    Tm = SE3.Delta(nu * h).inv().A                          # camera moves by nu*h; point seen in new frame
    pc2 = Tm[:3, :3] @ pc + Tm[:3, 3]
    uvw2 = K @ pc2
    fd = (uvw2[:2] / uvw2[2] - uv.ravel()) / h
    J = image_jacobian(uv, pc[2], K)
    assert np.allclose(J @ nu, fd, rtol=1e-3, atol=1e-2)
    try:
        from machinevisiontoolbox import CentralCamera
    except Exception:
        return
    cam = CentralCamera(f=K[0, 0], pp=(K[0, 2], K[1, 2]), imagesize=(320, 240), rho=1)
    assert np.allclose(J, cam.visjac_p(uv, depth=pc[2]), atol=1e-6)
 
 
def test_ibvs_converges_on_ur3(cfg):
    """Week 9 C2: eye-in-hand IBVS on four corners of a 10 cm square lying on the table."""
    robot = P.make_robot("camera")
    K = intrinsics_from_fovy(cfg["camera"]["fovy_deg"], cfg["sim"]["image_width"], cfg["sim"]["image_height"])
    q = np.array(cfg["modelling"]["home_q"], float)
    s = 0.05
    corners = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]]).T      # TL, TR, BR, BL on the table
    R_tool = frames.T_B_T(q)[:3, :3]
    yaw = np.arctan2(R_tool[1, 0], R_tool[0, 0])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    centre = np.array([0.30, 0.02, 0.0])
    P_w = Rz @ corners + centre[:, None]
    pd = np.array([[K[0, 2]], [K[1, 2]]]) + 55 * np.array([[-1, 1, 1, -1], [-1, -1, 1, 1]])
    dt, lam, peak = 0.01, cfg["servo"]["lam"], 0.0
    for k in range(4000):
        Tc = frames.T_B_C(q)
        pc = frames.inv(Tc)[:3, :3] @ P_w + frames.inv(Tc)[:3, 3:4]
        uvw = K @ pc
        uv = uvw[:2] / uvw[2]
        nu = ibvs_twist(uv, pd, pc[2], K, lam)
        qd, _ = resolved_rate(robot, q, nu, qd_max=cfg["servo"]["qd_limit"])
        peak = max(peak, np.abs(qd).max())
        q = q + qd * dt
        if np.linalg.norm(pd - uv) < 0.5:
            break
    assert np.linalg.norm(pd - uv) < 1.0, f"final image error {np.linalg.norm(pd - uv):.2f} px"