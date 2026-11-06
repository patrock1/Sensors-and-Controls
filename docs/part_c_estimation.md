# Part C: Estimation 

The pipeline runs two estimators, a joint kalman filter that reconstructs the six joint velocities, which are not measured (Part A shows the encoders alone to make the full state observable.) and an object filter which fuses the camera localisations into one estimate of the cube position. Both run every 10ms control tick and only ever see sensor data. 'tests/test_no_ground_truth.py' checks that nothing in 'estimation/' imports the simulator or the evaluation code 

## Joint Kalman filter

Model. 'estimation/joint_observer.py: JointKF' is a linear Kalman filter on the 12-state model from Part A, 'x = [q, qdot]', using the same coupled discrete model (Ad, Bd) as the controller ('modelling.coupled: true', full 6x6 mass matrix at the home pose). The prediction step uses the torque the controller applied in the last tick, "x- = Ad x + Bd u'. The update step uses six encoder angles y = q + v.

Measurement noise R. 'R = sigma_q^2 I' with 'sigma_q' = 0.0005 rad, the same value the simulator adds to the encoders (Part B). R is not a tuning knob. It is the sensor specification. 

Process noise Q Q stands for everything the linear model leaves out: the inertia change with pose (up to 44% on the base joint, Part A), Coriolis terms, and external pushes. It is modelled as an unknown torque 'w ~ N(0, sigma_tau^2)' on each joint. That torque becomes a white acceleration `w / M_i`, and the standard discretisation for white acceleration gives 
Qd_i = q_i * [ dt^3/3   dt^2/2 ]      q_i = (sigma_tau,i / M_i)^2 * dt
                 [ dt^2/2   dt     ]

(`modelling/state_space.py: process_noise`, `q_acc_from_torque_sigma`). Writing Q as a torque makes it physical: `sigma_tau` = [6, 6, 3, 0.9, 0.9, 0.9] N m is about 10% of each joints torque limit, largest on the shoulder and elbow where the inertia changes most

Initialisation. The state starts at the first encoder reading with zero velocity. The covariance starts at the steady state solution of the Riccati equation (`steady_state()`), so the filter is consistent with the first tick rather than shrinking from an arbitrary large P0.

Steady state. The steady state 1-sigma uncertainty is 0.48-0.50 mrad on the joint angles and 0.10-0.51 rad/s on the joint velocities (largest on the light wrist joints, where a given torque error gives the largest acceleration). `python -m scripts.design_report` prints these values.

Results over the 10 dev seeds

| Quantity | Value |
|---|---|
| Joint angle estimate error (RMS, all joints) | 0.48 mrad |
| Joint velocity estimate error (RMS, all joints) | 93 mrad/s |
| Mean joint NEES (12 states, consistent filter gives 12) | 6.6 (median over trials) |

The angle error equals the predicted 1-sigma, so the angle half of the state is consistent. A NEES of 6.6 instead of 12 means the filter is about 2x sonservative, meaning its velocity covariance is larger than the real velocity error. This is a deliberate consequence of sizing 'sigma_tau for the worst case. In a smooth nominal trial the unmodelled torque is small, so the filter covers more than it needs to. The trade-off below shows what happens when it is made smaller

Trade off: consistency against responsiveness

`configs/sweeps/observer_tuning.yaml` scales `sigma_tau` by one factor from x0.03 to x10 and changes nothing else. `observer_tuning_push.yaml` repeats the same levels with the 40 N, 1 s wrist push from Part E. Results (5 dev seeds per level, all levels 100% success)

| sigma_tau | Nominal: NEES | Nominal: velocity error | Nominal: torque u_rms | With 40 N push: NEES | With push: velocity error |
|---|---|---|---|---|---|
| x0.03 | 7.8 | 9 mrad/s | 0.5 N m | 2383 | 223 mrad/s |
| x0.1 | 6.3 | 23 | 0.6 | 110 | 123 |
| x0.33 | 6.7 | 59 | 1.1 | 11.0 | 84 |
| x1 (nominal) | 6.5 | 93 | 2.4 | 6.8 | 100 |
| x3 | 6.1 | 115 | 3.9 | 6.1 | 119 |
| x10 | 5.9 | 123 | 4.6 | 5.9 | 126 |

Figures: `docs/figures/tradeoff_observer_tuning.png` and `tradeoff_observer_tuning_push.png`.

Too small (x0.03-x0.1). In a nominal trial the filter looks excellent: it trusts the model, so encoder noise barely reaches the velocity estimate (9 mrad/s error) and the controller barely moves (0.5 N m). But it is responsive to nothing the model does not predict. When the arm is pushed it keeps following the model while the arm moves away, and the NEES jumps to 110-2400: the filter is badly overconfident exactly when it matters 
Too large (x3-x10). The filter trusts the encoders, so their noise passes straight into the velocity estimate (115-126 mrad/s). The LQR turns that noise into torque (3.9-4.6 N m). NEES falls further below 12 (conservative).
x0.33 is the only level that is close to consistent under the push (NEES 11.0, ideal 12) while also giving the smallest velocity error with the push. In nominal trials it halves the velocity error and the torque compared with x1. 

[TEAM: keep ONE of these two sentences, depending on what is in configs/default.yaml for the final runs]
*The final configuration uses x0.33 (`sigma_tau` = [2, 2, 1, 0.3, 0.3, 0.3] N m) because it is the most consistent setting under disturbance and the least noisy in nominal trials.*
*The final configuration keeps x1 as a margin for disturbances larger than the 40 N push tested. x0.33 is the better-tuned choice for the conditions we tested and is reported here as the recommended setting.*

Estimator speed

The estimaotors slowest closed loop mode decays 9.7 times faster than the controllers (`design_report.py`). The Week 8 rule of thumb is 2-5x for a pole-placement observer, because a faster observer amplifies measurement noise. That rule does not carry over directly to a Kalman filter: here the gain is the optimum for the started noise and it is high because the encoders are very precise (0.5 mrad). The cost of the fast observer is the velocity noise in the table above, and that is exactly what the `sigma_tau` sweep trades off. The seperation principle holds: the closed-loop eigenvalues of the combined controller and observer are the union of the two designs to 3.7e-15 (`modelling/analysis.py: separation_check`).

Object filter

Model `estimation/object_filter.py: ObjectFilter` estimates the cube position in the base frame. The cube is static, so the model is a random walk `p+ = p + w` with `w ~ N(0, q_obj dt I)` and `q_obj` = 1e-6 m^2/s. That allows about 1mm of drift per second. It is small because the cube does not move but not zero as the covariance would shrink without limit and the filter would stop listening to the camera if the cube were nudged. 

Measurements Each camera frame gives a position from `localise()` (Part B) with its own covariance R, built from the depth and pixel noise models and rotated into the base frame. The first valid measurement initialises the filter with `p_hat = p_meas`, `P = R`. Later measurements go through a standard Kalman update

Outlier gate. Before each update the filter computes the normalised innovation squared `NIS = nu' S^-1 nu`, where S = P + R. A measurement is rejected if NIS > 11.34, the 99% point of the chi-square distribution with 3 degrees of freedom. Late in the approach the camera is close enough that the cube starts to leave the image and the raw localisations degrade. The gate keeps those measurements out of the estimate.

Freezing. The estimate is frozen when the arm starts its descent (`GraspStateMachine.wants_object_updates`). Below the pre-grasp height the camera is too close to see the whole cube, so new measurements would only add error. The frozen value is also what the evaluation scores, because it is the estimate the arm commits to. 

Results over the 10 dev seeds

| Quantity | Value |
|---|---|
| Object error at the freeze point, median | 0.84 mm (xy 0.83 mm, z -0.26 mm) |
| Object error at the freeze point, 90th percentile | 1.3 mm |
| Object NEES at the freeze point (3 dof, consistent gives 3) | median 0.37, max 0.97 |
| Grasp capture radius (half the cube width) | 20 mm |

The estimate is about 25 times more accurate than the grasp needs. The object NEES shows the filter is conservative by about 8x meaning the R from `localise()` is larger than the actual measurement error. This is a safe direction, because an overconfident R would make the gate reject good measurements. It could be tightened by fitting `pixel_sigma` and the depth model to measured errors. Part E shows the estimate stays inside the capture radius until the hand eye calibration is off by more than 15mm. 

Summary

The joint velocities, which are not measured, are estimated to 93 mrad/s RMS, and the joint angles to 0.48 mrad, the encoder noise level
The joint filter is 2x conservative in nominal trials and consistent under a 40 N push at x0.33. The `sigma_tau` sweep quantifies the consistency/responsiveness trade-off and shows that a filter turned to look best nominally fails badly under disturbance. 
The object filter places the cube to 0.84 mm median with a chi-square gate against bad frames, and is conservative by design. 
All numbers come from `python -m scripts.design_report`, `python scripts/run_trials.py --seeds dev` and `python scripts/run_sweep.py observer_tuning` / `observer_tuning_push`.
