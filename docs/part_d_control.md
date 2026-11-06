# Part D: Control

Control has two layers. An outer loop turns the object estimate into a joint-space reference: a grasp state machine with position based visual servoing. An inner loop, a discrete LQR with integral action, makes the joints track that reference using torque. Both layers only see the estimates from Part C, never ground truth (`tests/test_no_ground_truth.py`).


    p_hat (object filter) -> grasp state machine + PBVS -> qdot_ref -> integrate -> x_ref
    x_hat (joint KF)      -> LQR + integral                              -> u (6 torques)

## Inner loop: LQR with integral action

DESIGN MODEL. The LQR is designed on the coupled discrete model from Part A: x = [q, qdot] with the full 6x6 mass matrix at the home pose, discretised with a zero-order hold at 100 Hz (`control/lqr.py: JointLQR`).

INTEGRAL ACTION. The state is augmented with the integral of the joint angle error, `xi+ = xi - dt (q - q_ref)`. This removes any constant offset that the model does not predict: residual gravity when the simulators gravity compensation is partial, or a steady inertia error. Part E tests this down to 50% gravity compensation, with 100% success. To prevent windup, the integrator is frozen on any tick where a torque saturates. 

**Weights (Bryson's rule).** Each weight is `1 / (largest acceptable value)^2`:

| Term | Largest acceptable value | Reason |
|---|---|---|
| joint angle error | 0.05 rad (2.9 deg) per joint | at the 0.3-0.4 m reach of the arm this is about 15-20 mm at the tool, the grasp capture radius |
| joint velocity error | 0.5 rad/s | half the 1 rad/s reference rate limit |
| integral of angle error | 0.05 rad s | a 0.05 rad error held for 1 s |
| torque | the joint's torque limit, 56 / 56 / 28 / 12 / 12 / 12 N m | the motors cannot give more |

The torque is clipped to the same limits after the LQR, matching the MJCF motors. The closed loop is stable with a largest discrete eigenvalue magnitude of 0.905, a time constant of about 0.1 s (`design_report.py`).

THE LAW is `u = -K (x_hat - x_ref) - Ki xi`. Because the estimator and the controller are designed separately, the separation principle holds for the combined system (Part C).

## Outer loop: grasp state machine and PBVS

`control/grasp.py: GraspStateMachine` moves through these phases:

| Phase | What happens | Exit condition (configs/default.yaml `grasp:`) |
|---|---|---|
| search | hold the home pose until the camera sees the cube | object filter has an estimate; fail if nothing within 2 s |
| approach | PBVS to put the cube 0.10 m ahead of the tool along the tool z axis | tool-frame error < 12 mm |
| descend | PBVS to 0.03 m ahead, using the estimate **frozen** at the start of this phase | error < 8 mm |
| grasp | hold still for 0.3 s, then close the gripper | settle time elapsed |
| lift | move the tool 0.15 m straight up in the base frame | error < 10 mm |
| done / failed | hold | trial ends 1 s later |

Every motion phase also has an 8 s timeout, which ends the trial as failed with cause "timeout". 

PBVS. The cube estimate is moved into the tool frame, `p_T = inv(T_B_T(q_hat)) p_hat`, and the commanded tool twist is `v = lambda (p_T - p_T*)` with `lambda` = 1.0 s^-1. That gives the error a first-order decay with a 1 s time constant (`control/pbvs.py`). The twist becomes joint rates through the pseudo-inverse of the tool Jacobian, `qdot = pinv(J(q_hat)) v` (`control/servo.py: resolved_rate`). If any joint would exceed 1 rad/s, all rates are scaled down together, so the tool keeps moving in the same direction. The rates are integrated into the reference `q_ref` (`ReferenceIntegrator`). The integrator integrates the *command*, not the noisy estimate, so encoder noise does not leak into the reference.

PBVS rather than IBVS was chosen because the depth camera gives a full 3D position for the cube and the grasp is defined in 3D (a point 3cm ahead of the tool). The camera also cannot see the cube in the last part of the descent, so an image-space law would lose its feedback just before the grasp. 

## Results over the 10 dev seeds

| Quantity | Value |
|---|---|
| Success | 10 / 10 (95 % Wilson interval 72-100 %) |
| Time to close the gripper, median | 5.1 s |
| Trial length | 9.0-9.2 s |
| Joint tracking error `q_ref - q_true`, RMS over all joints | 1.79 mrad median |
| Worst single-joint tracking error, 90th percentile | 18 mrad |
| RMS torque | 2.4 N m median |

Tracking error per joint (median RMS, mrad): q1 0.33, q2 0.78, q3 0.93, **q4 4.0**, q5 0.23, q6 1.27. Joint 4, the first wrist joint, lags most. The trial plot (`docs/figures/nominal_trial.png`, tracking panel) shows this happens at the start of each motion phase, when the reference velocity steps up. A likely reason is that the LQR applies the same 0.05 rad limit to every joint while joint 4 has one of the smallest torque limits (12 N m ), so the optimal gain accepts a larger transient there. Joint 4 is about 0.17 m from the tool, so even 18 mrad moves the tool by about 3mm, well inside the grasp tolerance.

### Diagnosed failure: joint 3 oscillation (issue #4)

The first design used the decoupled model (diagonal inertia only) for the LQR and the observer. In nominal trials joint 3 oscillated at about 43 Hz during the lift and the hold: its velocity flipped sign almost every control tick, and its torque swung between the +28 and -28 N m limits. Part A traces this to the neglected 0.64 shoulder-elbow coupling. A gain tuned for joint 3's diagonal inertia alone is too aggressive for the coupled shoulder-elbow mode. 

Designing on the full mass matrix fixed it. Same 4 seeds, same conditions: 

| Condition | Joint 3 saturated ticks per trial (decoupled -> coupled) | RMS torque (N m) | Tracking RMS (mrad) |
|---|---|---|---|
| nominal | 7.8 -> **0** | 4.4 -> 2.4 | 2.02 -> 1.79 |
| 40 N push | 9.0 -> **0** | 4.9 -> 2.9 | 12.8 -> 12.1 |
| gravity compensation 50 % | 9.5 -> **0** | 5.5 -> 3.8 | 3.38 -> 3.51 |
| encoder bias 0.05 rad | 104 -> **0** | 7.4 -> 2.4 | 49.8 -> 50.0 |
| 4x sensor noise | 69 -> **0.2** | 13.0 -> 9.3 | 3.70 -> 2.16 |

Success was 4/4 in every condition with both designs. The change removed the saturation, roughly halved the torque, and left tracking the same or better

### Trade-off: control effort against tracking error

`configs/sweeps/lqr_tuning.yaml` changes only the angle-error limit `q_err_max` (smaller = stiffer) and keeps everything else fixed (5 dev seeds per level, all 100 % success, `python scripts/run_sweep.py lqr_tuning --seeds dev --n 5`; figure from the committed 20-eval-seed run, `docs/figures/tradeoff_lqr_tuning.png`):

| q_err_max | Tracking RMS | RMS torque |
|---|---|---|
| 0.20 rad | 3.00 mrad | 2.2 N m |
| 0.10 rad | 2.49 | 2.2 |
| **0.05 rad (used)** | **1.79** | **2.4** |
| 0.025 rad | 1.22 | 2.7 |
| 0.0125 rad | 0.86 | 3.4 |

Each halving of `q_err_max` cuts the tracking error by about 30% for 10-25% more torque. With the coupled design the stiffer settings stay stable and saturation-free. With the earlier decoupled design, the stiffest setting made tracking worse (2.7 mrad at 12.9 N m), because the gain amplified sensor noise

The stiffer settings were not adopted because tracking does not limit the grasp. 1.79 mrad of joint error is under 1mm at the tool, the same size as the object estimate error (0.84mm, Part C), and both are far inside the 20mm capture radius. Halving the tracking error would not change the success rate, but it would raise the torque and make the loop more sensitive to the noise and disturbances in part E. Under 4x sensor noise the nominal gain already uses 9.3 N m. 

## Summary

LQR with integral action, Bryson weights justified from the grasp tolerance and the torque limits, designed on the coupled model. Closed-loop time constant about 0.1 s, joint tracking 1.79 mrad RMS. 
PBVS outer loop with a 1s time constant and a 1 rad/s rate limit, inside a 6 phase grasp state machine. 10/10 dev picks with a median 5.1 s to grasp
One controller failure (joint 3 oscillation) diagnosed to the model and fixed by the coupled design, with before/after numbers.
The effort/tracking trade off is quantified, and the chosen gain is defended by what limits the task (estimate accuracy and capture radius), not by tracking alone.

All numbers come from `python -m scripts.design_report`, `python scripts/run_trials.py --seeds dev` and `python scripts/run_sweep.py lqr_tuning --seeds dev --n 5`.