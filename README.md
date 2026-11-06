# UR3 vision-guided pick in MuJoCo

**41014 Sensors and Control, Integrated Project, Track B (manipulator: vision-guided pick)**

- Group: TODO
- Team: Patrick Blampied, Isaac Silva, Rayann Doha, Nathan Whibley
- Repository: https://github.com/patrock1/Sensors-and-Controls
- Demo video (5 min): TODO link

## Overview

A simulated UR3 arm with a wrist-mounted RGB-D camera finds a red cube placed at a random position and yaw on a table, works out where it is from the depth image, and picks it up. Nothing in the controller sees the true state of the simulator: joint angles come from noisy encoders through a Kalman filter (joint velocities aren't measured at all), and the cube position comes from the camera through a second filter. Ground truth is only used afterwards to score the trials.

TODO: pipeline diagram here (`docs/figures/pipeline.png`)

## 1. Results at a glance

Final evaluation is 50 held-out seeds (100 to 149), with the cube position and yaw randomised every trial. A trial counts as a success based on ground truth, not on what the state machine reports: the cube has to end up at least 0.10 m above the table and within 0.03 m of the tool.

TODO: results table from `results/summary/nominal_eval.json`
- success rate + 95 % Wilson interval
- object error when the arm commits to the grasp (xy / z)
- time to grasp
- joint NEES (12 = consistent)
- joint tracking RMS

Notes to include:
- all tuning was done on the dev seeds (0 to 9), eval seeds only run once at the end
- final runs were on MuJoCo 3.10.0 on an Intel Mac, which matched Linux to 0.1 mm

## 2. Install and run

```bash
git clone https://github.com/patrock1/Sensors-and-Controls
cd Sensors-and-Controls
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m scripts.build_ur3_model  # generates models/ur3/ur3.xml, needed before anything runs
python -m pytest -q
```

Intel Macs: MuJoCo 3.15.0 doesn't have an Intel macOS build, so install everything else and then `mujoco==3.10.0`. TODO: add the two commands.

TODO: table of the main commands
- `scripts/view.py` (viewer), `scripts/demo.py` (video of one pick)
- `scripts/run_trials.py --seeds eval` (the Section 1 results)
- `scripts/run_sweep.py <name>` (robustness and tuning sweeps)
- `python -m scripts.design_report` (prints the design numbers for Parts A, C, D)
- `scripts/calibrate_camera.py`, `scripts/check_localise.py` (Part B)
- `scripts/make_figures.py` (regenerates the figures)

All tunable values are in `configs/default.yaml`.

## 3. Pipeline and frames

Every 10 ms control tick:
1. sensors give noisy encoder angles (every tick) and an RGB-D frame (every 5th tick, so 20 Hz)
2. joint observer predicts with the last torque and updates with the encoder angles
3. on camera ticks: colour detection, depth back-projection, transform into the base frame using the *estimated* joint angles
4. object filter updates the cube position (with a gate to throw out bad measurements)
5. PBVS + grasp state machine turns the cube estimate into a joint reference
6. LQR computes torques from the estimated state and the reference

Ground truth is kept out structurally: `sim/sensors.py` is the only way into the pipeline, `evaluation/ground_truth.py` is the only thing that reads true state, and `tests/test_no_ground_truth.py` fails if any sensing/estimation/control module imports the simulator.

| Frame | Meaning |
|---|---|
| `B` | Robot base and world frame. Origin at the base on the table, z up, table top at z = 0 |
| `T` | Tool frame (`tool0` site), z along the approach direction |
| `C` | Camera frame, OpenCV convention (x right, y down, z forward) |
| `O` | Object frame, origin at the cube centre |

From a pixel to the torque:

```
pixel (u, v) + depth Z  ->  p_C = Z * inv(K) * [u, v, 1]
p_C  ->  p_B = T_B_T(q_hat) * T_T_C * p_C  ->  minus half the cube height (top face to centre)
p_B  ->  object filter  ->  PBVS target  ->  LQR torque
```

`q_hat` is the observer's estimate, never the true joint angles. Details and checks in `docs/part_b_sensing.md`.

## 4. Part A: Modelling

Full write-up: `docs/part_a_modelling.md`

- Each joint modelled as `M q̈ + b q̇ = τ`, 12 states (q and q̇), torques in, encoder angles out, discretised with ZOH at 10 ms
- M read from MuJoCo at the home pose, b matches the MJCF damping
- Assumptions: gravity cancelled by the sim's gravity compensation, inertia fixed at the home pose, joint coupling
- First design used only the diagonal of M (joints treated as independent). Joints 2 and 3 are strongly coupled and this caused the joint 3 oscillation in Section 9. Final design uses the full inertia matrix.
- Controllable and observable per joint and for the full system. Encoders alone make it observable, tachometers alone don't, so velocity has to be estimated (Part C)

TODO: check this matches the final Part A doc

## 5. Part B: Sensing, calibration and frames

Full write-up: `docs/part_b_sensing.md`

- Noise models: encoder noise, depth noise that grows with distance, dropped frames. The same values are used to build R, so R isn't hand-tuned
- Camera intrinsics from the field of view, checked against a fit over 40 views
- Depth sees the top face of the cube, not the centre, which gave a 20 mm bias. Fixed by shifting down half the cube size
- Noise sources: `sigma_q` = 0.5 mrad assumes a 13-bit encoder (one count is 0.77 mrad), depth noise uses the Week 4 RGB-D model `sigma_d = a + b * Z^2`

## 6. Part C: Estimation

TODO: link Part C write-up once it exists

- Joint observer: linear KF on [q, q̇], only q measured. R from encoder noise, Q from an unmodelled torque sigma_tau
- Object filter: cube treated as static (small random walk), 99 % chi-square gate, estimate frozen once the arm starts to descend
- Consistency checked with NEES (12 = consistent for 12 states)
- Trade-off: consistency vs responsiveness from the `observer_tuning` and `observer_tuning_push` sweeps. TODO: results + what sigma_tau we went with and why
- The gate doing real work: late in the approach the cube starts leaving the image, raw measurements get much worse, and the gate rejects them so the estimate stays good

## 7. Part D: Control

TODO: link Part D write-up once it exists

- Cascade: PBVS outer loop makes a joint reference, LQR inner loop tracks it
- Grasp state machine: search → approach → descend → grasp → lift → done, with timeouts
- LQR weights from Bryson's rule, integral action on position error, anti-windup when torque saturates
- Trade-off: control effort vs tracking from the `lqr_tuning` sweep. TODO: results + why the nominal weights
- Estimator vs controller speed. TODO: explain the ratio

## 8. Part E: Integration and robustness

Full write-up: `docs/part_e_robustness.md`

- Disturbances: wrist pushes, camera blackouts, reduced gravity compensation, encoder bias
- Calibration errors: hand-eye offset, focal length
- Sensor noise: filters that know the noise vs filters that don't
- Every failed trial gets a cause automatically (sensing / estimator / undiagnosed)
- TODO: summary of each sweep + figures, updated for the final controller

## 9. Failures found and how they were traced

- Cube fell through the table (collision settings)
- 20 mm localisation bias (depth sees the top face)
- Hand-eye calibration error never applied (sweep gave identical results at every level)
- Joint 3 oscillating at its torque limit (issue #4). TODO: symptom, diagnosis, fix, before/after numbers

## 10. Limitations

- Nothing checks whether the grasp actually worked
- Cube leaves the camera view late in the approach, so the arm commits to an early estimate
- R doesn't include the uncertainty in the joint estimates
- Observer has no bias state, so encoder bias makes it confidently wrong
- Model linearised at the home pose
- Detection relies on the cube being the only red object
- Simulation only

## 11. Repository layout

TODO: short tree (configs/, models/, ur3pick/ subfolders, scripts/, tests/, docs/, results/)

## 12. Individual contributions

- Patrick Blampied: see `docs/contributions/patrick.md`
- Isaac Silva: TODO
- Rayann Doha: see `docs/contributions/rayann.md`
- Nathan Whibley: TODO (documentation)

## 13. Generative AI declaration

TODO: each member lists the tools they used and what for (required by the marking guide)

## 14. Acknowledgements

- 41014 weekly tutorial material, adapted to this platform
- MuJoCo, Robotics Toolbox for Python, python-control, OpenCV
