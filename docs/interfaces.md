# Interfaces (team contract)

**Status:** DRAFT v0.1. Everything marked **PROPOSED** is a default we change by pull request, not by editing our own copy. Everything under **Open decisions** needs an answer before the owner of that area writes real code.

Purpose: each person can build and test their folder alone, then plug it into the pipeline without rewriting anything. If your code needs something that is not in this document, add it here first.

---

## 1. Global conventions

| Item | Convention |
|---|---|
| Units | SI only. Metres, radians, seconds, kg, N·m. Pixels only inside `sensing/detection.py`. |
| Arrays | `numpy.ndarray`, `float64`. Vectors are 1-D, shape `(n,)`. Never column vectors. |
| Poses | 4x4 homogeneous matrices `T_a_b` (pose of frame b expressed in frame a). Name them this way everywhere. |
| Joint order | `joint1 ... joint6`, base to wrist, matching `models/ur3/ur3.xml` and `ur3_params.py`. |
| State vector | `x = [q1..q6, qdot1..qdot6]`, shape `(12,)`. |
| Randomness | Always from a `numpy.random.Generator` created from the trial seed. No global `np.random`. |
| Config | All tunable numbers live in `configs/default.yaml`. No magic numbers in code. |
| Time | Simulator `dt_sim = 0.002 s`. **PROPOSED** control period `dt_ctrl = 0.01 s` (every 5 sim steps). **PROPOSED** camera period `dt_cam = 0.05 s` (20 Hz). |

### Frames

| Frame | Meaning |
|---|---|
| `B` | Robot base frame. Origin at the base on the table surface, z up. **This is also the world frame.** Table top is `z = 0`. |
| `T` | Tool frame (`tool0` site in the MJCF), z along the approach direction. |
| `C` | Camera frame, OpenCV convention: x right, y down, z forward. |
| `O` | Object frame. Origin at the cube centre. |

Notes:
- MuJoCo cameras look along their own **-z** with y up. Convert to the OpenCV convention in exactly one place (`sensing/frames.py`) and nowhere else.
- `T_T_C` (hand-eye) is a constant. Its source of truth is `scripts/build_ur3_model.py` and it is mirrored in `configs/default.yaml`. A test must check the two agree.
- The object position the rest of the pipeline sees is **`p_B_O`** (cube centre in frame `B`, metres).

---

## 2. Data types

Defined once in `ur3pick/interfaces.py` (dataclasses). Nobody redefines them locally.

```python
@dataclass
class SensorPacket:           # produced by sim/sensors.py every dt_ctrl
    t: float                  # simulation time, s
    q_meas: np.ndarray        # (6,) noisy encoder angles, rad
    rgb: np.ndarray | None    # (H, W, 3) uint8, or None if no new frame this step
    depth: np.ndarray | None  # (H, W) float32, metres along camera z, or None
    cam_stamp: float | None   # time the image was captured, s

@dataclass
class Detection:              # produced by sensing/detection.py
    found: bool
    centroid_px: np.ndarray   # (2,) [u, v] pixels; NaN if not found
    area_px: int
    mask: np.ndarray | None   # (H, W) bool, for debugging and the video only

@dataclass
class ObjectMeasurement:      # produced by sensing/localisation.py
    t: float
    valid: bool
    p_B: np.ndarray           # (3,) object centre in B, m; NaN if not valid
    R: np.ndarray             # (3, 3) measurement covariance in B, m^2
    p_C: np.ndarray           # (3,) object centre in C, m (kept for IBVS and debugging)

@dataclass
class Estimate:               # produced by the estimators
    t: float
    x_hat: np.ndarray         # (12,) joint state estimate
    P_x: np.ndarray           # (12, 12) its covariance
    p_hat: np.ndarray         # (3,) object position estimate in B, m
    P_p: np.ndarray           # (3, 3) its covariance
    p_seen: bool              # False while the object has not been observed yet

@dataclass
class Command:                # produced by control, consumed by sim
    u: np.ndarray             # (6,) joint torques, N m, already clipped to limits
    gripper: float            # 0 = open, 1 = closed (see Open decisions, item 3)
    phase: str                # current grasp phase, for logging
```

---

## 3. Module contracts

Signatures are the contract. Bodies are up to the owner.

### 3.1 Simulation (`ur3pick/sim/`)

```python
# scene.py
class Scene:
    def __init__(self, cfg: dict): ...
    def reset(self, seed: int) -> None            # reloads state, randomises object, seeds all RNGs
    def step(self, cmd: Command) -> None          # advances dt_ctrl (several sim steps)
    @property
    def t(self) -> float

# randomise.py
def sample_object_pose(rng: np.random.Generator, cfg: dict) -> tuple[np.ndarray, float]
    # returns (p_B_O (3,), yaw rad). PROPOSED region: radius 0.20 to 0.35 m from the base axis,
    # |angle| <= 60 deg about +x, z = 0.02 (cube resting on table). Yaw uniform in [0, pi/2).

# sensors.py   (THE ONLY pipeline-facing path to simulator state)
class SensorSuite:
    def __init__(self, scene: Scene, cfg: dict, rng: np.random.Generator): ...
    def read(self) -> SensorPacket
```

Rules for `sensors.py`:
- Encoders give **only** `q`, with Gaussian noise `sigma_q` from config. Joint **velocities are not measured**, and that is why the observer exists.
- Camera noise and dropouts (frame missing, object out of view) are injected here, never downstream.
- Depth noise model is range dependent (see config `depth_noise`), so the filter's `R` can be justified from it.

### 3.2 Modelling (`ur3pick/modelling/`)

```python
# state_space.py
def build_model(cfg: dict) -> tuple[A, B, C, D]   # numpy arrays for x = [q, qdot], u = torque
def gravity_torque(q: np.ndarray) -> np.ndarray   # (6,), uses q only (see Open decisions, item 1)

# analysis.py
def controllability_rank(A, B) -> int
def observability_rank(A, C) -> int
def observable_states(A, C) -> dict               # per-sensor summary for the README
```

`ur3_params.py` is the single source for DH values, masses and limits. Do not copy these numbers anywhere.

### 3.3 Sensing (`ur3pick/sensing/`)

```python
# calibration.py
def intrinsics_from_fovy(fovy_deg, width, height) -> np.ndarray     # K (3, 3)
def calibrate_from_images(images, board_shape, square_m) -> tuple[K, dist]

# detection.py
def detect_object(rgb: np.ndarray, cfg: dict) -> Detection

# localisation.py
def localise(det: Detection, depth: np.ndarray, K, q_hat: np.ndarray, t: float, cfg: dict) -> ObjectMeasurement

# frames.py
def T_B_T(q: np.ndarray) -> np.ndarray       # forward kinematics from ur3_params.fkine
def T_T_C() -> np.ndarray                    # constant hand-eye transform (OpenCV camera convention)
def T_B_C(q: np.ndarray) -> np.ndarray       # = T_B_T(q) @ T_T_C()
```

**Hard rule:** `localise` receives the **estimated** joint angles `q_hat`, never the true ones. Any error in `q_hat` must propagate into `p_B` and into `R`.

### 3.4 Estimation (`ur3pick/estimation/`)

```python
# joint_observer.py
class JointObserver:
    def __init__(self, cfg: dict): ...
    def predict(self, u: np.ndarray, dt: float) -> None
    def update(self, q_meas: np.ndarray) -> None
    x_hat: np.ndarray    # (12,)
    P: np.ndarray        # (12, 12)

# object_filter.py
class ObjectFilter:
    def __init__(self, cfg: dict): ...
    def predict(self, dt: float) -> None                     # object is static, so process noise is small
    def update(self, meas: ObjectMeasurement) -> None        # ignores meas when meas.valid is False
    p_hat: np.ndarray    # (3,)
    P: np.ndarray        # (3, 3)
    seen: bool

# consistency.py  (offline, used by evaluation and plots)
def nees(x_true, x_hat, P) -> float
def nis(innovation, S) -> float
```

### 3.5 Control (`ur3pick/control/`)

```python
# lqr.py
class JointLQR:
    def __init__(self, A, B, cfg: dict): ...                 # solves the Riccati equation once
    def command(self, x_hat: np.ndarray, x_ref: np.ndarray) -> np.ndarray   # (6,) torque, clipped

# pbvs.py / ibvs.py
def visual_servo_target(q_hat: np.ndarray, p_hat: np.ndarray, phase_goal: str, cfg: dict) -> np.ndarray
    # returns q_ref (6,) for the LQR. PBVS uses p_hat in B; IBVS (extension) uses image features.

# grasp.py
class GraspStateMachine:
    PHASES = ["search", "approach", "descend", "grasp", "lift", "done", "failed"]
    def step(self, est: Estimate) -> tuple[np.ndarray, float, str]   # (x_ref, gripper, phase)
```

Everything in `control/` consumes an `Estimate`. Nothing in `control/` imports `sim.scene` or `evaluation`.

### 3.6 Evaluation (`ur3pick/evaluation/`)

```python
# ground_truth.py   (the ONLY module that reads true state)
def read_truth(scene: Scene) -> dict      # keys: q (6,), qdot (6,), p_B_O (3,), object_lifted (bool)

# metrics.py
def trial_metrics(log: dict) -> dict      # success, final_pos_error, time_to_grasp, rms errors, etc.
```

---

## 4. Allowed imports (enforced by `tests/test_no_ground_truth.py`)

| Package | May import | Must not import |
|---|---|---|
| `sensing/`, `estimation/`, `control/`, `modelling/` | `interfaces`, `modelling`, `sensing.frames`, `numpy`, `scipy`, `cv2`, `control` | `evaluation`, `sim.scene`, `mujoco` |
| `sim/sensors.py` | `sim.scene`, `interfaces`, `mujoco` | `evaluation` |
| `evaluation/` | anything | n/a |
| `scripts/` | anything | n/a |

Only `scripts/` and `evaluation/` may read ground truth. The test scans imports in the first row and fails the build if the rule is broken.

---

## 5. The main loop (`ur3pick/pipeline.py`)

```text
reset(seed)
loop every dt_ctrl:
    pkt  = sensors.read()
    obs.predict(u_prev, dt_ctrl);  obs.update(pkt.q_meas)
    if pkt.rgb is not None:
        det  = detect_object(pkt.rgb)
        meas = localise(det, pkt.depth, K, obs.x_hat[:6], pkt.cam_stamp)
        objf.update(meas)
    objf.predict(dt_ctrl)
    est  = Estimate(...)
    x_ref, grip, phase = grasp.step(est)
    u    = lqr.command(est.x_hat, x_ref) + gravity_torque(est.x_hat[:6])   # see Open decisions, item 1
    scene.step(Command(u, grip, phase))
    log.append(pkt, est, cmd, truth=read_truth(scene))   # truth goes to the log only
```

Only `log.append` receives ground truth.

---

## 6. Trial log format

One `results/raw/<run_name>/trial_<seed>.npz` per trial, plus one `summary.csv` per run (committed, small).

| Key | Shape | Meaning |
|---|---|---|
| `t` | `(T,)` | time stamps |
| `q_meas` | `(T, 6)` | noisy encoder readings |
| `x_hat`, `P_x` | `(T, 12)`, `(T, 12, 12)` | joint estimate and covariance |
| `p_hat`, `P_p` | `(T, 3)`, `(T, 3, 3)` | object estimate and covariance |
| `p_meas`, `meas_valid` | `(T, 3)`, `(T,)` | raw localisations and whether they were valid (NaN rows allowed) |
| `u` | `(T, 6)` | commanded torques |
| `phase` | `(T,)` | grasp phase as strings |
| `gt_q`, `gt_qdot`, `gt_p_obj` | `(T, 6)`, `(T, 6)`, `(T, 3)` | ground truth (**evaluation only**) |
| `meta` | JSON string | `seed`, `config_hash`, `git_commit`, `success`, `failure_cause`, `t_grasp` |

`failure_cause` is one of `none, model, sensing, estimator, controller, timeout, collision`. Whoever diagnoses a failure sets it, so the README failure table can be generated.

**PROPOSED success definition:** the cube centre is at least 0.05 m above the table and within 0.03 m of the tool origin for 1.0 s after the lift phase.

---

## 7. Config keys (`configs/default.yaml`)

Top-level groups, so nobody edits the same lines by accident:

```yaml
sim:         { dt_sim, dt_ctrl, dt_cam, image_width, image_height }
object:      { size, region: { r_min, r_max, angle_max_deg }, yaw_range }
noise:       { sigma_q, depth_noise: { base_sigma, range_gain }, pixel_sigma, dropout_prob }
camera:      { fovy_deg, hand_eye: { position, axes } }
detection:   { hsv_low, hsv_high, min_area_px }
observer:    { Q, R, P0 }
object_filter: { Q, R_floor, P0 }
lqr:         { Q_diag, R_diag }
servo:       { type, gain, pre_grasp_height, descend_speed }
grasp:       { success_height, hold_time, timeout }
```

Each owner edits only their own group. Q and R matrices are written as diagonal vectors with a one-line comment saying where each number came from. The viva will ask.

---

## 8. Workflow rules

- `main` always runs. Work on `feat/<area>-<topic>` branches and merge by pull request.
- A pull request needs one review from someone who does **not** own that folder.
- Changing anything in this document needs approval from every area owner affected.
- If you add a dependency, regenerate `requirements.txt` in the same pull request.
- Each member commits from their own account. Commit messages say what changed and why.

---

## 9. Ownership

| Area | Criterion | Owner | Reviewer |
|---|---|---|---|
| `modelling/` | A | _tbc_ | _tbc_ |
| `sensing/`, `sim/sensors.py` | B | _tbc_ | _tbc_ |
| `estimation/` | C | _tbc_ | _tbc_ |
| `control/` | D | _tbc_ | _tbc_ |
| `pipeline.py`, | E | TBC | TBC |
`scripts/`, `evaluation/`, README | F | Pat | Isaac |

Everyone must be able to explain the whole pipeline in the viva. Review across areas is how that happens.

---

## 10. Open decisions

| # | Decision | Options | Owner | Needed by |
|---|---|---|---|---|
| 1 | Gravity handling | (a) model assumes gravity compensated, with `gravity_torque(q_hat)` added as feedforward; (b) linearise full dynamics about an operating point | modelling + control | before LQR work starts |
| 2 | Joint-level actuation | torque motors (current XML) vs position actuators | control | before LQR work starts |
| 3 | Gripper | simple finger geoms with a closing joint vs weld constraint between tool and cube | control + sim | before grasp work starts |
| 4 | Observer type | Kalman filter vs Luenberger observer vs EKF | estimation | after item 1 |
| 5 | Object detection | HSV colour segmentation (default) vs ArUco marker | sensing | this week |
| 6 | Camera rate and latency | keep 20 Hz, no latency vs add 1 frame of latency as a robustness test | sensing + sim | this week |
| 7 | Home pose | one fixed camera-down pose for every trial (current proposal: `q = [0.466, -3.029, 1.141, 0.318, -1.571, 0.0]`) | control | this week |
| 8 | Success definition | confirm section 6 proposal | all | before the first full evaluation |

---

## 11. Change log

| Version | Date | Change |
|---|---|---|
| 0.1 | _fill in_ | Initial draft |
