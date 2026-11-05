# Part B: Sensing, calibration and frames

## Sensors and noise models

All sensor noise is added in one place, `ur3pick/sim/sensors.py`. The pipeline only receives a `SensorPacket` and never reads simulator state directly.

**Encoders.** The encoders give joint angles only with Gaussian noise of `sigma_q` = 0.0005 rad per joint (an assumed 13-bit encoder). Joint velocities are not measured. This is why the joint observer in Part C exists.

**Wrist RGB-D camera.** The camera renders a 320 x 240 colour and depth image at 20 Hz (every 5th 100 Hz control tick) with a 60 degree vertical field of view. Depth noise follows the Week 4 RGB-D model `sigma_d = depth_a + depth_b * Z^2` with `depth_a` = 0.0015 m and `depth_b` = 0.0022 m^-1. At the 0.19 m working distance this gives about 1.6 mm. Each frame is dropped with probability 0.01. The same `depth_a`, `depth_b` and `pixel_sigma` values set the measurement covariance R in `localise()`, so R comes directly from the noise model instead of being tuned by hand.

**Reproducibility.** Each trial seed is split into separate random streams for the cube pose and the sensor noise. Changing the noise level does not move the cube for the same seed. The same seed gives the same result on any machine (checked in a fresh virtual environment).

The colour and depth images are rendered by two separate MuJoCo renderers. A single renderer switched between colour and depth mode dropped the cube from later colour frames on macOS.

## Detection and localisation

`detect_object()` thresholds the image in HSV. Red wraps around hue 0 so a pixel counts when hue <= 10 or >= 170 with saturation >= 100 and value >= 60. The centroid of the mask is the detection. At the home pose the cube covers about 1970 px.

`localise()` takes the median depth over the mask and back-projects the centroid with `p_C = Z * inv(K) * [u, v, 1]`. It then moves the point into the base frame with `T_B_C(q_hat)`. It uses the **estimated** joint angles `q_hat` from the observer and never the true ones. Any joint estimate error therefore shows up in the object position, the same as on a real robot.

The depth on the mask is the distance to the cube's **top face** and not its centre. Over 20 seeds the raw estimate sat +20.0 mm above the cube centre, which is exactly half the 40 mm cube. `localise()` now shifts the point down by `localisation.surface_to_centre` = 0.02 m along base z. This assumes the cube rests upright on the table, which holds because `randomise.py` only changes its yaw.

The measurement covariance is built in the camera frame and rotated into the base frame:

    R_C = diag(sxy^2, sxy^2, sz^2),   sxy = pixel_sigma * Z / fx,   sz = depth_a + depth_b * Z^2
    R_B = R_BC * R_C * R_BC^T

## Calibration

The intrinsics come from the field of view in the MJCF: `fy = (H / 2) / tan(fovy / 2)` = 207.8 px with the principal point at the image centre (160, 120).

`scripts/calibrate_camera.py` checks this against images. It places the cube at 40 known poses, detects the centroid in each and fits the pinhole model `u = fx * x / z + cx` by linear least squares. Ground truth is used here because this is an offline calibration step and not part of the running pipeline.

| | fx | fy | cx | cy |
|---|---|---|---|---|
| fitted (40 views) | 207.4 | 207.5 | 159.5 | 119.5 |
| from MJCF fovy | 207.8 | 207.8 | 160.0 | 120.0 |

The reprojection RMS is 0.15 px. The focal length agrees to within 0.2 %. The principal point is off by exactly half a pixel in both axes. This is the pixel centre convention: the mask centroid counts pixel `i` at `i` while the optical axis passes between pixels 159 and 160. It explains the small constant error seen in localisation: 0.5 px at 0.19 m with f = 207.8 px is 0.46 mm, close to the measured -0.6 mm in y. It was left uncorrected because it is below a millimetre.

Part E shows how sensitive the system is to calibration error. A focal length error of +10 degrees still gives 90 % success. A hand-eye error fails at the 20 mm capture radius.

## Frames

| Frame | Meaning |
|---|---|
| `B` | Robot base and world frame. Origin at the base on the table, z up, table top at z = 0 |
| `T` | Tool frame (`tool0` site), z along the approach direction |
| `C` | Camera frame, OpenCV convention (x right, y down, z forward) |
| `O` | Object frame, origin at the cube centre |

The chain from measurement to the quantity the controller acts on is:

    pixel (u, v) + depth Z  ->  p_C  ->  T_B_C(q_hat) = T_B_T(q_hat) * T_T_C  ->  p_B  ->  object filter (Part C)  ->  grasp target

`T_B_T` is the standard DH forward kinematics in `modelling/ur3_params.py`. The hand-eye transform `T_T_C` is a pure translation of [0, 0.05, 0.07] m in the tool frame because the camera axes are built to line up with the tool axes. Its value lives in one place (`CAM_OFFSET_IN_TOOL`) and both the MJCF generator and `frames.py` read it.

MuJoCo cameras look along their own -z with y up. The conversion to the OpenCV convention is done once in `scripts/build_ur3_model.py` through the camera `xyaxes` and nowhere else in the pipeline.

**Checks.** `tests/test_frames.py` confirms that the MuJoCo camera pose, converted to OpenCV axes, matches `frames.T_B_C()` to 1e-9. At the home pose both give the tool at (0.25, 0.00, 0.30) m and the camera at (0.295, 0.022, 0.23) m. Projecting the cube centre predicts pixel (97, 131) while the detected centroid is at (90.1, 131.3). The difference is expected. The camera mostly sees the top face, which is 20 mm closer and projects further from the image centre. Projecting the top face instead gives u = 90.4.

## Results

Localisation over 20 seeds from the home pose, using noisy encoder readings as `q_hat`:

| | x | y | z |
|---|---|---|---|
| mean error before top-face fix | | | +20.0 mm |
| mean error after fix | -0.2 mm | -0.6 mm | -0.0 mm |

The median lateral error is 0.7 mm and all 20 seeds were localised. In the full eval runs the object error at the moment the arm commits to the grasp has a median of 0.8 mm (Part E).

Reproduce with `python scripts/check_localise.py --n 20` and `python scripts/calibrate_camera.py`.

## Issues found and fixed

- **The cube fell through the table.** The geom default in the generated `ur3.xml` (`contype 1, conaffinity 0`) also applied to the table, floor and cube in `scene.xml`, so nothing collided. Found by logging the cube height over time while debugging a detection that failed. The table, floor and cube now have their own collision bits so the cube rests on the table and is hit by the arm while the arm still passes through the table.
- **Localisation measured the top face** (+20.0 mm in z), fixed as above.
- **One renderer for colour and depth dropped the cube on macOS**, fixed by using two renderers.

## Limitations

- **R ignores joint estimate uncertainty.** It only includes camera noise even though `q_hat` errors move `p_B`. This is why the encoder bias test in Part E produces object errors the filter does not expect.
- **The cube leaves the view late in the approach.** Raw localisations degrade to about 25 mm and are rejected by the object filter's gate.
- **Pixel noise is not added to the image.** `pixel_sigma` only sets R.
- **Detection relies on colour.** It works because the cube is the only red object in the scene.
- **The top-face correction assumes an upright cube.**
- **Depth precision on macOS** is limited by the missing `ARB_clip_control` extension. The measured 0.190 m depth on the cube matched the expected value so the effect was not visible here.
