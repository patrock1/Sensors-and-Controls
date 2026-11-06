# Part E: Integration and robustness

## Method

Each sweep runs the full pipeline on the 20 eval seeds (100 to 119) at every level. Success is judged from ground truth and not from the state machine: the cube centre must end at least 0.10 m above the table and within 0.03 m of the tool in xy. Success rates are shown with 95 % Wilson intervals because a normal interval breaks down near 100 % with only 20 trials.

All disturbances are injected inside `ur3pick/sim/` (`disturbances.py` and `sensors.py`). The pipeline only sees them through its sensor readings and the arm's response, the same way it would see a real disturbance.

Each failed trial gets a cause from `evaluation/metrics.py`. If the state machine reports its own reason (for example a search timeout) that wins. Otherwise the frozen object estimate is checked against the 20 mm capture radius (half the cube width). If the estimate was further off than that the trial is labelled `sensing` when the raw camera measurements were already about as wrong and `estimator` when the filter made them worse. Anything else stays `undiagnosed` rather than guessed.

Reproduce with:

    python scripts/run_sweep.py disturbance
    python scripts/run_sweep.py noise_levels
    python scripts/run_sweep.py calibration_error

Summaries are saved in `results/summary/sweep_*.json`.

## Disturbances

![disturbance sweep](figures/sweep_disturbance.png)

**Pushes.** A force on the wrist (world y) for 1 s at 5, 10, 20 and 40 N and a 20 N push during the descend phase all give 100 % success. The push does reach the arm. With 40 N on seed 0 the tool is knocked 23.4 mm off its nominal path (peak at 2.21 s) and is back to 15.3 mm by 2.5 s. The joint 1 peak torque rises from 17.0 to 29.9 N m as the LQR fights it. Success is unaffected because the object estimate is held in the base frame and the arm settles back onto its path before the grasp. We checked that the 4 to 5 s push lands fully inside the descend phase on all 20 seeds.

**Camera blackout.** Losing the camera for 0.5 s or for 1 to 3 s gives 100 %. The cube is static and the object filter already has an estimate from the first frames near the home pose. It holds that estimate while no measurements arrive. A blackout from 0 s was dropped from the sweep because it fails by design through the 2 s search timeout and says nothing new.

**Gravity compensation.** Reducing it to 90, 70 and 50 % gives 100 %. The integral action in the LQR absorbs the missing gravity torque.

**Encoder bias.** A constant offset on every encoder gives 100 % up to 0.03 rad and 65 % [43, 82] at 0.05 rad. The object error at 0.05 rad is 36.1 mm which is almost twice the capture radius. Most picks still succeed because the camera is mounted on the wrist. The same biased joint angles misplace the camera and the tool together and the error mostly cancels when the tool moves to where the camera saw the cube. The absolute object error therefore overstates the problem here. The observer has no bias state and its joint NEES median reaches about 69 000 (12 is consistent) which shows it is confidently wrong.

## Calibration error

![calibration sweep](figures/sweep_calibration_error.png)

The hand-eye offset the estimator assumes is moved along the tool x axis while the sim camera stays where it is mounted. The object error follows the offset 1:1 (4.9, 9.9, 14.9, 19.9 and 29.9 mm for 5 to 30 mm). Success is 100 % up to 15 mm, 55 % [34, 74] at 20 mm and 0 % at 30 mm. The failure starts exactly at the 20 mm capture radius. The gripper only catches the cube if its centre is within half a cube width of the tool axis and a hand-eye error moves the tool by the full offset. The hand-eye 30 mm bar is 0 % so only its error bar shows.

A focal length error has a smaller effect. Adding 2, 5 and 10 degrees to the 60 degree field of view gives object errors of 1.5, 4.3 and 11.7 mm. Only +10 degrees causes failures (90 % success, 2 failures labelled `sensing`).

Only translation along tool x was tested. Errors along other axes or in rotation were not.

## Sensor noise

![noise sweep](figures/sweep_noise_levels.png)

The original levels scale the true noise and the noise values the filters use for R together. With matched noise the system holds 100 % up to 32 * nominal with the object error growing from 0.8 to 9.9 mm. This is optimistic because the filters always know the true noise level.

To test a filter that underestimates its noise we added `noise.true_scale`. It raises only the noise the sim adds and the filters keep the nominal R. At 32 * true noise success drops to 60 % [39, 78] even though the object error (8.7 mm) is well inside the capture radius. The joint NEES median is about 6700 against 12 for a consistent filter. The joint observer trusts 32 * noisier encoders as if they were nominal and its joint velocity estimates fail. The mismatch costs 40 % of picks that matched noise did not.

## Failure diagnosis

| Level | Success | Cause | Traced to |
|---|---|---|---|
| hand-eye 20 mm | 55 % | sensing 7, undiagnosed 2 | offset at the capture radius |
| hand-eye 30 mm | 0 % | sensing 20 | offset beyond the capture radius |
| encoder bias 0.05 rad | 65 % | sensing 7 | bias not modelled in the observer |
| fovy +10 deg | 90 % | sensing 2 | focal length error |
| 32 * true noise, nominal R | 60 % | undiagnosed 8 | joint observer overconfident (NEES about 6700) |

The 2 undiagnosed hand-eye 20 mm failures had object errors just under 20 mm (19.9 mm median) where a small extra error tips the grasp over. The 8 undiagnosed noise failures are estimator failures. The cause rule only checks the object error which was fine in those trials. The joint NEES traces them to the observer.

In runs without encoder disturbances the joint NEES median is about 7. The observer is slightly conservative in normal operation.

## Limitations

- **No grasp check.** Every failed trial ended with the state machine reporting `done` while the cube was still on the table. Nothing senses whether the grasp worked after the gripper closes.
- **The cube leaves the camera view late in the approach.** Raw localisations degrade to about 25 mm and the object filter's chi-square gate rejects them. The object estimate stops updating after about 1 s and the arm commits to an estimate made near the home pose.
- **Joint 3 hits its 28 N m torque limit** even in nominal runs and oscillates at about 43 Hz during lift. The LQR is designed assuming unlimited torque.
- **The cause rule cannot see estimator failures in the joint state.** It only compares object errors.
- **Pixel noise is not added to the image.** `pixel_sigma` only sets R in `localise()`.
- **Everything is in simulation** with a single cube region chosen to be visible from the home pose.

With more time we would add a grasp check after the lift, add a bias state to the joint observer and stop the object filter from updating before the cube leaves the view.

## Fixes found during evaluation

- `calibration.hand_eye_error` was never applied. Every hand-eye level ran with perfect calibration and gave identical results. It is now passed from `pipeline.py` through `localise()` to `frames.T_B_C()` on the estimator side only.
- `scripts/demo.py` needed `imageio` and `imageio-ffmpeg`, which were missing from `requirements.txt`. They were added and the full pipeline was checked from a fresh virtual environment.
