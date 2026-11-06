Part A: Modelling, controlability and observability

PLATFORM
 The Robot is  UR3 in MuJuCo with six revolute joints driven by torque motors. The torque limits are
 56, 56, 28, 12, 12, 12, Nm
 (modelling/ur3_params.py)
 this matches the ctrlrange of each motor in models/ur3/ur3.xml
 The control loop runs at 100Hz (dt_ctrl = 0.01s) and the physics at 500 Hz.


KINEMATICS
 the arm uses the published UR3 standard DH Paramters: 
 d = [0.1519, 0, 0, 0.11235, 0.08535, 0.0819] m, 
 a = [0, -0.24365, -0.21325, 0, 0, 0] m, 
 alpha = [pi/2, 0, 0, pi/2, -pi/2, 0]

 models/ur3/ur3.xml are generated from these same numbers by 
 scripts/build_ur3_model.py, so the simulated arm and the model used by the pipeline cannot drift apart. scripts/check_fk.py compares the DH forward kinematics with the MuJuCo's tool pose at a 50 random configurations: the largest position error is 3.7e-16m and the largest rotation error is 2.0e-15, so the kinematic model is exact.


DYNAMIC MODEL
 Each joint is modelled on its own as a rotating inertia with the viscous friction:

    M_i * qddot_i + b * qdot_i = u_i,        i = 1..6

 With the state x = [q, qdot] (12 states), input u (6 torques) and output y = q (6 encoder angles):

    A = [ 0   I        ]     B = [ 0      ]     C = [ I  0 ]
    [ 0  -diag(b/M)]         [ diag(1/M) ]

 This is modeling/state_space.py: build_model().
 it is discretised with an exact zero_order hold at 0.01s (discretise(), the same call as the week 5 TUT), as opposed to forward Euler, because the fast wrist poles (-b/M up to -97 rad/s) are close to the 100 Hz sample rate.


WHERE THE PARAMTERS COME FROM

 |Parameter	     |   Value	   |     Source
 |M_i (kg m^2)	 |   0.3503,   |     diagonal of MuJoCo's full 
 |               |   0.4300    |     mass matrix at the home
 |               |   0.1060    |     pose (sim/inertia.py,
 |               |   0.0164    |     printed by scripts/
 |               |   0.0116    |     design_report.py)
 |               |   0.0103    | 
 |---------------|-------------|---------------------------------
 |b (N m s/rad)  | 1.0 for     |   the joint damping in the MJCF
 |               | every joint |   so the model friction is the 
 |               |             |   simulated friction
 |---------------|-------------|---------------------------------
 |home pose (rad)|  [0.466,    |   the start pose of every trial,
 |               |   -3.029,   |   with the camera looking down
 |               |    1.141,   |   at the table
 |               |    0.318,   |
 |               |   -1.571, 0]|
 |---------------|-------------|---------------------------------


 The continous-time eigenvalues are 0 (six integrators, the joint angles) and the:
 -b/M_i = -2.86, -2.33, -9.43, -61.0, -86.2 and -97.1 rad/s 
 (the velocity modes).

 After discretisation these map to 1.0 and 0.977 0.972, 0.910, 0.544, 0.422 and 0.379.
 Every mode is stable or marginally stable (the integrators) in th open loop


 ASSUMPTIONS
 1 - Gravity is cancelled by the simulator, MuJoCo's gravity     compensation (sim.gravecomp = 1.0) removes the gravity torue on the arm links, so the model has no gravity term and the controller adds no gravity feedforward. A partial concellation leaves a constat torque offset, which the integral action of the LQR ( Part D) removes. the Part E sweel tests this down to 50% gravity compensation with 100 % success. 

 2 - Joints are decoupled. The full dynamics are 
        M(q) qddot + C(q, qdot) qdot + b qdot = u.
     the model keeps only the diagonal of M at one pose and drops the off-diagnoal terms, the pose dependance and the coriolis/centrifugal term C. How much this throws away is measured below.

 3 - Friction is linear viscous. The motors are ideal torue sources with saturation and no actuator dynamics, which is how the MJCF motors behave.


HOW GOOD IS THE DECOUPLED MODEL?
 Coupling at the home pose. the normalise off-diagonal mass matrix |M_ij| / sqrt(M_ii M_jj) shows how strongly accelerating one joint loads another (0 = decoupled, 1 = rigidly coupled):

|        | **q1** | **q2**   | **q3**   | **q4** | **q5** | **q6** |
|--------|--------|----------|----------|--------|--------|--------|
| q1     | -      | 0.10     | 0.17     | 0.02   | 0.05   | 0.00   |
| q2     | 0.10   | -        | **0.64** | 0.20   | 0.00   | 0.00   |
| q3     | 0.17   | **0.64** | -        | 0.02   | 0.00   | 0.00   |
| q4     | 0.02   | 0.20     | 0.02     | -      | 0.00   | 0.00   |
| q5     | 0.05   | 0.00     | 0.00     | 0.00   | -      | 0.00   |
| q6     | 0.00   | 0.00     | 0.00     | 0.00   | 0.00   | -      |


 the Wrist joints (4-6) are nearly decoupled. The strong term is 0.64 between the shoulder (q4) and the elbow (q3): the forearms inertia is felt by both joints. Along the joint path of a real trial (dev seed 0, sampled every 0.1s) this ratio has a mediam of 0.69 and reaches 0.75 during the lift.

 POSE DEPENDANCE:
 Over the same trial the diagnoal inertias change as the arm extends and fols:

| Joint        | Model M   | Min along trial | Max along trial | Worst error of the model       |
|--------------|-----------|-----------------|-----------------|--------------------------------|
| 1 (base)     | 0.3503    | 0.3503          | 0.5037          | -30 % (true value 44 % higher) |
| 2 (shoulder) | 0.4300    | 0.4187          | 0.4973          | -14 % (true value 16 % higher) |
| 3 (elbow)    | 0.1060    | 0.0935          | 0.1060          | +13 % (true value 12 % lower)  |
| 4-6 (wrist)  | as config |                 |                 | under 0.5 %                    |


 the base joint is the worst: as the arm reaches out over the table the forearms swings further from the base axis and the base inertia grows by 44%

 What the mismatch does to the pipeline: The neglected coupling and inertia changes appear to the controller and the onbserver as a unmodelled torque on each joint. The design accounts for them in two places

    - Observer (Part C). the joint kalman filter treats the unmodelled torque as process noise with standard deviation observer.sigma_tau, set per joint. it is largest on joints 1-2 (6Nm) and on joint 3 ( 3Nm), where the coupling and the pose dependance are largest. The tuning of this value is studied in Part C.

    - Controller (Part D). the LDR's integral action removed any constant part of the error. The gains are moderate enough that, in practice, the loop stays stable even ehwere the modelled inertia is off by 44% all 10 dev trials succeed.

 The coupling is also the modt likely reason the shoulder and elbow behave worst in closed loop, Joint 3 shows a high frequency oscillation during the lift in nominal trials (issue # 4) and the tracking error in Part D is largest on joints 3 and 4


CONTROLLABILITY AND OBSERVABILITY

 For every joint the 2-state block (A_i, B_i) has a controllability matrix of rank 2 out of 2, and (A_i, C_i) an observability matrix of rank 2 out of 2.Thier smallest singular values are 0.86 to 1.0, so no joint is close to losing either property. The full 12-state system also passes the PBH test: rank [lambda I- A,B] = 12 and rankl [lambda ! - A:C] = 12 at every eigenvalue lambda of A ( modelling/analysis.py).

| Joint | ctrb rank | ctrb min singular value | obsv rank | obsv min singular value |
|-------|-----------|-------------------------|-----------|-------------------------|
| 1     | 2/2       | 0.900                   | 2/2       | 1.00                    |
| 2     | 2/2       | 0.862                   | 2/2       | 1.00                    |
| 3     | 2/2       | 0.989                   | 2/2       | 1.00                    |
| 4     | 2/2       | 1.00                    | 2/2       | 1.00                    |
| 5     | 2/2       | 1.00                    | 2/2       | 1.00                    |
| 6     | 2/2       | 1.00                    | 2/2       | 1.00                    |



 Becasue the model is decoupled, these results hold for any positive inertia and damping.
They do not depend on the exact parameter values above.

 Which snesors are needed. The same test tells us which sensor makes the state observable:

| Sensor set                  | Observable? | Why                                                                                                               |
|-----------------------------|-------------|-------------------------------------------------------------------------------------------------------------------|
| Encoders only (y = q)       | Yes         | qdot is the rate of change of q, so an observer can reconstruct it from the angle history                         |
| Tachometers only (y = qdot) | **No**      | the joint angle is an integrator of qdot: any constant angle offset gives the same velocity readings and can never be detected |


This is why the robot only needs its joint encoders and whhy part C needs an observer: the  velocties are no measured and have to be estimated.

All numbers in this section are printed by;

python -m scripts.design_report (Part A block) 

and 

python scripts/check_fk.py.