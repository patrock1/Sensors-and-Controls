"""
metrics.py
Turns one trial log (interfaces.md section 6) into numbers, and many trials into a summary.

Evaluation may read ground truth (the gt_* keys). Nothing in here feeds back into the pipeline.

Design choices (each one is a viva question):
  * success is judged from TRUTH, not from the state machine's own "done" phase. A controller
    that believes it succeeded while the cube sits on the table is the failure we most want to
    catch, so both are reported and a disagreement is flagged.
  * the object error is measured at the FREEZE point, the first step of the "descend" phase.
    That is the estimate the arm actually commits to; after the gripper closes the estimate
    no longer matters, so an end-of-trial error would hide the cause of a miss.
  * the error is split into lateral (xy) and vertical (z). A grasp misses laterally; a z error
    mostly shows up as the gripper closing too high or low. Splitting them also exposes a
    constant z offset, e.g. measuring the cube's top face instead of its centre (20 mm).
  * success-rate intervals are Wilson score intervals, not the normal approximation, because
    with ~50 trials and a success rate near 100 % the normal interval goes above 1 or has zero
    width when every trial succeeds.
"""

import json
import numpy as np

from ur3pick.estimation.consistency import nees

FREEZE_PHASE = "descend"
GRASP_PHASE = "grasp"
NEES_SKIP_S = 0.5        # s: ignore the start, while P is still shrinking from its initial value
SENSING_SHARE = 0.5      # raw error >= half the estimate's error -> the camera chain explains the miss
MEAS_WINDOW = 5         # first valid raw localisations used (median): one noisy frame should not decide the cause


def _phases(log):
    return np.asarray(log["phase"]).astype(str)


def _first(mask):
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def _meta(log):
    m = log.get("meta", "{}")
    m = m.item() if isinstance(m, np.ndarray) else m
    try:
        return json.loads(m) if isinstance(m, (str, bytes)) else dict(m)
    except (TypeError, ValueError):
        return {}


def truth_success(log, cfg):
    """Cube centre above success_height at the end and, if the tool position was logged, within
    success_hold_tol of the tool in xy (interfaces.md section 6 proposes 0.03 m), so a cube that
    was flung upward but not held does not count. Thresholds come only from configs/default.yaml
    (open decision 8)."""
    g = cfg["grasp"]
    p = np.asarray(log["gt_p_obj"])[-1]
    ok = bool(p[2] >= g["success_height"])
    if ok and "gt_p_tool" in log:
        tool = np.asarray(log["gt_p_tool"])[-1]
        ok = bool(np.linalg.norm(p[:2] - tool[:2]) <= g.get("success_hold_tol", 0.03))
    return ok


def joint_nees(log, skip_s=NEES_SKIP_S):
    """Mean NEES of the 12-state joint estimate. Expect ~12 for a consistent filter
    (Part C). Steps with a singular covariance are skipped rather than crashing the run."""
    t = np.asarray(log["t"])
    x_true = np.hstack([log["gt_q"], log["gt_qdot"]])
    vals = []
    for k in np.flatnonzero(t >= t[0] + skip_s):
        try:
            vals.append(nees(x_true[k] - log["x_hat"][k], log["P_x"][k]))
        except np.linalg.LinAlgError:
            continue
    return float(np.mean(vals)) if vals else float("nan")


def capture_radius(cfg):
    """Largest lateral (xy) estimate error the grasp can absorb: half the cube edge. Beyond it the
    tool closes beside the cube instead of over it, which is the 20 mm hand-eye limit in Part E."""
    return cfg["object"]["size"] / 2


def raw_meas_err_xy(log, k, n=MEAS_WINDOW):
    """Median xy error of the FIRST n valid raw localisations (p_meas) up to step k.
    Comparing this with the filtered error at the freeze point tells sensing apart from estimator.

    Why the first and not the last: late in approach the camera is close and the cube starts to
    leave the image, so raw localisations degrade (~25 mm off in nominal trials) and the object
    filter's chi-square gate rejects them. The estimate is built from the early measurements taken
    near the home pose, so those are the ones to compare it against."""
    if k is None or "p_meas" not in log:
        return float("nan")
    p_meas = np.asarray(log["p_meas"], float)[: k + 1]
    p_true = np.asarray(log["gt_p_obj"], float)[: k + 1]
    valid = np.asarray(log.get("meas_valid", np.isfinite(p_meas).all(axis=1)), bool)[: k + 1]
    idx = np.flatnonzero(valid & np.isfinite(p_meas).all(axis=1))[:n]
    if idx.size == 0:
        return float("nan")
    return float(np.median(np.linalg.norm(p_meas[idx, :2] - p_true[idx, :2], axis=1)))


def _cause(meta, success, err_xy, meas_xy, radius):
    """Failure cause for the README table, in order of how certain we are:
      1. the state machine's own diagnosis (e.g. 'sensing' on search timeout) wins: it knows why it stopped;
      2. the frozen estimate was off by more than the capture radius, so the arm went to the wrong place:
         'sensing' if the raw localisations explain most of that error (camera chain: calibration,
         detection, depth), 'estimator' if the filter made the estimate much worse than its inputs;
      3. anything else stays 'undiagnosed' rather than guessing, so it stands out until someone traces it."""
    if success:
        return "none"
    c = meta.get("failure_cause", "none")
    if c not in (None, "", "none"):
        return c
    if np.isfinite(err_xy) and err_xy > radius and np.isfinite(meas_xy):
        # compare the two errors with each other, not each with the radius: near the radius a bias of
        # 19.7 mm raw / 20.1 mm filtered is plainly the same sensing error passing through the filter
        return "sensing" if meas_xy >= SENSING_SHARE * err_xy else "estimator"
    return "undiagnosed"


def trial_metrics(log, cfg) -> dict:
    """One row per trial. Keys used by scripts/run_trials.py and run_sweep.py:
    success, final_phase, obj_err_freeze, t_end."""
    t = np.asarray(log["t"])
    ph = _phases(log)
    meta = _meta(log)
    p_hat = np.asarray(log["p_hat"])
    p_true = np.asarray(log["gt_p_obj"])

    # freeze point: first descend step; if the arm never got there, the last step with an estimate
    k = _first(ph == FREEZE_PHASE)
    if k is None:
        k = _first(np.all(np.isfinite(p_hat), axis=1)[::-1])
        k = None if k is None else len(t) - 1 - k
    if k is None:                                   # object never seen
        err = err_xy = err_z = float("nan")
    else:
        e = p_hat[k] - p_true[k]
        err, err_xy, err_z = float(np.linalg.norm(e)), float(np.linalg.norm(e[:2])), float(e[2])

    kg = _first(ph == GRASP_PHASE)
    success = truth_success(log, cfg)
    claimed = bool(ph[-1] == "done")
    meas_xy = raw_meas_err_xy(log, k)
    return dict(
        seed=meta.get("seed", -1),
        success=success,
        claimed_success=claimed,
        disagree=success != claimed,                 # state machine and truth disagree -> diagnose
        final_phase=str(ph[-1]),
        failure_cause=_cause(meta, success, err_xy, meas_xy, capture_radius(cfg)),
        obj_err_freeze=err,
        obj_err_freeze_xy=err_xy,
        obj_err_freeze_z=err_z,
        meas_err_freeze_xy=meas_xy,
        t_grasp=float(t[kg]) if kg is not None else float("nan"),
        t_end=float(t[-1]),
        nees_joint=joint_nees(log),
        u_rms=float(np.sqrt(np.mean(np.asarray(log["u"]) ** 2))) if "u" in log else float("nan"),
    )


def wilson(k, n, z=1.96):
    """95 % Wilson score interval for k successes in n trials."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z**2 / n
    mid = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return float(max(0.0, mid - half)), float(min(1.0, mid + half))   # clip float round-off at 0 and 1


def summarise(rows) -> dict:
    """Many trials -> one summary. Medians, not means, for errors and times: one wild failure
    should not move the headline number, and the failure table reports those separately."""
    n = len(rows)
    k = sum(r["success"] for r in rows)
    lo, hi = wilson(k, n)

    def med(key, only_success=False):
        v = np.array([r[key] for r in rows if r["success"] or not only_success], float)
        v = v[np.isfinite(v)]
        return float(np.median(v)) if v.size else float("nan")

    def p90(key):
        v = np.array([r[key] for r in rows], float)
        v = v[np.isfinite(v)]
        return float(np.percentile(v, 90)) if v.size else float("nan")

    causes = {}
    for r in rows:
        if not r["success"]:
            causes[r["failure_cause"]] = causes.get(r["failure_cause"], 0) + 1
    return dict(
        n=n,
        success_rate=k / n if n else float("nan"),
        ci_low=lo,
        ci_high=hi,
        obj_err_freeze_median=med("obj_err_freeze"),
        obj_err_freeze_p90=p90("obj_err_freeze"),
        obj_err_xy_median=med("obj_err_freeze_xy"),
        obj_err_z_median=med("obj_err_freeze_z"),
        t_grasp_median=med("t_grasp", only_success=True),
        nees_joint_median=med("nees_joint"),
        disagreements=int(sum(r["disagree"] for r in rows)),
        failure_causes=causes,
    )
