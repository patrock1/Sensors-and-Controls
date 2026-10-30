"""Run the nominal pipeline over many seeds and write per-trial metrics + a summary.

    python scripts/run_trials.py --seeds dev
    python scripts/run_trials.py --seeds eval --save-logs
"""
import argparse
import csv
import json

import _common as c
from ur3pick.pipeline import load_cfg, run_trial
from ur3pick.evaluation.metrics import trial_metrics, summarise
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="dev", choices=["dev", "eval"])
    ap.add_argument("--n", type=int, default=None, help="use only the first n seeds")
    ap.add_argument("--save-logs", action="store_true", help="keep each trial's full log in results/raw")
    ap.add_argument("--tag", default="nominal")
    a = ap.parse_args()

    cfg = load_cfg()
    seeds = c.seed_list(a.seeds)[: a.n]
    raw, summ = c.ROOT / "results" / "raw", c.ROOT / "results" / "summary"
    raw.mkdir(parents=True, exist_ok=True); summ.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in seeds:
        log = run_trial(s, cfg)
        m = trial_metrics(log, cfg)
        rows.append(m)
        print(f"seed {s:4d}  {'OK  ' if m['success'] else 'FAIL'}  phase={m['final_phase']:<8} "
              f"obj_err={m['obj_err_freeze']*1000:5.1f} mm  t={m['t_end']:.1f} s")
        if a.save_logs:
            np.savez_compressed(raw / f"{a.tag}_{s}.npz", **log)
    with open(summ / f"{a.tag}_{a.seeds}_trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    S = summarise(rows)
    with open(summ / f"{a.tag}_{a.seeds}.json", "w") as f:
        json.dump(S, f, indent=2)
    print(f"\nsuccess {S['success_rate']:.0%}  (95% CI {S['ci_low']:.0%} to {S['ci_high']:.0%}, n={S['n']})")


if __name__ == "__main__":
    main()
