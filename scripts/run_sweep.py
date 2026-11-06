"""Part E robustness sweep.

    python scripts/run_sweep.py noise_levels
    python scripts/run_sweep.py calibration_error --n 5      # quick check
    python scripts/run_sweep.py disturbance
    python scripts/run_sweep.py lqr_tuning                # Part D effort vs tracking trade-off
Writes results/summary/sweep_<name>.json and a figure to docs/figures/.
"""
import argparse
import json

import _common as c
from ur3pick.pipeline import load_cfg, run_trial
from ur3pick.evaluation.metrics import trial_metrics, summarise
from ur3pick.evaluation import plots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=sorted(p.stem for p in (c.ROOT / "configs" / "sweeps").glob("*.yaml")))
    ap.add_argument("--n", type=int, default=None, help="override the number of seeds per level")
    ap.add_argument("--seeds", default="eval", choices=["dev", "eval"])
    a = ap.parse_args()

    sweep = c.load_sweep(a.name)
    n = a.n or sweep["n_seeds"]
    seeds = c.seed_list(a.seeds)[:n]
    results = []
    for level in sweep["levels"]:
        cfg = load_cfg(overrides=level["overrides"])
        rows = [trial_metrics(run_trial(s, cfg), cfg) for s in seeds]
        S = summarise(rows)
        results.append((level["name"], S))
        print(f"{level['name']:<26} success {S['success_rate']:.0%} "
              f"[{S['ci_low']:.0%}, {S['ci_high']:.0%}]  obj_err median {S['obj_err_freeze_median']*1000:.1f} mm  "
              f"track rms {S['track_rms_median']*1000:.2f} mrad  u_rms {S['u_rms_median']:.1f} N m")
    out = c.ROOT / "results" / "summary"; out.mkdir(parents=True, exist_ok=True)
    with open(out / f"sweep_{a.name}.json", "w") as f:
        json.dump([{"level": n_, **S} for n_, S in results], f, indent=2)
    figs = c.ROOT / "docs" / "figures"; figs.mkdir(parents=True, exist_ok=True)
    plots.plot_sweep(results, a.name.replace("_", " "), figs / f"sweep_{a.name}.png")
    if sweep.get("tradeoff"):
        plots.plot_tradeoff(results, a.name.replace("_", " "), figs / f"tradeoff_{a.name}.png")


if __name__ == "__main__":
    main()
