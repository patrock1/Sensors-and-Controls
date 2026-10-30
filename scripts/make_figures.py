"""Regenerate every figure used in the README / report into docs/figures/.

    python scripts/make_figures.py
Run scripts/run_sweep.py first if you also want the sweep figures refreshed (they are written there).
"""
import _common as c
from ur3pick.pipeline import load_cfg, run_trial
from ur3pick.evaluation import plots


def main(seed=0):
    out = c.ROOT / "docs" / "figures"; out.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg()
    plots.plot_trial(run_trial(seed, cfg), str(out / "nominal_trial.png"))
    print("wrote", out / "nominal_trial.png")


if __name__ == "__main__":
    main()
