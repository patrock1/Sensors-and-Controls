import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PHASE_COLOURS = {"search": "#dddddd", "approach": "#cfe2f3", "descend": "#d9ead3",
                 "grasp": "#fff2cc", "lift": "#f4cccc", "done": "#d9d2e9", "failed": "#ea9999"}


def _shade_phases(ax, t, phase):
    start = 0
    for k in range(1, len(phase) + 1):
        if k == len(phase) or phase[k] != phase[start]:
            ax.axvspan(t[start], t[k - 1], color=PHASE_COLOURS.get(phase[start], "white"), lw=0, zorder=0)
            start = k 

def plot_trial(log, path):
    t = np.asarray(log["t"])
    phase = np.asarray(log["phase"]).astype(str)
    fig, axs = plt.subplots(5, 1, figsize=(9, 13.5), sharex=True)

    ax = axs[0]
    e = np.asarray(log["p_hat"]) - np.asarray(log["gt_p_obj"])
    sig = np.sqrt(np.stack([np.asarray(log["P_p"])[:, i, i] for i in range(3)], axis=1))
    for i, (name, c) in enumerate(zip("xyz", ["C0", "C1", "C2"])):
        ax.plot(t, 1000 * e[:, i], c, label=f"{name} error")
        ax.fill_between(t, -2000 * sig[:, i], 2000 * sig[:, i], color=c, alpha=0.15, lw=0)
    ax.set_ylabel("object error (mm)")
    ax.set_ylim(-10, 10)
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.set_title("Object estimate error with +-2 sigma (shaded)")

    # 2. joint estimate error
    ax = axs[1]
    eq = np.asarray(log["x_hat"])[:, :6] - np.asarray(log["gt_q"])
    for j in range(6):
        ax.plot(t, 1000 * eq[:, j], lw=0.8, label=f"q{j + 1}")
    ax.set_ylabel("joint error (mrad)")
    ax.legend(loc="upper right", fontsize=8, ncol=6)
    ax.set_title("Joint observer error")

    # 3. torques
    ax = axs[2]
    u = np.asarray(log["u"])
    for j in range(6):
        ax.plot(t, u[:, j], lw=0.8, label=f"u{j + 1}")
    ax.set_ylabel("torque (N m)")
    ax.legend(loc="upper right", fontsize=8, ncol=6)
    ax.set_title("Commanded torques")

    ax = axs[3]
    if "x_ref" in log:
        et = np.asarray(log["x_ref"])[:, :6] - np.asarray(log["gt_q"])
        for j in range(6):
            ax.plot(t, 1000 * et[:, j], lw=0.8, label=f"q{j + 1}")
        ax.legend(loc="upper right", fontsize=8, ncol=6)
    ax.set_ylabel("tracking error (mrad)")
    ax.set_title("Joint tracking error, q_ref - q_true")


    ax = axs[4]
    _shade_phases(ax, t, phase)
    ax.plot(t, np.asarray(log["gt_p_obj"])[:, 2], "r", label="cube centre z")
    if "gt_p_tool" in log:
        ax.plot(t, np.asarray(log["gt_p_tool"])[:, 2], "k", label="tool0 z")
    ax.set_ylabel("height (m)")
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper left", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PHASE_COLOURS[p]) for p in PHASE_COLOURS if p in set(phase)]
    ax2 = ax.twinx(); ax2.set_yticks([])
    ax2.legend(handles, [p for p in PHASE_COLOURS if p in set(phase)], loc="upper right", fontsize=8, ncol=4)
    ax.set_title(f"Pick: final phase '{phase[-1]}'")   
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_sweep(results, title, path):
    names = [n for n, _ in results]
    S = [s for _, s in results]
    x = np.arange(len(names))
    rate = np.array([s["success_rate"] for s in S])
    lo = rate - np.array([s["ci_low"] for s in S])
    hi = np.array([s["ci_high"] for s in S]) - rate
    err = 1000 * np.array([s["obj_err_freeze_median"] for s in S])

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(max(6, 1.3 * len(names)), 7), sharex=True)
    a1.bar(x, 100 * rate, yerr=[100 * lo, 100 * hi], capsize=4, color="C0")
    a1.set_ylabel("success (%)"); a1.set_ylim(0, 105)
    a1.set_title(f"Sweep: {title} (n = {S[0]['n']} per level, 95 % Wilson CI)")
    a2.bar(x, err, color="C1")
    a2.set_ylabel("median object error\nat freeze (mm)")
    a2.set_xticks(x); a2.set_xticklabels(names, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

def plot_tradeoff(results, title, path, x="u_rms_median", y="track_rms_median",
                  xlabel="control effort, median u_rms (N m)", ylabel="tracking error, median RMS (mrad)",
                  x_scale=1.0, y_scale=1000.0, x_ref=None, x_ref_label=None, x_log=False):
    """One point per sweep level: summary key `x` against summary key `y`, labelled with the level
    name and its success rate. Defaults draw Part D effort vs tracking; the sweep yaml can name other
    keys (Part C: NEES vs estimate error). x_ref draws a vertical reference line, e.g. NEES = 12."""
    names = [n for n, _ in results]
    S = [s for _, s in results]
    xs = x_scale * np.array([s[x] for s in S])
    ys = y_scale * np.array([s[y] for s in S])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, ys, "-o", color="C0")
    for xv, yv, name, s in zip(xs, ys, names, S):
        ax.annotate(f"{name}\n{s['success_rate']:.0%} success", (xv, yv), textcoords="offset points",
                    xytext=(6, 6), fontsize=8)
    if x_ref is not None:
        ax.axvline(x_ref, color="k", ls="--", lw=1, label=x_ref_label)
        if x_ref_label:
            ax.legend(loc="best", fontsize=8)
    if x_log:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}" if str(v)[0] in "25" else ""))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Trade-off: {title} (n = {S[0]['n']} per level)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
