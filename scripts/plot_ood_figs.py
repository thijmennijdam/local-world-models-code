"""Appendix figures of the no-bias failure mode (approach, then retreat) from Table 1 rollouts.

Picks a failed no-bias episode of <model> on a seen drawer that approaches the handle and then
retreats (closest approach < 3 cm, final end-effector-to-handle distance > 3x the closest),
preferring the largest retreat; plots its distances and planning cost, and overlays the 1-step
prediction errors of the same episode under the contact and contact + kinematics biases. Also
prints how common approach-then-retreat is over all no-bias episodes. Needs the trajectories
written by scripts/mpc_table1.py (logging.save_trajectories=true, the default there).

Usage: python scripts/plot_ood_figs.py <rollout root, e.g. outputs/table1> <model, e.g. LWM-200M> <outdir>
"""
import glob
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lwm.planning.success import SEEN_OBJECTS

SEEN_DRAWERS = ("44817", "44853", "45092", "45132")


def load(ep_dir):
    t = pickle.load(open(os.path.join(ep_dir, "trajectory.pkl"), "rb"))["trajectory"]
    m = json.load(open(os.path.join(ep_dir, "metrics.json")))
    return t, m


def smooth(x, k=5):
    x = np.asarray(x, float)
    return np.convolve(np.pad(x, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def retreat_stats(runs, model):
    rows = []
    for ep in glob.glob(f"{runs}/{model}/*/*/trajectory.pkl"):
        t, _ = load(os.path.dirname(ep))
        d = np.asarray(t["ee_to_aff_distances"]) * 100
        if len(d) < 10:
            continue
        rows.append((ep.split("/")[-3] in SEEN_OBJECTS, d.min(), d[-1], int(d.argmin())))
    rows = np.array(rows, dtype=float)
    if not len(rows):
        return "no episodes"
    retreat = (rows[:, 1] < 3) & (rows[:, 2] > 3 * np.maximum(rows[:, 1], 1))
    return (f"{model} no-bias: {len(rows)} episodes; closest approach < 3 cm in {np.mean(rows[:, 1] < 3):.0%}; "
            f"approach-then-retreat in {retreat.mean():.0%} (seen {retreat[rows[:, 0] == 1].mean():.0%}, "
            f"unseen {retreat[rows[:, 0] == 0].mean():.0%}); median final EE-aff distance {np.median(rows[:, 2]):.1f} cm")


def pick_episode(runs, model):
    best = None
    for obj in SEEN_DRAWERS:
        for ep_dir in sorted(glob.glob(f"{runs}/{model}/{obj}/*/")):
            if not os.path.exists(os.path.join(ep_dir, "trajectory.pkl")):
                continue
            t, m = load(ep_dir)
            d = np.asarray(t["ee_to_aff_distances"]) * 100
            if len(d) < 30 or d.min() > 3 or d[-1] < 3 * max(d.min(), 1):
                continue
            score = d[-1] - d.min()
            if best is None or score > best[0]:
                best = (score, ep_dir)
    return best[1] if best else None


def main(runs, model, outdir):
    os.makedirs(outdir, exist_ok=True)
    print(retreat_stats(runs, model))
    ep_dir = pick_episode(runs, model)
    if ep_dir is None:
        print("no approach-then-retreat episode on a seen drawer")
        return
    rel = os.path.relpath(ep_dir, f"{runs}/{model}")
    t, m = load(ep_dir)
    d = np.asarray(t["ee_to_aff_distances"]) * 100
    g = np.asarray(t["goal_distances"]) * 100
    c = np.asarray(t["costs"])
    i = int(d.argmin())
    print(f"episode {rel}: closest {d.min():.1f} cm at step {i}, final {d[-1]:.1f} cm, steps {len(d)}")
    plt.rcParams.update({"font.family": "serif", "font.size": 14})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(d, lw=2, color="#0072bd", label="EE-to-affordance")
    ax[0].plot(g, "--", lw=2, color="#ed7d31", label="Goal distance")
    ax[0].axvline(i, color="gray", ls=":", lw=1)
    ax[0].annotate(f"closest approach\n({d.min():.1f} cm, step {i})", (i, d.min()), (i + 12, d.max() * 0.55),
                   arrowprops=dict(arrowstyle="->", color="gray"), fontsize=10, ha="left")
    ax[0].set(title="(a)  Trajectory distances", xlabel="Environment step", ylabel="Distance (cm)", ylim=(0, None))
    ax[1].plot(c, lw=2, color="#009e73", label="Planning horizon cost")
    ax[1].axvline(i, color="gray", ls=":", lw=1)
    ax[1].set(title="(b)  MPC planning cost", xlabel="Environment step", ylabel="Cost (12-step horizon)", ylim=(0, None))
    for a in ax:
        a.legend(fontsize=10, loc="upper left")
        a.grid(True, color="#dddddd")
        a.set_xlim(0, len(d) - 1)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"ood_mpc_failure.{ext}"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for variant, lab, col in ((model, "No bias", "#d03c1f"), (f"{model}_contact", "Contact bias", "#ed7d31"),
                              (f"{model}_contact_kinematics", "Contact + kinematics bias", "#009e73")):
        p = os.path.join(runs, variant, rel)
        if not os.path.exists(os.path.join(p, "trajectory.pkl")):
            print(f"missing {variant}/{rel}")
            continue
        tv, mv = load(p)
        print(f"  {variant}: steps {len(tv['prediction_errors_aff'])}, final fraction {mv.get('normalized_performance')}")
        for a, key in zip(ax, ("prediction_errors_aff", "prediction_errors_ee")):
            a.plot(smooth(np.asarray(tv[key]) * 100), lw=2, color=col, label=lab)
    ax[0].set(title="(a)  Affordance point", ylabel="Affordance prediction error (cm)")
    ax[1].set(title="(b)  End-effector state", ylabel="End-effector prediction error (cm)")
    for a in ax:
        a.set(xlabel="Environment step", ylim=(0, None))
        a.grid(True, color="#dddddd")
    ax[0].legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"ood_prediction_errors.{ext}"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:4])
