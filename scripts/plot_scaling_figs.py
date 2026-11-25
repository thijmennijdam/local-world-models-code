"""Plot the paper's scaling figures (model scale, data scale) from the windowed evaluation outputs.

Metric: error at the last step of the 6-step horizon (cm), averaged over the expert and interaction
test sets, separately for seen and unseen objects; averaged over every evaluation summary found for
a model name (e.g. several window seeds).

Spec (YAML, see configs/figures/*.yaml):
  title: "6-step prediction error"
  xlabel: "Parameters"
  out: figures/lwm_arch_combined.pdf
  eval_dirs: [outputs/lwm_eval/model_size_ablation_large_seed42, ...]   # searched for summaries
  points:                       # one entry per checkpoint; model = model_names entry of the eval config
    - {x: "26M",  series: line, model: "d512-L8 / 110k"}
    - {x: "204M", series: star, model: "d1024-L16 / 220k"}
  star_label: "2$\\times$ interaction data"

Usage: python scripts/plot_scaling_figs.py configs/figures/model_scale.yaml [--eval-dir DIR ...]
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D

SOURCES = ("expert", "new_config")
CATS = {"seen": "Seen Objects", "unseen": "Unseen Objects"}
COLORS = {"seen": "#1f77b4", "unseen": "#d62728"}


def summaries(eval_dirs, model):
    """All evaluation_summary json files (newest per model directory) whose model_name matches."""
    out = []
    for d in eval_dirs:
        for md in glob.glob(os.path.join(d, "*/")):
            files = sorted(glob.glob(os.path.join(md, "evaluation_summary_*.json")), key=os.path.getmtime)
            if files:
                s = json.load(open(files[-1]))
                if s["model_name"] == model:
                    out.append(s)
    if not out:
        raise FileNotFoundError(f"no evaluation summary for model {model!r} under {eval_dirs}")
    return out


def step6(eval_dirs, model):
    """{seen|unseen: (aff_cm, ee_cm)} at the last horizon step, mean over sources and summaries."""
    out = {}
    for cat, name in CATS.items():
        aff, ee = [], []
        for s in summaries(eval_dirs, model):
            w = s["windowed_eval"]
            aff.append(np.mean([w[src][name]["avg_aff_drift_cm"][-1] for src in SOURCES]))
            ee.append(np.mean([w[src][name]["avg_agent_drift_cm"][-1] for src in SOURCES]))
        out[cat] = (float(np.mean(aff)), float(np.mean(ee)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec")
    ap.add_argument("--eval-dir", action="append", default=[], help="adds to / replaces the spec's eval_dirs")
    a = ap.parse_args()
    spec = yaml.safe_load(open(a.spec))
    eval_dirs = a.eval_dir or spec.get("eval_dirs", [])
    pts = [dict(p, vals=step6(eval_dirs, p["model"])) for p in spec["points"]]
    xs = list(dict.fromkeys(p["x"] for p in pts))
    plt.rcParams.update({"font.family": "serif", "font.size": 13})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, (m, name) in zip(axes, [(0, "Affordance"), (1, "End-effector")]):
        for cat in CATS:
            line = [(xs.index(p["x"]), p["vals"][cat][m]) for p in pts if p["series"] == "line"]
            if line:
                ax.plot(*zip(*line), "-o", color=COLORS[cat], mec="black", ms=8, lw=2)
            star = [(xs.index(p["x"]), p["vals"][cat][m]) for p in pts if p["series"] == "star"]
            if star:
                ax.plot(*zip(*star), "*", color=COLORS[cat], mec="black", ms=18, ls="none")
        ax.set_xticks(range(len(xs)), xs)
        ax.set_title(name, fontweight="bold")
        ax.grid(alpha=0.4)
    axes[0].set_ylabel("Prediction error (cm)")
    fig.supxlabel(spec.get("xlabel", ""))
    fig.suptitle(spec.get("title", "6-step prediction error"), fontweight="bold")
    handles = [Line2D([], [], color=COLORS["seen"], marker="o", mec="black", label="Seen objects"),
               Line2D([], [], color=COLORS["unseen"], marker="o", mec="black", label="Unseen objects")]
    if any(p["series"] == "star" for p in pts):
        handles.append(Line2D([], [], color="black", marker="*", ms=14, ls="none", label=spec.get("star_label", "")))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    os.makedirs(os.path.dirname(os.path.abspath(spec["out"])), exist_ok=True)
    fig.savefig(spec["out"], bbox_inches="tight")
    print(json.dumps({p["x"] + ("*" if p["series"] == "star" else ""): p["vals"] for p in pts}, indent=1))
    print(f"wrote {spec['out']}")


if __name__ == "__main__":
    main()
