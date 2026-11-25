"""Aggregate the windowed evaluation (scripts/eval_scaling.sh) into one table: the error at the last
step of the 6-step horizon in cm (handle point and end-effector), mean +- std over the window seeds.

Usage: python scripts/eval_table.py <eval root, e.g. outputs/lwm_eval> <config name> [seeds...]
Reads <root>/<config>_seed<seed>/*/evaluation_summary_*.json (default seeds: 42 1 2) and, if only
<root>/<config>/ exists (a single seed without the suffix), that directory.
"""
import collections
import glob
import os
import sys

import json
import numpy as np


def main(root, cfg, seeds):
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    dirs = [f"{root}/{cfg}_seed{s}" for s in seeds]
    if os.path.isdir(f"{root}/{cfg}"):
        dirs.append(f"{root}/{cfg}")
    for d in dirs:
        for md in glob.glob(d + "/*/"):
            files = sorted(glob.glob(md + "evaluation_summary_*.json"), key=os.path.getmtime)
            if not files:
                continue
            s = json.load(open(files[-1]))
            m, w = s["model_name"], s["windowed_eval"]
            for g in ("new_config", "expert"):
                for sp in ("Unseen Objects", "Seen Objects"):
                    res[m][(g, sp, "aff")].append(w[g][sp]["avg_aff_drift_cm"][-1])
                    res[m][(g, sp, "agent")].append(w[g][sp]["avg_agent_drift_cm"][-1])
                    if w[g][sp].get("avg_finger_err_mm"):
                        res[m][(g, sp, "finger")].append(w[g][sp]["avg_finger_err_mm"][-1])
            res[m]["n"].append((w["new_config"]["Unseen Objects"]["num_windows"],
                                w["new_config"]["Seen Objects"]["num_windows"],
                                w["expert"]["Unseen Objects"]["num_windows"]))
    cols = [("new_config", "Unseen Objects", "aff"), ("new_config", "Seen Objects", "aff"),
            ("expert", "Unseen Objects", "aff"), ("expert", "Seen Objects", "aff"),
            ("new_config", "Unseen Objects", "agent"), ("expert", "Unseen Objects", "agent")]
    hdr = ["int-unseen aff", "int-seen aff", "exp-unseen aff", "exp-seen aff", "int-unseen EE", "exp-unseen EE"]
    print(f"{'step-6 error in cm, mean+-std over seeds':40s} " + " ".join(f"{h:>14s}" for h in hdr)
          + "  seeds  windows(int-u, int-s, exp-u)")
    for m, r in res.items():
        fk = ("new_config", "Unseen Objects", "finger")
        finger = f"  finger int-unseen {np.mean(r[fk]):.3f} mm" if r.get(fk) else ""
        print(f"{m:40s} " + " ".join(f"{np.mean(r[c]):7.3f}+-{np.std(r[c]):.3f}" for c in cols)
              + f"  {len(r['n']):3d}    {r['n'][0]}" + finger)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3:] or ["42", "1", "2"])
