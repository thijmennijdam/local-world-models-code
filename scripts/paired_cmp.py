"""Paired comparison of two MPC rollout roots on the same episodes (same object and episode name).

Prints SR@thr per split (mean +- std over objects) for both roots and the paired per-object
difference (cmp - ref) with a bootstrap 95% CI over objects (* = CI excludes 0). Use it to compare
two models or planner settings run with ``scripts/mpc_table1.py`` on the same start states.

Usage: python scripts/paired_cmp.py --ref outputs/table1 --cmp outputs/table1_seed2 \
           --variants LWM-200M LWM-200M_contact LWM-200M_contact_kinematics [--n 15] [--thr 0.8]
"""
import argparse
import glob
import json

import numpy as np

from lwm.planning.success import SEEN_OBJECTS, fraction_of_goal


def load(root, variant, n):
    out = {}
    for o in sorted(glob.glob(f"{root}/{variant}/[0-9]*")):
        for e in sorted(glob.glob(o + "/*/metrics.json"))[:n]:
            out[(o.split("/")[-1], e.split("/")[-2])] = fraction_of_goal(json.load(open(e)))
    return out


def sr_table(d, thr):
    per = {}
    for (o, e), v in d.items():
        per.setdefault(o, []).append(v >= thr)
    s = [100 * np.mean(per[o]) for o in per if o in SEEN_OBJECTS]
    u = [100 * np.mean(per[o]) for o in per if o not in SEEN_OBJECTS]
    return (np.mean(s), np.std(s)), (np.mean(u), np.std(u))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cmp", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--thr", type=float, default=0.8)
    ap.add_argument("--bootstrap", type=int, default=4000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    print(f"SR@{a.thr} (%), first {a.n} episodes per object; ref={a.ref} cmp={a.cmp}")
    for v in a.variants:
        b, c = load(a.ref, v, a.n), load(a.cmp, v, a.n)
        ks = sorted(set(b) & set(c))
        if not ks:
            print(f"{v:28s} no common episodes")
            continue
        bb = {k: b[k] for k in ks}
        cc = {k: c[k] for k in ks}
        (rs, rss), (ru, rus) = sr_table(bb, a.thr)
        (cs, css), (cu, cus) = sr_table(cc, a.thr)
        row = []
        for split in ("seen", "unseen", "all"):
            per = {}
            for (o, e) in ks:
                if split == "all" or (o in SEEN_OBJECTS) == (split == "seen"):
                    per.setdefault(o, []).append((cc[(o, e)] >= a.thr) - (bb[(o, e)] >= a.thr))
            d = np.array([100 * np.mean(x) for x in per.values()])
            bs = [rng.choice(d, len(d)).mean() for _ in range(a.bootstrap)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            row.append(f"{split} {d.mean():+5.1f} [{lo:+5.1f},{hi:+5.1f}]{'*' if lo > 0 or hi < 0 else ' '}")
        print(f"{v:28s} n={len(ks):3d}  ref {rs:5.1f}+-{rss:4.1f} / {ru:5.1f}+-{rus:4.1f}   "
              f"cmp {cs:5.1f}+-{css:4.1f} / {cu:5.1f}+-{cus:4.1f}   paired: " + "  ".join(row))


if __name__ == "__main__":
    main()
