"""Table 1 (SR@0.4 / SR@0.8, seen / unseen; mean +- std over the per-object success rates) from the
per-episode ``metrics.json`` files written by ``scripts/mpc_table1.py``.

Usage: python scripts/table1_from_raw.py <root with one sub-directory per variant> [variants...]
         [--first-n N] [--json out.json]
  root        e.g. outputs/table1
  --first-n   only the first N episodes per object (sorted by name), e.g. 15 for the paper's rows
  --json      also write the rows (for scripts/table1_latex.py); default <root>/table1.json
"""
import argparse
import glob
import json
import os

import numpy as np

from lwm.planning.success import SEEN_OBJECTS, fraction_of_goal


def rows_for(root, variant, first_n=0):
    per = {}
    files = sorted(glob.glob(f"{root}/{variant}/*/*/metrics.json"))
    if first_n:
        by_obj = {}
        for f in files:
            by_obj.setdefault(f.split("/")[-3], []).append(f)
        files = [f for fs in by_obj.values() for f in fs[:first_n]]
    for f in files:
        per.setdefault(f.split("/")[-3], []).append(fraction_of_goal(json.load(open(f))))
    out = []
    for th in (0.4, 0.8):
        for split in ("seen", "unseen"):
            objs = [o for o in per if (o in SEEN_OBJECTS) == (split == "seen")]
            srs = [100 * np.mean(np.array(per[o]) >= th) for o in objs]
            out.append((float(np.mean(srs)), float(np.std(srs)), len(srs), sum(len(per[o]) for o in objs)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("variants", nargs="*")
    ap.add_argument("--first-n", type=int, default=0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    variants = a.variants or sorted(d for d in os.listdir(a.root) if os.path.isdir(os.path.join(a.root, d)))
    rows = {}
    for v in variants:
        rows[v] = rows_for(a.root, v, a.first_n)
        cells = "  ".join(f"{m:5.1f}+-{s:4.1f}" for m, s, _, _ in rows[v])
        n = "/".join(f"{k}obj,{e}ep" for _, _, k, e in rows[v][:2])
        print(f"{v:30s} SR@0.4 seen unseen | SR@0.8 seen unseen: {cells}   [{n}]")
    out = a.json or os.path.join(a.root, "table1.json")
    json.dump(rows, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
