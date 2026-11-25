"""Print the Table 1 body rows in the paper's LaTeX format from ``scripts/table1_from_raw.py`` output.
Bold = best mean per column. Usage: python scripts/table1_latex.py outputs/table1/table1.json"""
import json
import sys

ORDER = [("LWM-100M", "", ""), ("LWM-100M_contact", r"$\checkmark$", ""),
         ("LWM-100M_contact_kinematics", r"$\checkmark$", r"$\checkmark$"), None,
         ("LWM-200M", "", ""), ("LWM-200M_contact", r"$\checkmark$", ""),
         ("LWM-200M_contact_kinematics", r"$\checkmark$", r"$\checkmark$")]


def main(path):
    rows = json.load(open(path))
    best = [max(rows[v][j][0] for v in rows) for j in range(4)]
    for item in ORDER:
        if item is None:
            print(r"            \midrule")
            continue
        v, c, k = item
        if v not in rows:
            print(f"            % {v}: missing")
            continue
        cells = [rf"$\mathbf{{{m:.1f}}}_{{\pm{s:.1f}}}$" if abs(m - best[j]) < 1e-9 else rf"${m:.1f}_{{\pm{s:.1f}}}$"
                 for j, (m, s, _, _) in enumerate(rows[v])]
        print(f"            {v.split('_')[0]:17s} & {c:12s} & {k:12s} & " + " & ".join(cells) + r" \\")
    print("% episodes per variant: " + json.dumps({v: rows[v][0][3] + rows[v][1][3] for v in rows}))


if __name__ == "__main__":
    main(sys.argv[1])
