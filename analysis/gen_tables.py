"""Generate every results table of the paper from paper_numbers.py."""
import os
import paper_numbers as P

OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "tables")
os.makedirs(OUT, exist_ok=True)

NAME = {"Base": "Base model", "GT-GRPO": "GRPO, ground truth", "Random": "GRPO, random reward",
        "RENT": "RENT", "Intuitor": "Intuitor", "TTRL": "TTRL", "Co-rewarding": "Co-rewarding",
        "EVOL-RL": "EVOL-RL", "Self-Harmony": "Self-Harmony", "RESTRAIN": "RESTRAIN", "OM-GRPO": "OM-GRPO",
        "CoVerRL": "CoVerRL", "GivenRL": "\\textbf{GivenRL} (ours)", "GivenRL w/o inverse": "\\quad w/o inverse"}
REF = ["Base", "GT-GRPO", "Random"]


def f1(x):
    return f"{x:.1f}"


def avg(v):
    return sum(v) / len(v)


def r1(x):
    return P.r1(round(x, 6))


def best_lf(values_by_method, methods):
    return max(values_by_method[m] for m in methods)


def cell(val, best, is_lf):
    s = f1(val)
    if is_lf and abs(val - best) < 1e-9:
        s = "\\textbf{" + s + "}"
    return s


def main_table():
    rows4 = P.MAIN["Qwen3.5-4B"]
    lf = P.LABEL_FREE + ["GivenRL"]
    avgs = {bb: {m: r1(avg(v)) for m, v in P.MAIN[bb].items()} for bb in P.MAIN}
    lines = []
    for m in P.METHODS:
        is_lf = m in lf
        cells = []
        for i in range(8):
            col = {mm: rows4[mm][i] for mm in lf}
            cells.append(cell(rows4[m][i], max(col.values()), is_lf))
        for bb in ["Qwen3.5-4B", "Olmo-3-7B", "Qwen3.5-9B"]:
            col = {mm: avgs[bb][mm] for mm in lf}
            cells.append(cell(avgs[bb][m], max(col.values()), is_lf))
        prefix = "\\rowcolor{rowours}" if m == "GivenRL" else ""
        name = NAME[m]
        if m in REF[1:]:
            name = "\\textcolor{refgray}{" + name + "}"
            cells = ["\\textcolor{refgray}{" + c + "}" for c in cells]
        lines.append(prefix + name + " & " + " & ".join(cells[:8]) + " & " + cells[8] + " & " + cells[9] + " & " + cells[10] + " \\\\")
        if m == "Random":
            lines.append("\\midrule")
            lines.append("\\multicolumn{12}{@{}l}{\\textit{Label-free methods}} \\\\")
        if m == "CoVerRL":
            lines.append("\\midrule")
    open(os.path.join(OUT, "main.tex"), "w").write("\n".join(lines) + "\n")


def appendix_main():
    lf = P.LABEL_FREE + ["GivenRL"]
    out = []
    for bb in ["Olmo-3-7B", "Qwen3.5-9B"]:
        rows = P.MAIN[bb]
        out.append("\\midrule")
        out.append("\\multicolumn{10}{@{}l}{\\textit{" + bb + ("-Base" if bb.startswith("Qwen") else "") + "}} \\\\")
        for m in P.METHODS:
            is_lf = m in lf
            cells = []
            for i in range(8):
                col = {mm: rows[mm][i] for mm in lf}
                cells.append(cell(rows[m][i], max(col.values()), is_lf))
            a = {mm: r1(avg(v)) for mm, v in rows.items()}
            cells.append(cell(a[m], max(a[mm] for mm in lf), is_lf))
            prefix = "\\rowcolor{rowours}" if m == "GivenRL" else ""
            name = NAME[m]
            if m in REF[1:]:
                name = "\\textcolor{refgray}{" + name + "}"
                cells = ["\\textcolor{refgray}{" + c + "}" for c in cells]
            out.append(prefix + name + " & " + " & ".join(cells) + " \\\\")
    open(os.path.join(OUT, "main_appendix.tex"), "w").write("\n".join(out) + "\n")


def ood_table():
    order = ["Base", "GT-GRPO", "TTRL", "EVOL-RL", "OM-GRPO", "CoVerRL", "GivenRL", "GivenRL w/o inverse"]
    lf = ["TTRL", "EVOL-RL", "OM-GRPO", "CoVerRL", "GivenRL", "GivenRL w/o inverse"]
    lines = []
    for m in order:
        v = P.OOD[m]
        a, h = P.PASS64[m] if m in P.PASS64 else (None, None)
        cells = []
        for i in range(5):
            best = max(P.OOD[mm][i] for mm in lf)
            cells.append(cell(v[i], best, m in lf and m != "GivenRL w/o inverse"))
        if a is not None:
            pa, ph = 100 * a / 30, 100 * h / 33
            besta = max(100 * P.PASS64[mm][0] / 30 for mm in lf if mm in P.PASS64)
            besth = max(100 * P.PASS64[mm][1] / 33 for mm in lf if mm in P.PASS64)
            cells.append(cell(round(pa, 1), round(besta, 1), m in lf))
            cells.append(cell(round(ph, 1), round(besth, 1), m in lf))
        else:
            cells += ["--", "--"]
        name = NAME[m] if m != "GivenRL" else "\\textbf{GivenRL}"
        if m == "Base":
            name = "Base"
        prefix = "\\rowcolor{rowours}" if m == "GivenRL" else ""
        if m == "GT-GRPO":
            name = "\\textcolor{refgray}{GRPO, GT}"
            cells = ["\\textcolor{refgray}{" + c + "}" for c in cells]
        lines.append(prefix + name + " & " + " & ".join(cells) + " \\\\")
        if m == "GT-GRPO":
            lines.append("\\midrule")
    open(os.path.join(OUT, "ood.tex"), "w").write("\n".join(lines) + "\n")


def ablation_table():
    lines = []
    cur = None
    for grp, name, a, fixed, d in P.ABLATION:
        if grp != cur:
            if grp:
                lines.append("\\midrule")
                lines.append("\\multicolumn{4}{@{}l}{\\textit{" + grp + "}} \\\\")
            cur = grp
        full = P.ABLATION[0][2]
        delta = "" if grp == "" else f" \\dlt{{{a - full:+.1f}}}"
        prefix = "\\rowcolor{rowours}" if grp == "" else ""
        lines.append(prefix + ("" if grp == "" else "\\quad ") + name + f" & {a:.1f}{delta} & {fixed} & {d:.2f} \\\\")
    open(os.path.join(OUT, "ablation.tex"), "w").write("\n".join(lines) + "\n")


def diag_table():
    order = ["TTRL", "Co-rewarding", "Self-Harmony", "OM-GRPO", "CoVerRL", "GivenRL", "GT-GRPO"]
    lines = []
    for m in order:
        r0, rf, rcw, fcw, fun, brk = P.DIAG[m]
        name = NAME[m] if m not in ("GivenRL", "GT-GRPO") else ("\\textbf{GivenRL}" if m == "GivenRL" else "\\textcolor{refgray}{GRPO, GT}")
        cells = [f"{fcw}", f"{fun}", f"{brk:.1f}", f"{r0:.2f}", f"{rf:.2f}", f"{rcw:.2f}"]
        if m == "GT-GRPO":
            cells = ["\\textcolor{refgray}{" + c + "}" for c in cells[:3]] + ["\\textcolor{refgray}{1.00}"] * 3
            lines.append("\\midrule")
        if m == "GivenRL":
            cells = ["\\textbf{" + c + "}" for c in cells]
        prefix = "\\rowcolor{rowours}" if m == "GivenRL" else ""
        lines.append(prefix + name + " & " + " & ".join(cells) + " \\\\")
    open(os.path.join(OUT, "diag.tex"), "w").write("\n".join(lines) + "\n")


def seed_table():
    lines = []
    for m in P.METHODS:
        name = NAME[m]
        lines.append(name + " & " + " & ".join(f"{P.SEED_STD[bb][m]:.2f}" for bb in ["Qwen3.5-4B", "Olmo-3-7B", "Qwen3.5-9B"]) + " \\\\")
        if m == "Random":
            lines.append("\\midrule")
    open(os.path.join(OUT, "seeds.tex"), "w").write("\n".join(lines) + "\n")


def budget_table():
    lines = []
    for k, j, mm, a, t in P.BUDGET:
        prefix = "\\rowcolor{rowours}" if (k, j, mm) == (4, 2, 4) else ""
        lines.append(prefix + f"{k} & {j} & {mm} & {j*mm} & {a:.1f} & {t:.2f}$\\times$ \\\\")
    open(os.path.join(OUT, "budget.tex"), "w").write("\n".join(lines) + "\n")


def matched_table():
    lines = []
    for m, setting, g, a in P.MATCHED:
        prefix = "\\rowcolor{rowours}" if m == "GivenRL" else ""
        lines.append(prefix + f"{m} & {setting} & {g:.2f} & {a:.1f} \\\\")
    open(os.path.join(OUT, "matched.tex"), "w").write("\n".join(lines) + "\n")


def ttt_long_table():
    lines = []
    for m in ["Base", "TTRL", "EVOL-RL", "Self-Harmony", "OM-GRPO", "GivenRL"]:
        a, h = P.TTT[m]
        sa, sh = P.TTT_STD[m]
        prefix = "\\rowcolor{rowours}" if m == "GivenRL" else ""
        if m == "Base":
            lines.append(prefix + f"Base model & {a:.1f} & {h:.1f} \\\\")
        else:
            lines.append(prefix + f"{m} & {a:.1f}\\,\\std{{{sa:.1f}}} & {h:.1f}\\,\\std{{{sh:.1f}}} \\\\")
    open(os.path.join(OUT, "ttt.tex"), "w").write("\n".join(lines) + "\n")
    lines = []
    for m in ["GT-GRPO", "TTRL", "EVOL-RL", "GivenRL"]:
        e2, e4, e6 = P.LONG[m]
        nm = "GRPO, GT" if m == "GT-GRPO" else m
        prefix = "\\rowcolor{rowours}" if m == "GivenRL" else ""
        lines.append(prefix + f"{nm} & {e2:.1f} & {e4:.1f} & {e6:.1f} & {P.LEN_CHANGE[m]:+d}\\% & {P.REWARD_STD_RATIO[m]:.2f} \\\\")
    open(os.path.join(OUT, "long.tex"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main_table(); appendix_main(); ood_table(); ablation_table(); diag_table(); seed_table()
    budget_table(); matched_table(); ttt_long_table()
    for f in sorted(os.listdir(OUT)):
        print("==", f); print(open(os.path.join(OUT, f)).read())
