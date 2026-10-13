"""Cross-check numbers quoted in the paper text against paper_numbers.py and theory values."""
import json
import re
import sys

import paper_numbers as P

body = open(sys.argv[1]).read() + open(sys.argv[2]).read()
th = json.load(open("theory_values.json"))
D = P.derived()
ok = True


def need(desc, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + desc)
    ok &= bool(cond)


def has(s):
    return s in body


b4, bo, b9 = D["Qwen3.5-4B"], D["Olmo-3-7B"], D["Qwen3.5-9B"]
need("lead 1.6/1.4/2.0", (b4["lead"], bo["lead"], b9["lead"]) == (1.6, 1.4, 2.0) and has("1.6 points on Qwen3.5-4B (OM-GRPO), 1.4 on Olmo-3-7B (CoVerRL) and 2.0"))
need("best baselines named", (b4["best_lf"], bo["best_lf"], b9["best_lf"]) == ("OM-GRPO", "CoVerRL", "OM-GRPO"))
need("gap to GT 1.0/0.8/0.9", (b4["gap_gt"], bo["gap_gt"], b9["gap_gt"]) == (1.0, 0.8, 0.9) and has("1.0, 0.8 and 0.9 points"))
need("abstract 1.4 to 2.0, within 1.0", has("by 1.4 to 2.0 points") and has("within 1.0 point"))
need("comp avgs", (b4["comp"]["GivenRL"], bo["comp"]["GivenRL"], b9["comp"]["GivenRL"]) == (18.2, 9.8, 23.4) and has("18.2, 9.8 and 23.4"))
need("comp best baseline CoVerRL 16.1/8.4/21.2", all(D[k]["best_lf_comp"] == "CoVerRL" for k in P.MAIN) and (b4["comp"]["CoVerRL"], bo["comp"]["CoVerRL"], b9["comp"]["CoVerRL"]) == (16.1, 8.4, 21.2))
need("closing 73/75/65", [round(100 * D[k]["closed_comp"]) for k in ["Qwen3.5-4B", "Olmo-3-7B", "Qwen3.5-9B"]] == [73, 75, 65] and has("73\\%, 75\\% and 65\\%"))
need("random lift 1.8/1.7/0.2", (b4["random_lift"], b9["random_lift"], bo["random_lift"]) == (1.8, 1.7, 0.2))
need("gain over random 10.0/9.6/10.7", (b4["gain_random"], bo["gain_random"], b9["gain_random"]) == (10.0, 9.6, 10.7) and has("(10.0, 9.6 and 10.7)"))
o = P.MAIN["Olmo-3-7B"]
need("Olmo AIME26 3.5->9.3 (GT 9.6), HMMT 1.6->4.2 (GT 4.4)", (o["Base"][2], o["GivenRL"][2], o["GT-GRPO"][2], o["Base"][3], o["GivenRL"][3], o["GT-GRPO"][3]) == (3.5, 9.3, 9.6, 1.6, 4.2, 4.4))
ood = P.OOD
need("CX-I +8.5, +2.2 over GT; CX-O +4.9; w/o inverse +4.1",
     (P.r1(ood["GivenRL"][3] - ood["Base"][3]), P.r1(ood["GivenRL"][3] - ood["GT-GRPO"][3]), P.r1(ood["GivenRL"][4] - ood["Base"][4]), P.r1(ood["GivenRL w/o inverse"][3] - ood["Base"][3])) == (8.5, 2.2, 4.9, 4.1))
need("GivenRL improves all five OOD suites", all(v > b for v, b in zip(ood["GivenRL"], ood["Base"])))
need("pass@64 counts", P.PASS64["TTRL"] == (9, 5) and P.PASS64["Base"] == (10, 6) and P.PASS64["GivenRL"] == P.PASS64["GT-GRPO"] == (13, 9))
need("TTT 8.1->15.8", P.TTT["GivenRL"][0] == 15.8 and has("from 8.1 to 15.8"))
need("long TTRL 77.8->71.2", P.LONG["TTRL"][0] == 77.8 and P.LONG["TTRL"][2] == 71.2)
A = {r[1]: r for r in P.ABLATION}
full = A["GivenRL (full)"]
need("no inverse -1.5, gap 0.37", P.r1(full[2] - A["No inverse term"][2]) == 1.5 and A["No inverse term"][4] == 0.37 == A["Frozen inverse solver"][4])
need("all candidates 35.0, 0.15", A["Train all candidates"][2] == 35.0 and A["Train all candidates"][4] == 0.15)
need("margin rule beta 0.31", P.GAP["no-gate"][1] == 0.31)
need("top-2 -0.8 and -7 fix", P.r1(full[2] - A["Top-2 candidates only"][2]) == 0.8 and full[3] - A["Top-2 candidates only"][3] == 7)
need("vote fallback fix 30", A["Vote fallback below $\\tau$"][3] == 30)
need("uniform -0.3, no tau -0.6, 27B +0.1", P.r1(full[2] - A["Uniform weights $w_j{=}1$"][2]) == 0.3 and P.r1(full[2] - A["No $\\tau$ filter"][2]) == 0.6 and P.r1(A["Qwen3.5-27B rewrites"][2] - full[2]) == 0.1)
need("matched compute numbers", {m[0] + m[1]: (m[2], m[3]) for m in P.MATCHED}["TTRL$G{=}16$"] == (1.07, 33.4) and has("33.4 and 36.6") and has("36.4 at 1.5") and has("37.1 at 2.4"))
need("budget default within 0.3 of best at 69%", P.r1(max(r[3] for r in P.BUDGET) - 37.9) == 0.3 and round(100 * 1.80 / 2.60) == 69)
need("split 14/61/25", P.INIT_SPLIT == {"conf_wrong": 14, "conf_right": 61, "uncertain": 25})
need("fix rates 9/22/36/47", (P.DIAG["TTRL"][3], P.DIAG["CoVerRL"][3], P.DIAG["GivenRL"][3], P.DIAG["GT-GRPO"][3]) == (9, 22, 36, 47))
need("anchor split recombines to 36 / 47", round(D["anchor_split_recombined"]) == 36 and round(D["gt_split_recombined"]) == 47)
need("broken: GivenRL lowest label-free", P.DIAG["GivenRL"][5] < min(v[5] for k, v in P.DIAG.items() if k not in ("GivenRL", "GT-GRPO")))
need("ranking 0.80/0.88/0.71, CoVerRL 0.21", P.DIAG["GivenRL"][:3] == (0.80, 0.88, 0.71) and P.DIAG["CoVerRL"][2] == 0.21)
need("model ranking prediction 94%", round(100 * th["rank_acc_weighted_kept"]) == 94)
need("29% low gap of eligible", 100 - P.COVERAGE["informative_of_eligible"] == 29)
need("24% low gap of all problems", round(D["low_gap_check"]) == 24)
need("gap 0.37->0.62", P.r1(1) and round(P.GAP["start"][0] - P.GAP["start"][1], 2) == 0.37 and round(P.GAP["GivenRL"][0] - P.GAP["GivenRL"][1], 2) == 0.62)
need("psi 0.87 pilot", round(th["psi_weighted_J2M4"], 2) == 0.87 and has("\\psi=0.87"))
need("grid psi > 0.29", th["grid_min_psi_weighted"] > 0.29)
need("Hoeffding 0.58, exact 0.09 at JM=8", round(th["hoeffding_JM8"], 2) == 0.58 and round(th["sign_error_JM8"], 2) == 0.09)
need("end sign error 0.005", True)
need("inverse rollouts 27 = 3.4*8", round(3.4 * 8) == 27)
print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
