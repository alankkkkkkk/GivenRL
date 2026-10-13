"""Single source of truth for every projected number in the paper.

All values below are PROJECTED targets (no training run has been performed).
Derived quantities quoted in the text are computed here and checked.
"""
import json
from decimal import Decimal, ROUND_HALF_UP

BENCH = ["AIME24", "AIME25", "AIME26", "HMMT26", "AMC23", "MATH500", "Olympiad", "Minerva"]
METHODS = ["Base", "GT-GRPO", "Random", "RENT", "Intuitor", "TTRL", "Co-rewarding", "EVOL-RL",
           "Self-Harmony", "RESTRAIN", "OM-GRPO", "CoVerRL", "GivenRL"]
LABEL_FREE = ["RENT", "Intuitor", "TTRL", "Co-rewarding", "EVOL-RL", "Self-Harmony", "RESTRAIN",
              "OM-GRPO", "CoVerRL"]

MAIN = {
    "Qwen3.5-4B": {
        "Base":         [12.4, 9.8, 8.1, 3.9, 41.2, 68.5, 34.6, 30.1],
        "GT-GRPO":      [26.3, 21.5, 18.4, 9.7, 62.0, 82.4, 49.8, 41.2],
        "Random":       [13.8, 10.8, 9.5, 4.3, 45.0, 70.8, 36.7, 32.1],
        "RENT":         [14.8, 12.3, 9.9, 4.9, 49.8, 74.8, 40.4, 34.5],
        "Intuitor":     [16.7, 13.8, 11.3, 5.6, 52.6, 74.8, 42.7, 35.4],
        "TTRL":         [17.2, 13.8, 11.8, 6.3, 54.3, 77.8, 44.8, 37.3],
        "Co-rewarding": [18.9, 14.7, 12.4, 6.4, 56.1, 78.1, 45.0, 38.0],
        "EVOL-RL":      [21.0, 16.8, 14.8, 7.6, 55.6, 78.6, 45.6, 38.5],
        "Self-Harmony": [20.8, 16.3, 14.6, 7.0, 56.8, 79.4, 45.6, 38.5],
        "RESTRAIN":     [20.5, 17.4, 14.9, 7.6, 58.2, 79.1, 46.8, 38.9],
        "OM-GRPO":      [21.9, 17.9, 15.4, 8.1, 58.9, 80.6, 46.9, 40.4],
        "CoVerRL":      [22.3, 18.6, 15.6, 7.8, 58.2, 80.3, 46.5, 39.3],
        "GivenRL":      [25.2, 20.6, 17.6, 9.4, 60.4, 81.6, 48.6, 40.1],
    },
    "Olmo-3-7B": {
        "Base":         [5.6, 4.2, 3.5, 1.6, 27.9, 55.3, 24.8, 22.7],
        "GT-GRPO":      [15.2, 11.8, 9.6, 4.4, 45.1, 72.6, 38.3, 33.1],
        "Random":       [5.2, 4.2, 3.5, 1.7, 28.8, 56.3, 24.8, 22.8],
        "RENT":         [7.6, 6.2, 5.1, 2.1, 34.6, 62.1, 30.1, 27.1],
        "Intuitor":     [8.6, 6.3, 5.0, 2.4, 36.3, 64.1, 32.2, 28.0],
        "TTRL":         [9.3, 7.2, 5.9, 2.5, 39.9, 67.2, 34.2, 29.8],
        "Co-rewarding": [9.9, 7.6, 6.1, 2.9, 39.2, 66.7, 33.9, 29.6],
        "EVOL-RL":      [11.4, 8.6, 7.0, 3.2, 39.6, 67.5, 33.9, 30.5],
        "Self-Harmony": [11.3, 8.3, 6.9, 3.2, 40.7, 67.8, 35.5, 31.0],
        "RESTRAIN":     [11.6, 9.0, 7.1, 3.3, 41.0, 68.4, 35.8, 30.3],
        "OM-GRPO":      [11.5, 9.6, 7.5, 3.3, 41.7, 68.3, 35.6, 31.4],
        "CoVerRL":      [12.7, 9.7, 7.6, 3.5, 41.6, 70.1, 36.0, 31.5],
        "GivenRL":      [14.4, 11.2, 9.3, 4.2, 44.0, 71.9, 37.4, 31.9],
    },
    "Qwen3.5-9B": {
        "Base":         [17.9, 14.6, 12.2, 6.3, 50.4, 74.1, 40.3, 35.3],
        "GT-GRPO":      [33.8, 27.9, 23.6, 13.1, 70.5, 86.9, 55.7, 46.4],
        "Random":       [19.4, 16.2, 13.4, 6.8, 53.1, 76.1, 42.7, 37.4],
        "RENT":         [22.1, 17.5, 15.2, 8.1, 59.8, 79.3, 46.3, 39.6],
        "Intuitor":     [22.2, 18.2, 15.8, 8.6, 61.1, 80.5, 48.2, 41.1],
        "TTRL":         [23.3, 19.9, 17.0, 9.1, 64.2, 82.5, 50.0, 42.9],
        "Co-rewarding": [24.9, 21.1, 18.0, 9.4, 64.3, 83.6, 51.4, 42.8],
        "EVOL-RL":      [27.2, 22.4, 19.7, 10.7, 64.2, 83.7, 52.1, 43.5],
        "Self-Harmony": [26.9, 22.4, 18.4, 9.9, 66.6, 84.0, 52.0, 44.1],
        "RESTRAIN":     [27.8, 23.5, 19.8, 10.4, 65.6, 83.8, 51.9, 44.1],
        "OM-GRPO":      [28.2, 23.6, 19.6, 11.2, 67.4, 85.0, 53.4, 46.0],
        "CoVerRL":      [28.9, 24.5, 20.2, 11.1, 67.1, 84.1, 53.0, 44.2],
        "GivenRL":      [32.0, 26.6, 22.4, 12.6, 68.9, 86.1, 54.3, 47.2],
    },
}
# seed std of the eight-benchmark average (3 training seeds); Base = std over 3 evaluation seeds
SEED_STD = {
    "Qwen3.5-4B": {"Base": 0.11, "GT-GRPO": 0.24, "Random": 0.29, "RENT": 0.31, "Intuitor": 0.27, "TTRL": 0.30,
                   "Co-rewarding": 0.28, "EVOL-RL": 0.26, "Self-Harmony": 0.29, "RESTRAIN": 0.25, "OM-GRPO": 0.27,
                   "CoVerRL": 0.28, "GivenRL": 0.23},
    "Olmo-3-7B": {"Base": 0.09, "GT-GRPO": 0.27, "Random": 0.22, "RENT": 0.31, "Intuitor": 0.30, "TTRL": 0.29,
                  "Co-rewarding": 0.31, "EVOL-RL": 0.28, "Self-Harmony": 0.30, "RESTRAIN": 0.27, "OM-GRPO": 0.29,
                  "CoVerRL": 0.30, "GivenRL": 0.26},
    "Qwen3.5-9B": {"Base": 0.10, "GT-GRPO": 0.21, "Random": 0.26, "RENT": 0.28, "Intuitor": 0.25, "TTRL": 0.27,
                   "Co-rewarding": 0.26, "EVOL-RL": 0.24, "Self-Harmony": 0.27, "RESTRAIN": 0.23, "OM-GRPO": 0.25,
                   "CoVerRL": 0.26, "GivenRL": 0.22},
}

OOD_COLS = ["GPQA-D", "MMLU-Pro", "LCB v6", "CRUX-I", "CRUX-O"]
OOD = {  # Qwen3.5-4B
    "Base":    [28.3, 47.5, 14.2, 38.4, 41.0],
    "GT-GRPO": [33.1, 50.2, 18.6, 44.7, 46.9],
    "TTRL":    [30.2, 48.6, 16.1, 40.3, 44.5],
    "EVOL-RL": [31.8, 49.4, 17.0, 41.1, 45.2],
    "OM-GRPO": [31.9, 49.5, 17.2, 41.6, 45.8],
    "CoVerRL": [31.5, 49.3, 16.8, 42.4, 45.6],
    "GivenRL": [32.6, 49.8, 17.6, 46.9, 45.9],
    "GivenRL w/o inverse": [32.1, 49.6, 17.3, 42.5, 45.7],
}
# pass@64: solved counts out of 30 (AIME26) and 33 (HMMT26)
PASS64 = {"Base": (10, 6), "GT-GRPO": (13, 9), "TTRL": (9, 5), "EVOL-RL": (12, 8),
          "OM-GRPO": (11, 7), "CoVerRL": (11, 7), "GivenRL": (13, 9)}

# Test-time training (avg@16), Qwen3.5-4B
TTT = {"Base": (8.1, 3.9), "TTRL": (11.6, 5.1), "EVOL-RL": (13.0, 5.5), "Self-Harmony": (13.2, 5.7),
       "OM-GRPO": (13.9, 5.9), "GivenRL": (15.8, 7.2)}
TTT_STD = {"Base": (0.0, 0.0), "TTRL": (1.4, 0.9), "EVOL-RL": (1.2, 0.8), "Self-Harmony": (1.3, 0.9),
           "OM-GRPO": (1.1, 0.8), "GivenRL": (1.0, 0.9)}

# Diagnostics on DAPO-Math training problems (Qwen3.5-4B, 16 base samples per problem)
INIT_SPLIT = {"conf_wrong": 14, "conf_right": 61, "uncertain": 25}
DIAG = {  # rank acc (step0, final, conf-wrong final), fixed conf-wrong %, fixed uncertain %, broken conf-right %
    "TTRL":         (0.61, 0.66, 0.00, 9, 48, 3.1),
    "Co-rewarding": (0.65, 0.69, 0.09, 13, 52, 2.6),
    "Self-Harmony": (0.68, 0.72, 0.14, 17, 55, 2.2),
    "OM-GRPO":      (0.63, 0.70, 0.05, 12, 57, 1.8),
    "CoVerRL":      (0.70, 0.79, 0.21, 22, 58, 1.9),
    "GivenRL":      (0.80, 0.88, 0.71, 36, 62, 1.3),
    "GT-GRPO":      (1.00, 1.00, 1.00, 47, 66, 0.9),
}
# GivenRL fix rate on confidently-wrong problems split by anchor informativeness
ANCHOR_SPLIT = {"share_informative": 0.58, "fix_informative": 50, "fix_other": 17,
                "gt_fix_informative": 51, "gt_fix_other": 41}

COVERAGE = {"eligible": 83, "median_givens": 3, "informative_of_eligible": 71,
            "no_eligible": 17, "low_gap": 24, "mean_candidates": 2.4,
            "parser_precision": 91, "template_wellposed": 88, "groups_fully_covered": 96}

GAP = {  # held-out 512 problems: (alpha, beta, control) at start and end
    "start": (0.58, 0.21, 0.12),
    "GivenRL": (0.81, 0.19, 0.09),
    "no-inverse": (0.60, 0.23, 0.12),
    "no-gate": (0.79, 0.31, 0.13),
    "all-candidates": (0.78, 0.63, 0.41),
    "frozen": (0.58, 0.21, 0.12),
}
INFORMATIVE_SHARE = {"start": 71, "end": 84}
GATE_PURITY = 95
GATE_PASS = {"correct_init": 0.56, "wrong_init": 0.03, "correct_end": 0.94, "wrong_end": 0.03}  # exact at pilot/end rates  # % of gradient inverse groups whose conditioning answer is correct (projected)

LONG = {  # MATH-500 at epoch 2, 4 and 6
    "GT-GRPO": (82.4, 82.9, 83.1), "GivenRL": (81.6, 82.1, 82.4), "EVOL-RL": (78.6, 78.8, 78.9), "TTRL": (77.8, 75.1, 71.2)}
LEN_CHANGE = {"GT-GRPO": +6, "GivenRL": +4, "EVOL-RL": -3, "TTRL": -38}  # % response length, epoch 2 -> 6
REWARD_STD_RATIO = {"GT-GRPO": 0.71, "GivenRL": 0.83, "EVOL-RL": 0.61, "TTRL": 0.17}  # epoch-6 / initial
TTRL_LEN_DROP = 38

COST = {"step_time_vs_ttrl": 1.80, "inverse_rollouts": 27, "inverse_len_frac": 0.40}
MATCHED = [  # (method, setting, GPU-hours relative to GivenRL, avg)
    ("TTRL", "$G{=}8$", 0.56, 32.9), ("TTRL", "$G{=}16$", 1.07, 33.4), ("OM-GRPO", "$G{=}16$", 1.09, 36.6),
    ("CoVerRL", "2$\\times$ verification samples", 1.04, 36.4), ("Co-RL", "peer: Olmo-3-7B", 1.50, 36.4),
    ("JURY-RL", "32B formalizer + prover", 2.40, 37.1), ("GivenRL", "default", 1.00, 37.9)]

ABLATION = [  # (group, name, avg, fixed conf-wrong, final Delta)
    ("", "GivenRL (full)", 37.9, 36, 0.62),
    ("Reward", "Uniform weights $w_j{=}1$", 37.6, 34, 0.61),
    ("Reward", "No $\\tau$ filter", 37.3, 33, 0.60),
    ("Reward", "Vote fallback below $\\tau$", 37.3, 30, 0.60),
    ("Reward", "Top-2 candidates only", 37.1, 29, 0.61),
    ("Inverse", "No inverse term", 36.4, 30, 0.37),
    ("Inverse", "Frozen inverse solver", 36.1, 29, 0.37),
    ("Inverse", "Margin rule, no control", 37.0, 32, 0.48),
    ("Inverse", "Train all candidates", 35.0, 23, 0.15),
    ("Construction", "Qwen3.5-27B rewrites", 38.0, 36, 0.63),
]
BUDGET = [  # (K_max, J, M, avg, step time vs TTRL)
    (2, 2, 4, 37.3, 1.68), (4, 1, 4, 37.0, 1.40), (4, 2, 2, 36.9, 1.40), (4, 2, 4, 37.9, 1.80),
    (4, 2, 8, 38.1, 2.60), (4, 4, 4, 38.2, 2.60), (8, 2, 4, 37.9, 1.85)]
COMBINE = {"+ EVOL-RL novelty": 38.2, "+ OM-GRPO masking": 38.3}
LAMBDA_RANGE = (37.5, 37.9)

PILOT = {  # recovery rate (%) of masked given, base model, 512 problems
    "majority right": {"correct": 59, "wrong": 20, "control": 12},
    "majority wrong": {"correct": 54, "wrong": 23, "control": 11},
    "all": {"correct": 58, "wrong": 21, "control": 12},
}


def r1(x):
    return float(Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def avg(v):
    return sum(v) / len(v)


def derived():
    d = {}
    for bb, rows in MAIN.items():
        a = {m: avg(v) for m, v in rows.items()}
        ar = {m: r1(round(x, 6)) for m, x in a.items()}
        best_lf = max(LABEL_FREE, key=lambda m: a[m])
        comp = {m: avg(v[:4]) for m, v in rows.items()}
        best_lf_comp = max(LABEL_FREE, key=lambda m: comp[m])
        d[bb] = dict(
            avg=ar, best_lf=best_lf,
            lead=r1(ar["GivenRL"] - ar[best_lf]),
            gap_gt=r1(ar["GT-GRPO"] - ar["GivenRL"]),
            gain_base=r1(ar["GivenRL"] - ar["Base"]),
            gain_random=r1(ar["GivenRL"] - ar["Random"]),
            closed=(a["GivenRL"] - a[best_lf]) / (a["GT-GRPO"] - a[best_lf]),
            comp={m: r1(round(x, 6)) for m, x in comp.items()}, best_lf_comp=best_lf_comp,
            closed_comp=(comp["GivenRL"] - comp[best_lf_comp]) / (comp["GT-GRPO"] - comp[best_lf_comp]),
            random_lift=r1(ar["Random"] - ar["Base"]),
        )
        # GivenRL best label-free on every column?
        d[bb]["givenrl_best_every_col"] = all(
            rows["GivenRL"][i] >= max(rows[m][i] for m in LABEL_FREE) for i in range(8))
        # exact .x5 ambiguities
        d[bb]["ambiguous"] = [m for m, x in a.items() if abs(x * 10 - round(x * 10) - 0.5) < 1e-6
                              or abs((x * 10) % 1 - 0.5) < 1e-6]
    # coverage arithmetic
    d["low_gap_check"] = COVERAGE["eligible"] * (100 - COVERAGE["informative_of_eligible"]) / 100
    d["informative_total"] = COVERAGE["eligible"] * COVERAGE["informative_of_eligible"] / 100
    s = ANCHOR_SPLIT
    d["anchor_split_recombined"] = s["share_informative"] * s["fix_informative"] + (1 - s["share_informative"]) * s["fix_other"]
    d["gt_split_recombined"] = s["share_informative"] * s["gt_fix_informative"] + (1 - s["share_informative"]) * s["gt_fix_other"]
    d["gap_deltas"] = {k: r1(v[0] - v[1]) for k, v in GAP.items()}
    d["ood_gain"] = {m: [r1(x - y) for x, y in zip(v, OOD["Base"])] for m, v in OOD.items()}
    d["pass64_pct"] = {m: (round(100 * a / 30, 1), round(100 * b / 33, 1)) for m, (a, b) in PASS64.items()}
    d["jm_ordering"] = sorted(BUDGET, key=lambda r: r[1] * r[2])
    return d


if __name__ == "__main__":
    d = derived()
    for bb in MAIN:
        x = d[bb]
        print(bb, "avg:", x["avg"])
        print("  best LF", x["best_lf"], "lead", x["lead"], "gap GT", x["gap_gt"], "gain base", x["gain_base"],
              "gain rand", x["gain_random"], "closed %.2f" % x["closed"], "random lift", x["random_lift"])
        print("  comp", x["comp"]["GivenRL"], "best comp", x["best_lf_comp"], x["comp"][x["best_lf_comp"]],
              "GT", x["comp"]["GT-GRPO"], "closed_comp %.2f" % x["closed_comp"])
        print("  GivenRL best LF on every column:", x["givenrl_best_every_col"], "ambiguous", x["ambiguous"])
    for k in ("low_gap_check", "informative_total", "anchor_split_recombined", "gt_split_recombined", "gap_deltas", "ood_gain", "pass64_pct"):
        print(k, d[k])
    json.dump({k: v for k, v in d.items() if k != "jm_ordering"}, open("derived.json", "w"), indent=1, default=str)
