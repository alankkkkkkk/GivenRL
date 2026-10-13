"""Exact computations for the two-outcome analysis (Section 3.5, Appendix A).

Everything here is computed from the model equations; no training data is involved.
"""
import itertools
import json
import math
from functools import lru_cache

import numpy as np
from scipy.stats import binom


# ---------------------------------------------------------------------------
# GRPO two-outcome dynamics
# ---------------------------------------------------------------------------
def grpo_factor(p, G=8):
    """E_C[ sqrt(C(G-C))/G ], C ~ Bin(G, p)."""
    c = np.arange(G + 1)
    return float(np.sum(binom.pmf(c, G, p) * np.sqrt(c * (G - c)) / G))


def field_ground_truth(p, G=8):
    return p * (1 - p) * grpo_factor(p, G)


def field_majority(p, G=8):
    """Self majority vote: direction sign(C - G/2), ties give no update."""
    c = np.arange(G + 1)
    w = binom.pmf(c, G, p) * np.sqrt(c * (G - c)) / G * np.sign(c - G / 2)
    return p * (1 - p) * float(np.sum(w))


def field_peer(p, q, G=8):
    """Agent with correct-prob p rewarded by a peer's majority (peer prob q)."""
    cq = np.arange(G + 1)
    direction = float(np.sum(binom.pmf(cq, G, q) * np.sign(cq - G / 2)))
    return p * (1 - p) * grpo_factor(p, G) * direction


# ---------------------------------------------------------------------------
# Anchored score: exact psi
# ---------------------------------------------------------------------------
def psi_uniform(alpha, beta, n):
    """psi = P(X>Y) - P(X<Y), X~Bin(n,alpha), Y~Bin(n,beta)."""
    k = np.arange(n + 1)
    px, py = binom.pmf(k, n, alpha), binom.pmf(k, n, beta)
    cdf_y = np.cumsum(py)
    p_gt = float(np.sum(px[1:] * cdf_y[:-1]))
    p_eq = float(np.sum(px * py))
    p_lt = 1 - p_gt - p_eq
    return p_gt - p_lt, p_gt, p_eq, p_lt


def psi_weighted(alphas, betas, controls, M=4, tau=0.5):
    """Exact psi for the GivenRL score with range weights and the tau filter.

    alphas[j], betas[j], controls[j]: recovery prob. of given j under the correct
    answer, the wrong answer and the control answer.  Returns (psi, P(filtered)).
    """
    J = len(alphas)
    k = np.arange(M + 1)
    pa = [binom.pmf(k, M, a) for a in alphas]
    pb = [binom.pmf(k, M, b) for b in betas]
    pc = [binom.pmf(k, M, c) for c in controls]
    plus = minus = filt = 0.0
    for sa in itertools.product(range(M + 1), repeat=J):
        wa = np.prod([pa[j][sa[j]] for j in range(J)])
        for sb in itertools.product(range(M + 1), repeat=J):
            wb = np.prod([pb[j][sb[j]] for j in range(J)])
            for sc in itertools.product(range(M + 1), repeat=J):
                wc = np.prod([pc[j][sc[j]] for j in range(J)])
                prob = wa * wb * wc
                w = [(max(sa[j], sb[j], sc[j]) - min(sa[j], sb[j], sc[j])) / M for j in range(J)]
                if sum(w) < tau:
                    filt += prob
                    continue
                d = sum(w[j] * (sa[j] - sb[j]) for j in range(J))
                if d > 1e-12:
                    plus += prob
                elif d < -1e-12:
                    minus += prob
    return plus - minus, plus, minus, filt


def sign_error_exact(alpha, beta, n):
    """P(score(correct) <= score(wrong)) with ties counted as errors (uniform weights)."""
    _, p_gt, _, _ = psi_uniform(alpha, beta, n)
    return 1 - p_gt


def hoeffding(delta, n):
    return math.exp(-n * delta ** 2 / 2)


# ---------------------------------------------------------------------------
# Consistency gate
# ---------------------------------------------------------------------------
def gate_pass(lead, control, M=4, m=0.5):
    """P(s_hat(lead) - s_hat(control) >= m) for one given."""
    k = np.arange(M + 1)
    pl, pc = binom.pmf(k, M, lead), binom.pmf(k, M, control)
    need = math.ceil(m * M - 1e-9)
    return float(sum(pl[i] * pc[j] for i in k for j in k if i - j >= need))


if __name__ == "__main__":
    out = {}
    # pilot-level recovery rates used throughout (projected pilot values)
    A, B, Cc = 0.58, 0.21, 0.12
    n = 8
    psi_u, gt, eq, lt = psi_uniform(A, B, n)
    out["psi_uniform_JM8"] = psi_u
    out["rank_acc_uniform_JM8_ties_half"] = gt + eq / 2
    out["sign_error_JM8"] = 1 - gt
    out["hoeffding_JM8"] = hoeffding(A - B, n)
    for nn in (4, 8, 12, 16, 32):
        out[f"psi_uniform_JM{nn}"] = psi_uniform(A, B, nn)[0]
        out[f"hoeffding_JM{nn}"] = hoeffding(A - B, nn)
        out[f"sign_error_JM{nn}"] = sign_error_exact(A, B, nn)
    psi_w, plus, minus, filt = psi_weighted([A, A], [B, B], [Cc, Cc])
    out["psi_weighted_J2M4"] = psi_w
    out["psi_weighted_J2M4_conditional"] = psi_w / (1 - filt)
    out["p_filtered_J2M4"] = filt
    out["rank_acc_weighted_kept"] = (plus + 0.5 * (1 - filt - plus - minus)) / (1 - filt)
    # end-of-training gap
    out["psi_weighted_end"] = psi_weighted([0.81, 0.81], [0.19, 0.19], [0.09, 0.09])[0]
    # gate
    out["gate_correct_leader"] = gate_pass(A, Cc)
    out["gate_wrong_leader"] = gate_pass(B, Cc)
    out["gate_correct_leader_end"] = gate_pass(0.81, 0.09)
    out["gate_wrong_leader_end"] = gate_pass(0.19, 0.09)
    # grid check: psi_weighted > 0 whenever alpha > beta (both givens)
    worst = 1.0
    for a in np.linspace(0.05, 0.95, 10):
        for b in np.linspace(0.05, 0.95, 10):
            if a <= b + 1e-9:
                continue
            for c in (0.05, 0.3, 0.6):
                v = psi_weighted([a, a], [b, b], [c, c])[0]
                worst = min(worst, v)
    out["grid_min_psi_weighted"] = worst
    # mixed per-given gaps: one strong, one null given
    out["psi_weighted_one_null_given"] = psi_weighted([A, 0.5], [B, 0.5], [Cc, 0.5])[0]
    for k_, v in out.items():
        print(f"{k_:40s} {v:.4f}")
    json.dump(out, open("theory_values.json", "w"), indent=1)
