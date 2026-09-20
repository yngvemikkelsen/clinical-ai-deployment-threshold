#!/usr/bin/env python3
"""
paper9_boundary_sweep.py
========================

Editorial item 8 — the zero-boundary classification rule.

The letter states that classifying a single configuration (E5-Mistral-7B-ablation)
changes six-condition specificity from 0.996 to 0.936 and moves the screening
requirement roughly tenfold, and asks the authors to address the methodological
risk of categorising continuous technical outcomes at a strict zero boundary.

This script characterises that boundary rather than reporting a single pair. It
sweeps the classification rule

    configuration is assigned to the harmed group if its mean dMRR@10 < tau

across a window of tau spanning zero, and at each step recomputes group
membership, group means, group dispersion and the deployment threshold p*.

Reads:  appendix5_epsilon_percondition.csv   (13 configurations x 6 conditions)
Writes: boundary_item08_sweep.csv
        boundary_item08_flips.csv
        boundary_item08_summary.txt

Usage:
    python3 paper9_boundary_sweep.py
    python3 paper9_boundary_sweep.py --csv /path/to/appendix5_epsilon_percondition.csv

Requires numpy only (statistics from the standard library).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics as st
import sys
from statistics import NormalDist

# ─────────────────────────────────────────────────────────────────────────────
# Base case, Appendix 5 (response-defined grouping, v1.1.0)
# ─────────────────────────────────────────────────────────────────────────────

EPSILON = "1e-05"               # base-case regularisation
D_BEN_EXPECTED = 0.075745
D_HARM_EXPECTED = -0.076593
N_BEN_EXPECTED = 6
N_HARM_EXPECTED = 7
BOUNDARY_CONFIG = "E5-Mistral-7B-ablation"

# Screening derivation (recovered from the Experiment A screen-curve script).
#   P(sign error) = Phi(-|d_m| / (s_m / sqrt(n)))
#   se = 1 - mean P(error) over the benefited group
#   sp = 1 - mean P(error) over the harmed group
#   requirement = [(1-sp)(|d_harm|+t) + s] / [se(d_ben-t) + (1-sp)(|d_harm|+t)]
#   with s = S/M and t = T/M (exact form; see the manuscript Methods)
# This reproduces the se/sp arrays hardcoded in make_figures.py figure2() to
# four decimals at every n from 1 to 6, and reproduces both figures the editor
# quotes in item 8 (sp 0.996 -> 0.936, requirement 0.0066 -> 0.0712).
N_COND_MAX = 6
PUB_SE = [0.687, 0.742, 0.777, 0.804, 0.825, 0.843]
PUB_SP = [0.819, 0.885, 0.912, 0.925, 0.932, 0.936]

# Strategy cost terms, for p* including cost (K/M is ~4.3e-06, so p* is
# effectively the zero-cost form; carried here for completeness).


TOL = 1e-6

# Sweep window for tau, in dMRR@10 units. Zero is the submitted rule.
TAU_MIN, TAU_MAX, TAU_STEP = -0.05, 0.05, 0.0025   # fine enough to show every flip

DEFAULT_CSV = "appendix5_epsilon_percondition.csv"


def load(path: str, epsilon: str = EPSILON) -> dict:
    """Per-configuration mean and SD of dMRR@10 across scored conditions."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["epsilon"] == epsilon]
    if not rows:
        raise SystemExit(f"No rows at epsilon={epsilon} in {path}")

    per = {}
    for m in sorted({r["model"] for r in rows}):
        v = [float(r["dMRR@10"]) for r in rows if r["model"] == m]
        per[m] = {
            "values": v,
            "n_cond": len(v),
            "mean": st.mean(v),
            "sd": st.stdev(v) if len(v) > 1 else 0.0,
        }
    return per


_ND = NormalDist()


def screen_accuracy(per: dict, benefited, harmed, n_cond: int):
    """se, sp at n_cond scored conditions, by normal approximation."""
    def err(group):
        return [_ND.cdf(-abs(per[m]["mean"]) / (per[m]["sd"] / math.sqrt(n_cond)))
                for m in group if per[m]["sd"] > 0]
    e_ben, e_harm = err(benefited), err(harmed)
    se = 1 - sum(e_ben) / len(e_ben) if e_ben else float("nan")
    sp = 1 - sum(e_harm) / len(e_harm) if e_harm else float("nan")
    return se, sp


# Strategy cost terms for the exact screening requirement (Methods): S is the
# fixed screening cost, incurred whichever way the screen reads; T is the
# deployment cost, incurred only on the systems the screen selects. Folding T
# into a single strategy cost would require assuming a value for p.
import importlib.util as _ilu
_S = _ilu.spec_from_file_location(
    "model", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "clinical_ai_deployment_model.py"))
MODEL = _ilu.module_from_spec(_S)
_S.loader.exec_module(MODEL)
M_BASE = MODEL.value_exposed()      # value exposed to the intervention, EUR
T_DEPLOY = MODEL.T_DEPLOY           # deployment cost, EUR
S_PER_COND = MODEL.S_PER_CONDITION  # fixed screening cost per condition, EUR


def screen_requirement(se, sp, d_ben, d_harm, n_cond=N_COND_MAX):
    """Minimum prior probability at which screen-and-treat has positive net effect.

    Exact form with the fixed screening cost and the conditional deployment
    cost carried separately, as in the manuscript Methods.
    """
    t = T_DEPLOY / M_BASE
    s = (S_PER_COND * n_cond) / M_BASE
    b = (1 - sp) * (abs(d_harm) + t)
    den = se * (d_ben - t) + b
    return (b + s) / den if den > 0 else float("nan")


def classify(per: dict, tau: float) -> dict:
    """Split configurations at the boundary tau and summarise both groups."""
    harmed = [m for m in per if per[m]["mean"] < tau]
    benefited = [m for m in per if per[m]["mean"] >= tau]

    def summarise(group):
        if not group:
            return {"n": 0, "mean": float("nan"), "mean_sd": float("nan")}
        return {
            "n": len(group),
            "mean": st.mean([per[m]["mean"] for m in group]),
            "mean_sd": st.mean([per[m]["sd"] for m in group]),
        }

    b, h = summarise(benefited), summarise(harmed)

    if b["n"] == 0 or h["n"] == 0:
        p_star = float("nan")
    else:
        s = b["mean"] - h["mean"]
        p_star = (abs(h["mean"]) + T_DEPLOY / M_BASE) / s if s > 0 else float("nan")

    se6, sp6 = (screen_accuracy(per, benefited, harmed, N_COND_MAX)
                if b["n"] and h["n"] else (float("nan"), float("nan")))
    req6 = (screen_requirement(se6, sp6, b["mean"], h["mean"])
            if b["n"] and h["n"] else float("nan"))

    return {
        "tau": tau,
        "se_n6": se6, "sp_n6": sp6, "screen_requirement_n6": req6,
        "n_benefited": b["n"], "n_harmed": h["n"],
        "d_ben": b["mean"], "d_harm": h["mean"],
        "sd_benefited": b["mean_sd"], "sd_harmed": h["mean_sd"],
        "p_star": p_star,
        "boundary_config_group": ("harmed"
                                  if BOUNDARY_CONFIG in harmed else "benefited"),
        "harmed_members": "|".join(sorted(harmed)),
    }


def verify(per: dict, log) -> None:
    base = classify(per, 0.0)
    log("BASE CASE CHECK (tau = 0, the submitted rule)")
    log(f"  configurations              {len(per)}")
    log(f"  conditions per config       "
        f"{sorted({v['n_cond'] for v in per.values()})}")
    log(f"  benefited                   n={base['n_benefited']}  "
        f"mean {base['d_ben']:+.6f}")
    log(f"  harmed                      n={base['n_harmed']}  "
        f"mean {base['d_harm']:+.6f}")
    log(f"  p*                          {base['p_star']:.6f}")

    errs = []
    if base["n_benefited"] != N_BEN_EXPECTED:
        errs.append(f"benefited n={base['n_benefited']}, "
                    f"expected {N_BEN_EXPECTED}")
    if base["n_harmed"] != N_HARM_EXPECTED:
        errs.append(f"harmed n={base['n_harmed']}, expected {N_HARM_EXPECTED}")
    if abs(base["d_ben"] - D_BEN_EXPECTED) > TOL:
        errs.append(f"d_ben {base['d_ben']:.6f}, expected {D_BEN_EXPECTED}")
    if abs(base["d_harm"] - D_HARM_EXPECTED) > TOL:
        errs.append(f"d_harm {base['d_harm']:.6f}, expected {D_HARM_EXPECTED}")
    if BOUNDARY_CONFIG not in per:
        errs.append(f"{BOUNDARY_CONFIG} not present in the file")

    if errs:
        log("")
        log("BASE CASE CHECK FAILED:")
        for e in errs:
            log(f"  - {e}")
        sys.exit(1)
    ben = [m for m in per if per[m]["mean"] >= 0]
    harm = [m for m in per if per[m]["mean"] < 0]
    bad = []
    for n in range(1, N_COND_MAX + 1):
        se, sp = screen_accuracy(per, ben, harm, n)
        if abs(se - PUB_SE[n - 1]) > 5e-4 or abs(sp - PUB_SP[n - 1]) > 5e-4:
            bad.append(f"n={n}: se {se:.4f} vs {PUB_SE[n-1]}, "
                       f"sp {sp:.4f} vs {PUB_SP[n-1]}")
    if bad:
        log("")
        log("SCREEN DERIVATION CHECK FAILED:")
        for e in bad:
            log(f"  - {e}")
        sys.exit(1)
    log("  screen se/sp reproduce published arrays at n=1..6   PASS")
    log("  base-case check              PASS")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Item 8 boundary-classification sweep")
    ap.add_argument("--csv", default=DEFAULT_CSV,
                    help=f"per-condition file (default {DEFAULT_CSV})")
    ap.add_argument("--epsilon", default=EPSILON)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"Not found: {args.csv}\nPass --csv with the path.")

    os.makedirs(args.outdir, exist_ok=True)
    lines: list[str] = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 78)
    log("Paper 9 — editorial item 8: zero-boundary classification")
    log(f"source: {args.csv}   epsilon: {args.epsilon}")
    log("=" * 78)
    log("")

    per = load(args.csv, args.epsilon)
    verify(per, log)

    # ── per-configuration table, ordered by mean ─────────────────────────────
    log("")
    log("PER-CONFIGURATION MEANS AND DISPERSION")
    log(f"  {'configuration':<26} {'mean dMRR':>11} {'SD':>9} "
        f"{'|mean|/SD':>10}  group at tau=0")
    for m in sorted(per, key=lambda k: per[k]["mean"]):
        d = per[m]
        ratio = abs(d["mean"]) / d["sd"] if d["sd"] else float("inf")
        grp = "harmed" if d["mean"] < 0 else "benefited"
        mark = "   <- boundary configuration" if m == BOUNDARY_CONFIG else ""
        log(f"  {m:<26} {d['mean']:>+11.4f} {d['sd']:>9.4f} "
            f"{ratio:>10.2f}  {grp}{mark}")
    log("")
    log("  The boundary configuration is distinguished by the ratio in column 4:")
    log("  its mean is small relative to its between-condition dispersion, so its")
    log("  group assignment is the least determined of the panel.")

    # ── sweep ────────────────────────────────────────────────────────────────
    rows = []
    tau = TAU_MIN
    while tau <= TAU_MAX + 1e-12:
        rows.append(classify(per, round(tau, 6)))
        tau += TAU_STEP

    log("")
    log("BOUNDARY SWEEP")
    log("  A configuration is assigned to the harmed group if its mean dMRR@10")
    log("  is below tau. tau = 0 is the rule used in the manuscript.")
    log("")
    log(f"  {'tau':>8}  {'n_ben':>5} {'n_harm':>6}  {'d_ben':>9} {'d_harm':>9}"
        f"  {'sp(n=6)':>8}  {'req(n=6)':>8}  {'p* (2dp)':>8}  {'p* exact':>9}"
        f"  boundary config")
    for r in rows:
        mark = "   <- submitted rule" if abs(r["tau"]) < 1e-12 else ""
        ps = r["p_star"]
        ps2 = f"{ps:.2f}" if ps == ps else "n/a"
        pse = f"{ps:.6f}" if ps == ps else "n/a"
        log(f"  {r['tau']:>+8.4f}  {r['n_benefited']:>5} {r['n_harmed']:>6}"
            f"  {r['d_ben']:>+9.4f} {r['d_harm']:>+9.4f}"
            f"  {r['sp_n6']:>8.4f}  {r['screen_requirement_n6']:>8.4f}"
            f"  {ps2:>8}  {pse:>9}"
            f"  {r['boundary_config_group']}{mark}")

    # ── the editor's pair, reproduced ────────────────────────────────────────
    log("")
    log("EDITORIAL ITEM 8 — THE QUOTED PAIR, REPRODUCED")
    base = classify(per, 0.0)
    alt = classify(per, -0.02)      # ablation reassigned to the benefited group
    log(f"  {'classification':<34} {'sp(n=6)':>9} {'se(n=6)':>9}"
        f" {'requirement':>12} {'p*':>8}")
    for lab, r in (("ablation benefited", alt),
                   ("ablation harmed (submitted)", base)):
        log(f"  {lab:<34} {r['sp_n6']:>9.4f} {r['se_n6']:>9.4f}"
            f" {r['screen_requirement_n6']:>12.4f} {r['p_star']:>8.4f}")
    ratio = base["screen_requirement_n6"] / alt["screen_requirement_n6"]
    log(f"  requirement ratio: {ratio:.1f}x   "
        f"(the letter says 'roughly tenfold')")
    log("")
    log("  The requirement is very nearly proportional to (1 - sp), because")
    log("  (1 - sp)|d_harm| is small beside se*d_ben. The move from 0.996 to")
    log("  0.936 is a ~17-fold change in the false-positive rate, so a ~11-fold")
    log("  change in the requirement is proportionality, not instability.")
    log("")
    log(f"  {'sp':>7}  {'1 - sp':>8}  {'requirement':>12}")
    for spv in (0.999, 0.996, 0.99, 0.98, 0.96, 0.936, 0.90):
        log(f"  {spv:>7.3f}  {1-spv:>8.3f}  "
            f"{screen_requirement(0.843, spv, D_BEN_EXPECTED, D_HARM_EXPECTED):>12.4f}")

    # ── flip points ──────────────────────────────────────────────────────────
    log("")
    log("FLIP POINTS — the tau at which each configuration changes group")
    log(f"  {'configuration':<26} {'flips at tau':>13}  "
        f"{'distance from 0':>16}")
    flips = []
    for m in sorted(per, key=lambda k: abs(per[k]["mean"])):
        mu = per[m]["mean"]
        mark = "   <- nearest the boundary" if m == BOUNDARY_CONFIG else ""
        log(f"  {m:<26} {mu:>+13.4f}  {abs(mu):>16.4f}{mark}")
        flips.append({"configuration": m, "flips_at_tau": mu,
                      "distance_from_zero": abs(mu),
                      "sd": per[m]["sd"],
                      "abs_mean_over_sd": (abs(mu) / per[m]["sd"]
                                           if per[m]["sd"] else None)})

    nearest = min(per, key=lambda k: abs(per[k]["mean"]))
    second = sorted(per, key=lambda k: abs(per[k]["mean"]))[1]
    log("")
    log(f"  Nearest the boundary: {nearest} at "
        f"{per[nearest]['mean']:+.4f}")
    log(f"  Next nearest:         {second} at {per[second]['mean']:+.4f}")
    log(f"  The classification is stable for any tau in "
        f"({per[nearest]['mean']:+.4f}, {per[second]['mean']:+.4f}) "
        f"other than at the flip itself.")

    # ── what the sweep shows about p* ────────────────────────────────────────
    valid = [r["p_star"] for r in rows if r["p_star"] == r["p_star"]]
    lo, hi = min(valid), max(valid)
    rounded = sorted({round(p, 2) for p in valid})
    log("")
    log("=" * 78)
    log("WHAT MOVES")
    log("=" * 78)
    log(f"  p* across the whole sweep:   {lo:.4f} to {hi:.4f}")
    log(f"  at two decimals:             "
        f"{', '.join(f'{v:.2f}' for v in rounded)}")
    log(f"  regularisation-sweep range:  0.32 to 0.70 (for comparison)")
    log("")
    if len(rounded) > 1:
        log("  The threshold DOES move at manuscript precision across this")
        log("  sweep. That is the honest statement of item 8: the boundary rule")
        log("  is a modelling choice with consequences, and the sweep shows its")
        log("  size rather than leaving it at a single before-and-after pair.")
    else:
        log("  The threshold does not move at manuscript precision across this")
        log("  sweep.")

    # ── outputs ──────────────────────────────────────────────────────────────
    def write(name, data):
        if not data:
            return
        p = os.path.join(args.outdir, name)
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
        return p

    write("boundary_item08_sweep.csv", rows)
    write("boundary_item08_flips.csv", flips)
    with open(os.path.join(args.outdir, "boundary_item08_summary.txt"),
              "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    log("")
    log("=" * 78)
    log("FILES WRITTEN")
    for n in ("boundary_item08_sweep.csv", "boundary_item08_flips.csv",
              "boundary_item08_summary.txt"):
        log(f"  {os.path.join(args.outdir, n)}")
    log("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
