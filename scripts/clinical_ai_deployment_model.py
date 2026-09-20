#!/usr/bin/env python3
"""
clinical_ai_deployment_model.py
===============================

Canonical decision model for "Deployment-Specific Benefit and Harm in Clinical
Artificial Intelligence" (JMIR AI ms#109863, major revision).

This script computes every load-bearing number in the manuscript and verifies
each against the value printed in the paper. It replaces clinical_rag_he_model
v1-v6, which grew around an earlier analysis and retained machinery the revised
paper no longer uses.

DELIBERATELY ABSENT, and why
----------------------------
* No prevalence prior. Earlier versions carried a Beta prior derived from a
  k = 2 of 55 mechanistic count. The revised paper states that the publication
  frame does not identify a distribution for the response-defined prevalence p,
  that the no-assumption bound is 0 to 1, and that the 1/55 count is descriptive
  and is not entered into the model. Any quantity requiring a distribution for p
  is therefore not computed here.
* No population strategy probabilities, no EVPI, no EVPPI. Each requires a
  distribution for p.
* No "minimum viable" or "hard minimum" condition count. The revised paper
  reports 1 to 6 conditions as a prior-dependent design trade-off: one condition
  suffices where the site prior exceeds 0.2101, six where it exceeds 0.0712.
* No zero-cost screening formula. The screening threshold separates the fixed
  screening cost S from the conditional deployment cost T, because T is incurred
  at a rate that itself depends on p (Methods).

WHAT IT COMPUTES
----------------
1. Response-defined groups at each regularisation setting, and the resulting
   universal-deployment threshold at each.
2. The base-case threshold, exactly and in the zero-cost limit.
3. Screen sensitivity and specificity by number of scored conditions.
4. The exact screening requirement and specificity threshold, with S and T
   carried separately.
5. The monetary scale: value per event, value exposed, strategy costs, and the
   cost ratios shown to be negligible.

The economic sensitivity analyses are in paper9_sensitivity.py and the
classification-boundary sweep is in paper9_boundary_sweep.py; both write the
tables reported in Multimedia Appendix 9.

Standard library only.

    python3 clinical_ai_deployment_model.py
    python3 clinical_ai_deployment_model.py --percondition path/to/file.csv
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
# PARAMETERS, with provenance. All monetary quantities are in 2024 euros.
# ─────────────────────────────────────────────────────────────────────────────

# Exchange rate: Norges Bank annual average for 2024, the price year of the
# SAMDATA bed-day estimate (Methods; editorial comment 3).
NOK_PER_EUR = 11.6276

# Willingness to pay: Meld. St. 34 (2015-2016), NOK 275,000 per quality-adjusted
# life year at the base severity class, rising to NOK 825,000 at the highest.
# Applied at nominal policy value without indexation, which is Norwegian
# practice; the Directorate for Medical Products reviewed the thresholds in 2026
# and recommended no interim adjustment.
WTP_NOK = 275_000.0
WTP_NOK_SEVERITY = 825_000.0
WTP_EUR = WTP_NOK / NOK_PER_EUR

# Adverse-event treatment cost: SAMDATA 2024 somatic bed-day NOK 26,153 times
# 5.11 excess bed-days per event (Hoogervorst-Schilp 2015). A treatment cost,
# not an indemnity payment.
BEDDAY_NOK = 26_153.0
EXCESS_DAYS = 5.11
COST_EVENT_EUR = (BEDDAY_NOK * EXCESS_DAYS) / NOK_PER_EUR

# Utility loss per event, and the Kirwan 2023 alternative.
QALY_EVENT = 0.0015
QALY_EVENT_ALT = 0.028

# Deployment scale: Statistics Norway recorded 2,656,857 patients in somatic
# specialist health care in 2024; 133,000 annual retrievals is approximately
# one twentieth of that, the scale of a single helseforetak.
PATIENTS_2024 = 2_656_857
N_RETRIEVALS = 133_000.0

# Illustrative parameters. These have no published source and set the absolute
# monetary scale only; they reach the threshold solely through the cost ratio.
ALPHA = 0.70          # technical metric to decision accuracy
ADOPTION = 0.60       # retrieval output adopted into a decision
P_EVENT = 0.12        # adverse event given an adopted erroneous retrieval

# Horizon and discounting.
HORIZON_YEARS = 5
DISCOUNT_RATE = 0.04
D = sum(1.0 / (1.0 + DISCOUNT_RATE) ** t for t in range(1, HORIZON_YEARS + 1))

# Strategy costs. Scenario inputs expressed in 2024 euros, not empirically
# estimated. T is the deployment cost, incurred only where the correction is
# applied; S is the fixed screening cost, incurred whichever way the screen
# reads. They are carried separately because T is incurred at a rate that
# depends on p.
K_IMPL = 800.0
K_ANNUAL = 150.0
T_DEPLOY = K_IMPL + K_ANNUAL * D
S_PER_CONDITION = 400.0

BASE_EPSILON = "1e-05"
N_COND_MAX = 6

# ─────────────────────────────────────────────────────────────────────────────
# PER-CONDITION EFFECTS
#
# Mean and standard deviation of dMRR@10 across the six scored conditions
# (three benchmark corpora x two query formats) for each of the 13
# configurations, at the base regularisation setting. Full precision, so the
# derived quantities reproduce the manuscript without rounding drift.
#
# Read from appendix5_epsilon_percondition.csv when present, which also supplies
# the other regularisation settings; the table below is the base case and lets
# the script run standalone.
# ─────────────────────────────────────────────────────────────────────────────

BASE_EFFECTS = {
    "BERT-base-uncased":      (+0.036854, 0.124457),
    "BGE-base":               (-0.073103, 0.066369),
    "BioBERT":                (+0.063154, 0.158623),
    "BioLORD-2023":           (-0.078546, 0.072132),
    "BioMistral-7B":          (+0.058944, 0.159129),
    "ClinicalBERT":           (+0.052957, 0.139808),
    "E5-Mistral-7B":          (+0.203428, 0.146357),
    "E5-Mistral-7B-ablation": (-0.015669, 0.199636),
    "GTE-base":               (-0.068725, 0.061315),
    "MedCPT":                 (-0.134591, 0.100692),
    "Nomic-embed-text":       (-0.084626, 0.078146),
    "Nomic-embed-text-nopfx": (-0.080890, 0.082039),
    "Phi-3-mini":             (+0.039134, 0.106232),
}

DEFAULT_PERCONDITION = "appendix5_epsilon_percondition.csv"

# Values as printed in the manuscript, checked on every run.
EXPECTED = {
    "d_ben": 0.075745, "d_harm": -0.076593,
    "n_ben": 6, "n_harm": 7,
    "p_star_zero": 0.502782, "p_star_exact": 0.502810,
    "cost_event": 11_494, "value_event": 11_529,
    "exposed": 344.0e6, "t_deploy": 1_468, "m_over_t": 234_000,
    "se": [0.687, 0.742, 0.777, 0.804, 0.825, 0.843],
    "sp": [0.819, 0.885, 0.912, 0.925, 0.932, 0.936],
    "requirement": [0.2101, 0.1359, 0.1025, 0.0858, 0.0767, 0.0712],
    "epsilon_range": (0.3163, 0.6993),
}

_ND = NormalDist()


# ─────────────────────────────────────────────────────────────────────────────
# EFFECTS AND GROUPING
# ─────────────────────────────────────────────────────────────────────────────

def load_effects(path: str | None = None) -> dict[str, dict[str, tuple]]:
    """Per-configuration mean and SD of dMRR@10, keyed by regularisation setting.

    Falls back to the embedded base-case table when the file is absent, in which
    case only the base setting is available.
    """
    path = path or DEFAULT_PERCONDITION
    if not os.path.exists(path):
        print(f"  note: {path} not found; using the embedded base-case table "
              f"({BASE_EPSILON} only)")
        return {BASE_EPSILON: dict(BASE_EFFECTS)}

    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[str, dict[str, tuple]] = {}
    for eps in sorted({r["epsilon"] for r in rows}, key=float):
        sub = [r for r in rows if r["epsilon"] == eps]
        per = {}
        for m in sorted({r["model"] for r in sub}):
            v = [float(r["dMRR@10"]) for r in sub if r["model"] == m]
            per[m] = (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)
        out[eps] = per
    return out


def response_groups(per: dict[str, tuple]) -> tuple[list, list, float, float]:
    """Split configurations by the SIGN of their measured mean effect.

    Grouping is by measured response, not by nominal training objective. One
    configuration (E5-Mistral-7B-ablation) is assigned differently under the two
    rules; the consequence is quantified in paper9_boundary_sweep.py.
    """
    benefited = [m for m in per if per[m][0] > 0]
    harmed = [m for m in per if per[m][0] < 0]
    d_ben = st.mean([per[m][0] for m in benefited]) if benefited else float("nan")
    d_harm = st.mean([per[m][0] for m in harmed]) if harmed else float("nan")
    return benefited, harmed, d_ben, d_harm


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

def value_per_event(cost=COST_EVENT_EUR, qaly=QALY_EVENT, wtp=WTP_EUR) -> float:
    """V, the monetary value of one averted adverse event."""
    return cost + wtp * qaly


def value_exposed(alpha=ALPHA, adoption=ADOPTION, p_event=P_EVENT,
                  n=N_RETRIEVALS, d=D, v=None) -> float:
    """M, the monetary value exposed to the intervention over the horizon."""
    return alpha * adoption * p_event * n * d * (value_per_event() if v is None else v)


def p_star(d_ben: float, d_harm: float, m: float | None = None,
           t: float = T_DEPLOY) -> float:
    """Universal-deployment threshold, exact form including the cost ratio.

        p* = (|d_harm| + T/M) / (d_ben + |d_harm|)

    Pass m=None for the zero-cost limit.
    """
    h = abs(d_harm)
    cost_term = 0.0 if m is None else t / m
    return (h + cost_term) / (d_ben + h)


def derive_screen(n_cond: int, per: dict[str, tuple]) -> tuple[float, float]:
    """Screen sensitivity and specificity at n_cond scored conditions.

    A screen scores n conditions and takes the sign of the mean. The probability
    of a sign error for configuration m is Phi(-|d_m| / (s_m / sqrt(n))), where
    s_m is its between-condition standard deviation. Sensitivity is one minus
    the mean error probability over the benefited group, specificity the same
    over the harmed group.
    """
    benefited, harmed, _, _ = response_groups(per)

    def err(group):
        return [_ND.cdf(-abs(per[m][0]) / (per[m][1] / math.sqrt(n_cond)))
                for m in group if per[m][1] > 0]

    e_ben, e_harm = err(benefited), err(harmed)
    se = 1 - sum(e_ben) / len(e_ben) if e_ben else float("nan")
    sp = 1 - sum(e_harm) / len(e_harm) if e_harm else float("nan")
    return se, sp


def screen_requirement(se: float, sp: float, d_ben: float, d_harm: float,
                       n_cond: int, m: float, t: float = T_DEPLOY,
                       s_per_cond: float = S_PER_CONDITION) -> float:
    """Minimum site prior probability at which screen-and-treat beats no action.

        p* = [(1-sp)(h + t) + s] / [se(d_ben - t) + (1-sp)(h + t)]

    with s = S/M and t = T/M, h = |d_harm|. S is the fixed screening cost and T
    the deployment cost, incurred only on the systems the screen selects.
    Combining them would require assuming a value for p.
    """
    h = abs(d_harm)
    tt = t / m
    ss = (s_per_cond * n_cond) / m
    b = (1 - sp) * (h + tt)
    den = se * (d_ben - tt) + b
    return (b + ss) / den if den > 0 else float("nan")


def sp_star(p: float, se: float, d_ben: float, d_harm: float, n_cond: int,
            m: float, t: float = T_DEPLOY,
            s_per_cond: float = S_PER_CONDITION) -> float:
    """Specificity at which screen-and-treat beats no action at site prior p.

        sp* = 1 - [p*se(d_ben - t) - s] / [(1-p)(h + t)]
    """
    h = abs(d_harm)
    tt = t / m
    ss = (s_per_cond * n_cond) / m
    den = (1 - p) * (h + tt)
    return 1 - (p * se * (d_ben - tt) - ss) / den if den > 0 else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--percondition", default=None,
                    help=f"per-condition effects (default {DEFAULT_PERCONDITION})")
    args = ap.parse_args()

    failures: list[str] = []

    def check(name, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL':<5}{name:<44}{detail}")
        if not ok:
            failures.append(name)

    print("=" * 78)
    print("Canonical decision model - JMIR AI ms#109863")
    print("=" * 78)

    effects = load_effects(args.percondition)
    per = effects[BASE_EPSILON]
    benefited, harmed, d_ben, d_harm = response_groups(per)
    h = abs(d_harm)

    v = value_per_event()
    m = value_exposed(v=v)
    p0 = p_star(d_ben, d_harm, None)
    p1 = p_star(d_ben, d_harm, m)

    print("\nBASE CASE, response-defined grouping at epsilon = " + BASE_EPSILON)
    print(f"  benefited                   n={len(benefited)}  mean {d_ben:+.6f}")
    print(f"  harmed                      n={len(harmed)}  mean {d_harm:+.6f}")
    print(f"  p* (zero-cost limit)        {p0:.6f}")
    print(f"  p* (exact, with T/M)        {p1:.6f}")

    print("\nMONETARY SCALE, 2024 euros")
    print(f"  exchange rate               {NOK_PER_EUR} NOK/EUR "
          f"(Norges Bank annual average 2024)")
    print(f"  adverse-event cost          EUR {COST_EVENT_EUR:,.2f}"
          f"   (NOK {BEDDAY_NOK * EXCESS_DAYS:,.2f})")
    print(f"  willingness to pay          EUR {WTP_EUR:,.1f}/QALY"
          f"   (NOK {WTP_NOK:,.0f})")
    print(f"  value per event V           EUR {v:,.2f}")
    print(f"  value exposed M             EUR {m:,.0f}")
    print(f"  deployment cost T           EUR {T_DEPLOY:,.2f}   (M/T = {m / T_DEPLOY:,.0f})")
    print(f"  T/M                         {T_DEPLOY / m:.3e}")
    print(f"  discount factor D           {D:.4f}  ({HORIZON_YEARS} yr at {DISCOUNT_RATE:.0%})")

    print("\nSCREENING BY EVIDENCE BREADTH")
    print(f"  {'conditions':>10} {'se':>8} {'sp':>8} {'S (EUR)':>9} "
          f"{'requirement':>12} {'vs p*':>8}")
    req = []
    for n in range(1, N_COND_MAX + 1):
        se, sp = derive_screen(n, per)
        r = screen_requirement(se, sp, d_ben, d_harm, n, m)
        req.append(r)
        print(f"  {n:>10} {se:>8.4f} {sp:>8.4f} {S_PER_CONDITION * n:>9,.0f} "
              f"{r:>12.4f} {p0 / r:>7.1f}x")

    print("\nREGULARISATION SENSITIVITY, groups recomputed at each setting")
    print(f"  {'epsilon':>10} {'n_ben':>6} {'n_harm':>7} {'d_ben':>10} "
          f"{'d_harm':>10} {'p*':>8}")
    eps_p = []
    for eps in sorted(effects, key=float):
        b_, hm_, db_, dh_ = response_groups(effects[eps])
        pp = p_star(db_, dh_, None)
        eps_p.append(pp)
        mark = "   <- base" if eps == BASE_EPSILON else ""
        print(f"  {eps:>10} {len(b_):>6} {len(hm_):>7} {db_:>+10.6f} "
              f"{dh_:>+10.6f} {pp:>8.4f}{mark}")
    if len(eps_p) > 1:
        print(f"  range across settings: {min(eps_p):.4f} to {max(eps_p):.4f}")

    print("\nVERIFICATION AGAINST THE MANUSCRIPT")
    e = EXPECTED
    check("group sizes", len(benefited) == e["n_ben"] and len(harmed) == e["n_harm"],
          f"{len(benefited)}/{len(harmed)}")
    check("d_ben", abs(d_ben - e["d_ben"]) < 5e-7, f"{d_ben:+.6f}")
    check("d_harm", abs(d_harm - e["d_harm"]) < 5e-7, f"{d_harm:+.6f}")
    check("p* zero-cost", abs(p0 - e["p_star_zero"]) < 5e-7, f"{p0:.6f}")
    check("p* exact", abs(p1 - e["p_star_exact"]) < 5e-7, f"{p1:.6f}")
    check("adverse-event cost", round(COST_EVENT_EUR) == e["cost_event"],
          f"{COST_EVENT_EUR:,.2f}")
    check("value per event", round(v) == e["value_event"], f"{v:,.2f}")
    check("value exposed", abs(m - e["exposed"]) / e["exposed"] < 1e-3,
          f"{m:,.0f}")
    check("deployment cost", round(T_DEPLOY) == e["t_deploy"], f"{T_DEPLOY:,.2f}")
    check("M/T", abs(m / T_DEPLOY - e["m_over_t"]) < 1_000, f"{m / T_DEPLOY:,.0f}")
    se_ok = sp_ok = True
    for n in range(1, N_COND_MAX + 1):
        se, sp = derive_screen(n, per)
        se_ok &= abs(se - e["se"][n - 1]) < 5e-4
        sp_ok &= abs(sp - e["sp"][n - 1]) < 5e-4
    check("screen sensitivity, n=1..6", se_ok)
    check("screen specificity, n=1..6", sp_ok)
    check("screening requirement, n=1..6",
          all(abs(a - b) < 5e-5 for a, b in zip(req, e["requirement"])),
          " ".join(f"{r:.4f}" for r in req))
    if len(eps_p) > 1:
        lo, hi = e["epsilon_range"]
        check("regularisation range",
              abs(min(eps_p) - lo) < 5e-4 and abs(max(eps_p) - hi) < 5e-4,
              f"{min(eps_p):.4f} to {max(eps_p):.4f}")

    print("\n" + "=" * 78)
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED: {', '.join(failures)}")
        print("The model and the manuscript disagree. Reconcile before release.")
    else:
        print("ALL CHECKS PASSED - the model reproduces the manuscript.")
    print("=" * 78)
    print("\nEconomic sensitivity analyses: paper9_sensitivity.py")
    print("Classification-boundary sweep:  paper9_boundary_sweep.py")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
