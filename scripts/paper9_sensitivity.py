#!/usr/bin/env python3
"""
paper9_sensitivity.py
=====================

Sensitivity analyses for JMIR AI ms#109863 (Paper 9), major revision,
editorial comments 3, 7, 19 and 21.

Base case is taken from Appendix 5 (response-defined grouping, v1.1.0) and is
verified on startup: the script aborts if it does not reproduce the submitted
threshold.

Model
-----
    S    = d_ben + |d_harm|
    M    = alpha * adoption * p_event * N * D * V      (monetary value exposed)
    V    = C_event + (lambda_NOK / FX) * Q_event       (value per adverse event)
    K    = K_impl + K_annual * D                       (strategy cost)
    p*   = ( |d_harm| + K/M ) / S

Every illustrative parameter (alpha, adoption, p_event) and every monetary
parameter (C_event, Q_event, lambda, N, D) enters ONLY through M, and therefore
reaches p* only through the term K/M. That term is 4.3e-06 at base case, so the
analyses below quantify how far p* can be moved at all — which is the substantive
answer to comments 7, 19 and 21.

Outputs (written next to this script unless --outdir is given)
-------------------------------------------------------------
    sens_item03_exchange_rate.csv
    sens_item07_oneway.csv
    sens_item07_psa_summary.csv
    sens_item19_horizon.csv
    sens_item21_cost_event.csv
    sens_summary.txt

Usage
-----
    python3 paper9_sensitivity.py
    python3 paper9_sensitivity.py --draws 100000 --seed 7 --outdir ./sens

Requires numpy and scipy only.
"""

from __future__ import annotations

import argparse
import os
import sys

import importlib.util

import numpy as np
from scipy import stats

# Single source of truth. Every effect size, cost, rate and threshold function
# comes from the canonical model; nothing is re-implemented here.
_SPEC = importlib.util.spec_from_file_location(
    "model", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "clinical_ai_deployment_model.py"))
MODEL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(MODEL)

# ─────────────────────────────────────────────────────────────────────────────
# BASE CASE — Appendix 5, response-defined grouping (v1.1.0)
# ─────────────────────────────────────────────────────────────────────────────

_EFFECTS = MODEL.load_effects()
_BEN, _HARM, D_BEN, D_HARM = MODEL.response_groups(_EFFECTS[MODEL.BASE_EPSILON])

ALPHA = 0.70              # ILLUSTRATIVE: technical metric -> decision accuracy
ADOPTION = 0.60           # ILLUSTRATIVE: retrieval output adopted into decision
P_EVENT = 0.12            # ILLUSTRATIVE: adverse event | adopted erroneous retrieval

C_EVENT_SUBMITTED = 11_422.0   # EUR, as submitted (FX 11.70, undated)
Q_EVENT = 0.0015          # QALY, utility loss per adverse event
Q_EVENT_ALT = 0.028       # QALY, Kirwan 2023 alternative

LAMBDA_NOK = MODEL.WTP_NOK    # NOK/QALY, Meld. St. 34 (2015-2016)
LAMBDA_NOK_SEVERITY = 825_000.0   # NOK/QALY, severity-weighted upper tier

N_RETRIEVALS = MODEL.N_RETRIEVALS  # annual retrievals
HORIZON = 5               # years
DISCOUNT = 0.04           # annual discount rate

K_IMPL = MODEL.K_IMPL            # EUR, implementation
K_ANNUAL = MODEL.K_ANNUAL          # EUR, annual maintenance

BEDDAY_NOK = 26_153.0     # SAMDATA 2024
EXCESS_DAYS = 5.11        # Hoogervorst-Schilp 2015

# ─────────────────────────────────────────────────────────────────────────────
# ITEM 3 — exchange rate, stated and dated.
#
# The submitted manuscript used an implicit rate of 11.70 NOK/EUR with no date
# or source. That rate is not a market rate for any period near submission
# (EUR/NOK traded at roughly 10.85-10.98 in August 2026).
#
# The adverse-event cost is built from a SAMDATA 2024 bed-day rate, so its
# price year is 2024 and it converts at the 2024 annual average, consistent
# with CHEERS 2022 reporting of currency and conversion.
#
#   Norges Bank annual average 2024:  11.6276 NOK/EUR
#   (arithmetic mean of monthly averages; Norges Bank daily middle rates)
#
# For reference, the Norges Bank annual average for 2025 is 11.7188.
# ─────────────────────────────────────────────────────────────────────────────

FX_NOK_PER_EUR = MODEL.NOK_PER_EUR
FX_DATE = "Norges Bank annual average for 2024 (price year of the unit cost)"
FX_SUBMITTED = 11.70      # undated rate used in the submitted manuscript

# Derived, not asserted: the cost follows from the unit cost and the rate.
C_EVENT = MODEL.COST_EVENT_EUR                          # EUR 11,493.5

# ─────────────────────────────────────────────────────────────────────────────
# ITEM 21 — lower European unit-cost estimates.
#
# Appendix 5 records that the Norwegian unit cost EXCEEDS the estimates of
# Durand 2024 and Laroche 2025 but is reported without reconciliation to them.
# Both published sources give RANGES, not point estimates, so each is run at
# its low, central and high value.
#
#   Durand 2024   Health Econ Rev 14:11, doi 10.1186/s13561-024-00481-y
#                 Systematic review, 20 studies. Costs per hospitalisation
#                 approximately EUR 6,000-10,000 from hospital, health-insurance
#                 or health-system perspectives.
#
#   Laroche 2025  Br J Clin Pharmacol 91(2):439-450, doi 10.1111/bcp.16266
#                 IATROSTAT-ECO, 196 patients, 38 French public hospitals,
#                 French public health insurance perspective, direct medical
#                 costs over 3 months from the first day of the ADR admission.
#                 Table 3, total cost per patient with ADR-HA:
#                   2018 tariffs  mean EUR 5,208 +/- 3,719, range 514-23,355
#                   2023 tariffs  mean EUR 5,974 +/- 4,232, range 618-27,380
#                 The 2023-tariff figures are the closer vintage to SAMDATA
#                 2024 and are treated as the primary comparator.
#                 VERIFIED against the primary article.
# ─────────────────────────────────────────────────────────────────────────────

VERIFY_LAROCHE = False    # confirmed against Br J Clin Pharmacol 2025;91:439-450

COST_SCENARIOS = [
    # (label, low, central, high)
    ("Norwegian (SAMDATA 2024, NB 2024 rate)", None, C_EVENT, None),
    ("Durand 2024 (review, per hospitalisation)", 6_000.0, 8_000.0, 10_000.0),
    ("Laroche 2025 (IATROSTAT-ECO, 2023 tariffs)", 618.0, 5_974.0, 27_380.0),
    ("Laroche 2025 (IATROSTAT-ECO, 2018 tariffs)", 514.0, 5_208.0, 23_355.0),
]

# Horizons examined for comment 19 (AI model obsolescence).
HORIZONS = (1, 2, 3, 5, 10)
DISCOUNT_RATES = (0.00, 0.03, 0.04)

TOL = 5e-5                # tolerance on the base-case reproduction check


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

def discount_factor(rate: float = DISCOUNT, years: int = HORIZON) -> float:
    """Sum of annual discount factors, end-of-year convention."""
    return float(sum(1.0 / (1.0 + rate) ** t for t in range(1, years + 1)))


def value_per_event(c_event: float = C_EVENT,
                    q_event: float = Q_EVENT,
                    lambda_nok: float = LAMBDA_NOK,
                    fx: float = FX_NOK_PER_EUR) -> float:
    """V = treatment cost + monetised utility loss, in EUR."""
    return c_event + (lambda_nok / fx) * q_event


def exposed_value(alpha=ALPHA, adoption=ADOPTION, p_event=P_EVENT,
                  n=N_RETRIEVALS, d=None, v=None) -> float:
    """M = alpha * adoption * p_event * N * D * V."""
    d = discount_factor() if d is None else d
    v = value_per_event() if v is None else v
    return alpha * adoption * p_event * n * d * v


def strategy_cost(k_impl=K_IMPL, k_annual=K_ANNUAL, d=None) -> float:
    """K = implementation + discounted annual maintenance."""
    d = discount_factor() if d is None else d
    return k_impl + k_annual * d


def p_star(m: float, k: float,
           d_ben: float = D_BEN, d_harm: float = D_HARM) -> float:
    """Universal-deployment threshold."""
    s = d_ben + abs(d_harm)
    return (abs(d_harm) + k / m) / s


def p_star_zero_cost(d_ben: float = D_BEN, d_harm: float = D_HARM) -> float:
    return abs(d_harm) / (d_ben + abs(d_harm))


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP CHECK
# ─────────────────────────────────────────────────────────────────────────────

def verify_base_case(log) -> dict:
    """Reproduce the submitted base case, or abort."""
    d = discount_factor()
    v = value_per_event()
    m = exposed_value(d=d, v=v)
    k = strategy_cost(d=d)
    p0 = p_star_zero_cost()
    p1 = p_star(m, k)

    implied_fx = (BEDDAY_NOK * EXCESS_DAYS) / C_EVENT
    c_submitted_check = (BEDDAY_NOK * EXCESS_DAYS) / FX_SUBMITTED

    log("BASE CASE (Appendix 5, response-defined grouping)")
    log(f"  d_ben                       {D_BEN:+.6f}")
    log(f"  d_harm                      {D_HARM:+.6f}")
    log(f"  S = d_ben + |d_harm|        {D_BEN + abs(D_HARM):.6f}")
    log(f"  D ({HORIZON} yr at {DISCOUNT:.0%})            {d:.4f}")
    log(f"  lambda                      NOK {LAMBDA_NOK:,.0f}/QALY "
        f"= EUR {LAMBDA_NOK / FX_NOK_PER_EUR:,.1f}/QALY at {FX_NOK_PER_EUR}")
    log(f"  V per event                 EUR {v:,.2f}")
    log(f"  M                           EUR {m:,.0f}")
    log(f"  K                           EUR {k:,.1f}")
    log(f"  K/M                         {k / m:.4e}")
    log(f"  p* (zero cost)              {p0:.6f}")
    log(f"  p* (with cost)              {p1:.6f}")
    log(f"  FX rate                     {FX_NOK_PER_EUR:.4f} NOK/EUR")
    log(f"  FX basis                    {FX_DATE}")
    log(f"  C_event (this rate)         EUR {C_EVENT:,.1f}")
    log(f"  C_event (as submitted)      EUR {c_submitted_check:,.1f} "
        f"at the undated {FX_SUBMITTED} rate")
    log(f"  implied FX check            {implied_fx:.4f} NOK/EUR")

    errors = []
    if abs(p0 - MODEL.p_star(D_BEN, D_HARM, None)) > TOL:
        errors.append(f"zero-cost p* = {p0:.6f}, expected "
                      f"{MODEL.p_star(D_BEN, D_HARM, None):.6f}")
    if abs(p1 - MODEL.p_star(D_BEN, D_HARM, m)) > TOL:
        errors.append(f"p* with cost = {p1:.6f}, expected "
                      f"{MODEL.p_star(D_BEN, D_HARM, m):.6f}")
    if abs(k - 1468.0) > 1.0:
        errors.append(f"K = {k:.1f}, expected ~1468")
    if abs(implied_fx - FX_NOK_PER_EUR) > 1e-6:
        errors.append(f"implied FX {implied_fx:.6f} != FX_NOK_PER_EUR "
                      f"{FX_NOK_PER_EUR} — C_event is no longer derived")
    if abs(c_submitted_check - C_EVENT_SUBMITTED) > 1.0:
        errors.append(f"submitted C_event reconstructs to "
                      f"{c_submitted_check:,.1f}, expected "
                      f"{C_EVENT_SUBMITTED:,.0f}")

    if errors:
        log("")
        log("BASE CASE CHECK FAILED:")
        for e in errors:
            log(f"  - {e}")
        log("Parameters do not reproduce the submitted manuscript. Aborting.")
        sys.exit(1)

    log("  base-case check              PASS")
    return {"D": d, "V": v, "M": m, "K": k, "p0": p0, "p1": p1}


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 3 — exchange rate
# ─────────────────────────────────────────────────────────────────────────────

def item03_exchange_rate(log, rows):
    log("")
    log("ITEM 3 — EXCHANGE RATE USED TO RECONCILE NOK AND EUR")
    log(f"  NOK {BEDDAY_NOK:,.0f} bed-day x {EXCESS_DAYS} excess days "
        f"= NOK {BEDDAY_NOK * EXCESS_DAYS:,.2f} per event")
    log("")
    log(f"  {'FX (NOK/EUR)':>14}  {'C_event (EUR)':>15}  {'lambda (EUR/QALY)':>18}"
        f"  {'V (EUR)':>12}  {'p*':>9}")
    # Completed Norges Bank annual averages only. An incomplete year has no
    # annual average, and a year-to-date figure must be computed from the
    # daily series and archived before it can be reported as one.
    for fx, tag in ((FX_NOK_PER_EUR, "NB annual average 2024"),
                    (11.7188, "NB annual average 2025")):
        c = (BEDDAY_NOK * EXCESS_DAYS) / fx
        lam = LAMBDA_NOK / fx
        v = c + lam * Q_EVENT
        d = discount_factor()
        m = exposed_value(d=d, v=v)
        p = p_star(m, strategy_cost(d=d))
        mark = "  <- used" if abs(fx - FX_NOK_PER_EUR) < 1e-9 else ""
        log(f"  {fx:>14.4f}  {c:>15,.1f}  {lam:>18,.1f}  {v:>12,.1f}  "
            f"{p:>9.6f}   {tag}{mark}")
        rows.append({
            "fx_nok_per_eur": fx, "c_event_eur": c, "lambda_eur_qaly": lam,
            "v_eur": v, "p_star": p,
            "basis": tag,
            "used": abs(fx - FX_NOK_PER_EUR) < 1e-9,
        })
    log("")
    log(f"  Methods statement: {FX_NOK_PER_EUR} NOK/EUR, {FX_DATE}.")
    log(f"  This revises C_event from EUR {C_EVENT_SUBMITTED:,.0f} (submitted) "
        f"to EUR {C_EVENT:,.1f}.")


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 7 — illustrative parameters
# ─────────────────────────────────────────────────────────────────────────────

def _alpha_dist():
    """TruncN(0.70, 0.15) on (0, inf)."""
    lo = (0.0 - ALPHA) / 0.15
    return stats.truncnorm(a=lo, b=np.inf, loc=ALPHA, scale=0.15)


def item07_oneway(log, rows, base):
    log("")
    log("ITEM 7 — ONE-WAY SENSITIVITY ON THE ILLUSTRATIVE PARAMETERS")
    log("  Each varied across the 2.5th-97.5th percentile of its Appendix 5")
    log("  distribution, others held at base case.")
    log("")

    specs = [
        ("alpha", _alpha_dist(), ALPHA),
        ("adoption", stats.beta(9, 6), ADOPTION),
        ("p_event", stats.beta(6, 44), P_EVENT),
    ]

    d = discount_factor()
    v = value_per_event()
    k = strategy_cost(d=d)

    log(f"  {'parameter':>10}  {'low':>8}  {'base':>8}  {'high':>8}"
        f"  {'M low (EUR)':>16}  {'M high (EUR)':>16}"
        f"  {'p* low':>9}  {'p* high':>9}  {'p* range':>10}")

    for name, dist, base_val in specs:
        lo, hi = dist.ppf(0.025), dist.ppf(0.975)
        out = {}
        for tag, val in (("low", lo), ("high", hi)):
            kwargs = {"alpha": ALPHA, "adoption": ADOPTION,
                      "p_event": P_EVENT}
            kwargs[name] = float(val)
            m = exposed_value(d=d, v=v, **kwargs)
            out[tag] = (m, p_star(m, k))
        spread = abs(out["high"][1] - out["low"][1])
        log(f"  {name:>10}  {lo:>8.4f}  {base_val:>8.4f}  {hi:>8.4f}"
            f"  {out['low'][0]:>16,.0f}  {out['high'][0]:>16,.0f}"
            f"  {out['low'][1]:>9.6f}  {out['high'][1]:>9.6f}  {spread:>10.2e}")
        rows.append({
            "parameter": name, "low": lo, "base": base_val, "high": hi,
            "m_low_eur": out["low"][0], "m_high_eur": out["high"][0],
            "p_star_low": out["low"][1], "p_star_high": out["high"][1],
            "p_star_range": spread,
        })

    log("")
    log(f"  Base-case p* = {base['p1']:.6f}. The illustrative parameters set the")
    log("  absolute monetary scale M; they reach p* only through K/M.")


def item07_psa(log, rows, base, draws: int, seed: int):
    log("")
    log(f"ITEM 7 — PROBABILISTIC SENSITIVITY ANALYSIS ({draws:,} draws)")
    log("  alpha, adoption, p_event, C_event, Q_event, K_impl, K_annual drawn")
    log("  jointly from their Appendix 5 distributions.")

    rng = np.random.default_rng(seed)

    alpha = _alpha_dist().rvs(draws, random_state=rng)
    adoption = stats.beta(9, 6).rvs(draws, random_state=rng)
    p_event = stats.beta(6, 44).rvs(draws, random_state=rng)
    # Gamma(2.5, C_EVENT/2.5): shape fixed, scale DERIVED so the declared
    # mean and the sampled mean cannot drift apart (Appendix 5 records
    # Gamma(2.5, 4597); a stale literal of 4569 was corrected here).
    c_event = stats.gamma(2.5, scale=C_EVENT / 2.5).rvs(draws, random_state=rng)
    q_event = stats.gamma(1.5, scale=0.001).rvs(draws, random_state=rng)
    k_impl = stats.gamma(2, scale=400).rvs(draws, random_state=rng)
    k_annual = stats.gamma(1.5, scale=100).rvs(draws, random_state=rng)

    d = discount_factor()
    v = c_event + (LAMBDA_NOK / FX_NOK_PER_EUR) * q_event
    m = alpha * adoption * p_event * N_RETRIEVALS * d * v
    k = k_impl + k_annual * d
    s = D_BEN + abs(D_HARM)
    p = (abs(D_HARM) + k / m) / s

    def q(a, arr):
        return float(np.percentile(arr, a))

    for label, arr, fmt in (
        ("M (EUR)", m, ",.0f"),
        ("K (EUR)", k, ",.1f"),
        ("K/M", k / m, ".3e"),
        ("p*", p, ".6f"),
    ):
        log(f"  {label:>10}  mean {format(float(np.mean(arr)), fmt):>18}"
            f"   2.5% {format(q(2.5, arr), fmt):>18}"
            f"   97.5% {format(q(97.5, arr), fmt):>18}")
        rows.append({
            "quantity": label, "mean": float(np.mean(arr)),
            "p2_5": q(2.5, arr), "p50": q(50, arr), "p97_5": q(97.5, arr),
            "min": float(np.min(arr)), "max": float(np.max(arr)),
        })

    log("")
    log(f"  p* spans {q(97.5, p) - q(2.5, p):.2e} across the 95% PSA interval,")
    log("  against a structural regularisation-sweep range of 0.32-0.70.")


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 19 — time horizon
# ─────────────────────────────────────────────────────────────────────────────

def item19_horizon(log, rows):
    log("")
    log("ITEM 19 — TIME HORIZON (AI MODEL OBSOLESCENCE)")
    log("  D enters M and K together, so shortening the horizon moves both.")
    log("")
    log(f"  {'rate':>6}  {'years':>6}  {'D':>8}  {'M (EUR)':>18}"
        f"  {'K (EUR)':>10}  {'K/M':>12}  {'p*':>9}")

    v = value_per_event()
    for rate in DISCOUNT_RATES:
        for years in HORIZONS:
            d = discount_factor(rate, years)
            m = exposed_value(d=d, v=v)
            k = strategy_cost(d=d)
            p = p_star(m, k)
            base = (abs(rate - DISCOUNT) < 1e-12 and years == HORIZON)
            mark = "   <- base" if base else ""
            log(f"  {rate:>6.0%}  {years:>6d}  {d:>8.4f}  {m:>18,.0f}"
                f"  {k:>10,.1f}  {k / m:>12.3e}  {p:>9.6f}{mark}")
            rows.append({
                "discount_rate": rate, "horizon_years": years,
                "discount_factor": d, "m_eur": m, "k_eur": k,
                "k_over_m": k / m, "p_star": p, "base_case": base,
            })


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 21 — adverse-event unit cost
# ─────────────────────────────────────────────────────────────────────────────

def item21_cost_event(log, rows):
    log("")
    log("ITEM 21 — ADVERSE-EVENT TREATMENT COST")
    log("  The Norwegian unit cost exceeds published European estimates.")
    if VERIFY_LAROCHE:
        log("  NOTE: Laroche 2025 figures taken from a secondary summary and")
        log("  NOT yet verified against the primary article. Verify before use.")

    # Q_event and lambda variants recorded alongside, since they scale V too.
    variants = [
        ("base", Q_EVENT, LAMBDA_NOK),
        ("Q_event 0.028 (Kirwan 2023)", Q_EVENT_ALT, LAMBDA_NOK),
        ("lambda NOK 825,000 (severity)", Q_EVENT, LAMBDA_NOK_SEVERITY),
    ]

    d = discount_factor()
    k = strategy_cost(d=d)

    log("")
    log(f"  {'cost scenario':>48}  {'bound':>8}  {'variant':>30}"
        f"  {'C (EUR)':>9}  {'M (EUR)':>16}  {'p* (2dp)':>9}  {'p* (exact)':>11}")

    for label, lo, central, hi in COST_SCENARIOS:
        for bound, c in (("low", lo), ("central", central), ("high", hi)):
            if c is None:
                continue
            for vlabel, q_ev, lam in variants:
                v = value_per_event(c_event=c, q_event=q_ev, lambda_nok=lam)
                m = exposed_value(d=d, v=v)
                p = p_star(m, k)
                is_base = (label.startswith("Norwegian")
                           and vlabel == "base")
                mark = "   <- base" if is_base else ""
                log(f"  {label:>48}  {bound:>8}  {vlabel:>30}"
                    f"  {c:>9,.0f}  {m:>16,.0f}  {p:>9.2f}  {p:>11.6f}{mark}")
                rows.append({
                    "cost_scenario": label, "bound": bound, "variant": vlabel,
                    "c_event_eur": c, "q_event_qaly": q_ev, "lambda_nok": lam,
                    "v_eur": v, "m_eur": m,
                    "p_star_reported_2dp": round(p, 2), "p_star_exact": p,
                    "base_case": is_base,
                    "needs_verification": (label.startswith("Laroche")
                                           and VERIFY_LAROCHE),
                })


# ─────────────────────────────────────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(path: str, rows: list) -> None:
    if not rows:
        return
    import csv
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Paper 9 sensitivity analyses for editorial items 3, 7, 19, 21")
    ap.add_argument("--draws", type=int, default=50_000,
                    help="PSA draws (default 50000)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)),
                    help="output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    lines: list[str] = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 78)
    log("Paper 9 — sensitivity analyses for JMIR AI ms#109863 major revision")
    log(f"PSA draws: {args.draws:,}   seed: {args.seed}")
    log("=" * 78)
    log("")

    base = verify_base_case(log)

    r03: list = []
    r07a: list = []
    r07b: list = []
    r19: list = []
    r21: list = []

    item03_exchange_rate(log, r03)
    item07_oneway(log, r07a, base)
    item07_psa(log, r07b, base, args.draws, args.seed)
    item19_horizon(log, r19)
    item21_cost_event(log, r21)

    write_csv(os.path.join(args.outdir, "sens_item03_exchange_rate.csv"), r03)
    write_csv(os.path.join(args.outdir, "sens_item07_oneway.csv"), r07a)
    write_csv(os.path.join(args.outdir, "sens_item07_psa_summary.csv"), r07b)
    write_csv(os.path.join(args.outdir, "sens_item19_horizon.csv"), r19)
    write_csv(os.path.join(args.outdir, "sens_item21_cost_event.csv"), r21)

    # ── reporting precision guard ────────────────────────────────────────────
    all_p = []
    for r in r03:
        all_p.append(r["p_star"])
    for r in r07a:
        all_p += [r["p_star_low"], r["p_star_high"]]
    for r in r19:
        all_p.append(r["p_star"])
    for r in r21:
        all_p.append(r["p_star_exact"])
    psa = next((r for r in r07b if r["quantity"] == "p*"), None)
    if psa:
        all_p += [psa["p2_5"], psa["p97_5"]]

    lo, hi = min(all_p), max(all_p)
    rounded = {round(p, 2) for p in all_p}

    log("")
    log("=" * 78)
    log("REPORTING PRECISION")
    log("=" * 78)
    log(f"  p* across every analysis above: {lo:.6f} to {hi:.6f}")
    log(f"  span: {hi - lo:.2e}")
    log(f"  at manuscript precision (2 dp): "
        f"{', '.join(f'{v:.2f}' for v in sorted(rounded))}")
    if len(rounded) == 1:
        log("")
        log("  p* IS UNCHANGED AT 0.50 TO TWO DECIMALS IN EVERY SCENARIO.")
        log("  Report it that way. Do NOT quote six-decimal values in the")
        log("  manuscript or response letter — the six-decimal columns above")
        log("  exist only to make the movement visible at all, and quoting")
        log("  them would reintroduce the false precision removed at v36.")
    else:
        log("")
        log("  WARNING: p* changes at two decimals in at least one scenario.")
        log("  Identify which before writing the response.")
    log(f"  Structural regularisation-sweep range for comparison: 0.32 to 0.70")

    log("")
    log("=" * 78)
    log("FILES WRITTEN")
    for fn in ("sens_item03_exchange_rate.csv", "sens_item07_oneway.csv",
               "sens_item07_psa_summary.csv", "sens_item19_horizon.csv",
               "sens_item21_cost_event.csv", "sens_summary.txt"):
        log(f"  {os.path.join(args.outdir, fn)}")
    log("=" * 78)

    with open(os.path.join(args.outdir, "sens_summary.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
