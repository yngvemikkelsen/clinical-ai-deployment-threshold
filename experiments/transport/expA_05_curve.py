#!/usr/bin/env python3
"""
Paper 9 - Experiment A, step 5: the screening curve, held out and pooled.

Paper 12's screening derivation rests on 6 conditions (3 benchmark corpora x 2
query formats) from ONE benchmark. Cluster-robust inference on six clusters is
thin, and no intensity above six could be reported without extrapolating past
the observed range. Experiment A adds 4 conditions from two institutional
corpora - UCSF ED provider notes and BIDMC discharge summaries - giving 10.

TWO ANALYSES, ANSWERING DIFFERENT QUESTIONS
-------------------------------------------
held-out  Curve derived on Paper 12's 6 conditions only, then applied unchanged
          to the 4 external ones. Does the prevalence requirement transport?
          This is the external validation. It is not a bigger sample.

pooled    Curve recomputed on all 10. Tighter between-condition variance and a
          wider reportable range. This is what a deployer should use. It is
          in-sample and tests nothing.

The paper's argument is that findings do not transport unless checked, so the
held-out analysis is the one that carries weight; pooled is the improved
estimate that follows once transport is established.

THE DERIVATION
--------------
Screening accuracy is the probability that a finite local sample recovers the
correct SIGN of dMRR, not the accuracy of a proxy classifier. For model m with
mean effect d_m and between-condition SD s_m, a site scoring n conditions has
SE = s_m / sqrt(n), and

    P(sign error) = Phi(-|d_m| / (s_m / sqrt(n)))

    se = 1 - mean P(error) over tier 2      (should treat, does not)
    sp = 1 - mean P(error) over tier 1      (should withhold, does not)

Inverting the screen-and-treat condition gives the minimum prevalence at which
a screen of that accuracy has positive net effect:

    p_screen* = [(1-sp)|d_harm| + K_C/M] / [se*d_ben + (1-sp)|d_harm|]

reported here in the negligible-cost limit (K_C/M ~ 5e-6 in the main model,
shifting the requirement by <1.4e-4).

Inputs:
    --p12   epsilon_sensitivity.parquet from Paper 12, filtered to eps=1e-5
    --expa  expA_panel_clean.csv (deduplicated)

Usage:
    python expA_05_curve.py --p12 epsilon_sensitivity.parquet \\
                            --expa expA_panel_clean.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import norm

EPS = 1e-5
TIER1 = {"BioLORD-2023", "MedCPT", "BGE-base", "GTE-base",
         "Nomic-embed-text", "Nomic-embed-text-nopfx"}


def load_p12(path):
    d = pd.read_parquet(path)
    d = d[np.isclose(d["epsilon"], EPS)].copy()
    d["condition"] = d["corpus"] + "|" + d["query_format"]
    d["tier"] = np.where(d["model"].isin(TIER1), 1, 2)
    return d.rename(columns={"delta_MRR@10": "delta"})[
        ["model", "tier", "condition", "delta"]]


def load_expa(path):
    d = pd.read_csv(path)
    d = d[d["variant"] == "hpi"].copy()
    d["condition"] = d["corpus"] + "|" + d["query_format"]
    return d.rename(columns={"delta_MRR10": "delta"})[
        ["model", "tier", "condition", "delta"]]


def stats(df):
    """Per model: mean effect, between-condition SD, n conditions."""
    g = df.groupby(["model", "tier"])["delta"]
    return pd.DataFrame({"d": g.mean(), "s": g.std(ddof=1),
                         "n": g.size()}).reset_index()


def screen(st, n_cond):
    """se, sp at n scored conditions."""
    err = {}
    for t in (1, 2):
        sub = st[st.tier == t]
        err[t] = np.mean([norm.cdf(-abs(r.d) / (r.s / np.sqrt(n_cond)))
                          for r in sub.itertuples()])
    return 1 - err[2], 1 - err[1]


def p_screen_star(se, sp, d_ben, d_harm):
    b = (1 - sp) * abs(d_harm)
    return b / (se * d_ben + b)


def curve(st, label, d_ben=None, d_harm=None, ns=(1, 2, 3, 4, 5, 6, 8, 10)):
    d_ben = d_ben if d_ben is not None else st[st.tier == 2].d.mean()
    d_harm = d_harm if d_harm is not None else st[st.tier == 1].d.mean()
    n_obs = int(st.n.max())
    print(f"\n{label}")
    print(f"  d_ben {d_ben:+.4f}  d_harm {d_harm:+.4f}  "
          f"universal p* {abs(d_harm)/(d_ben+abs(d_harm)):.4f}  "
          f"| {n_obs} conditions observed")
    print(f"  {'n_cond':>7}{'se':>8}{'sp':>8}{'min prevalence':>17}   ")
    out = []
    for n in ns:
        se, sp = screen(st, n)
        ps = p_screen_star(se, sp, d_ben, d_harm)
        tag = "" if n <= n_obs else "  EXTRAPOLATED"
        out.append((n, se, sp, ps))
        print(f"  {n:>7}{se:>8.3f}{sp:>8.3f}{ps:>17.4f}{tag}")
    return pd.DataFrame(out, columns=["n_cond", "se", "sp", "p_screen_star"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p12", required=True)
    ap.add_argument("--expa", required=True)
    a = ap.parse_args()

    p12, expa = load_p12(a.p12), load_expa(a.expa)
    print(f"Paper 12 : {p12.condition.nunique()} conditions, "
          f"{p12.model.nunique()} models")
    print(f"Exp A    : {expa.condition.nunique()} conditions, "
          f"{expa.model.nunique()} models  "
          f"({', '.join(sorted(expa.condition.unique()))})")
    common = sorted(set(p12.model) & set(expa.model))
    if len(common) != 13:
        print(f"  WARNING only {len(common)} models in common: {common}")

    st12 = stats(p12[p12.model.isin(common)])
    stA = stats(expa[expa.model.isin(common)])
    stAll = stats(pd.concat([p12, expa])[
        lambda d: d.model.isin(common)] if False else
        pd.concat([p12[p12.model.isin(common)], expa[expa.model.isin(common)]]))

    print("\n" + "=" * 78)
    print("HELD OUT: curve derived on Paper 12, applied to the external corpora")
    print("=" * 78)
    c12 = curve(st12, "derived on Paper 12's 6 conditions")

    print("\n  external effect sizes, for comparison:")
    print(f"    Exp A  d_ben {stA[stA.tier==2].d.mean():+.4f}  "
          f"d_harm {stA[stA.tier==1].d.mean():+.4f}")
    print("  applying the Paper 12 screen accuracy to the external effects:")
    print(f"  {'n_cond':>7}{'min prevalence (external)':>28}")
    for n in (1, 2, 3, 4, 5, 6):
        se, sp = screen(st12, n)
        ps = p_screen_star(se, sp, stA[stA.tier == 2].d.mean(),
                           stA[stA.tier == 1].d.mean())
        print(f"  {n:>7}{ps:>28.4f}")
    print("\n  If these track the derived column, the requirement transports:")
    print("  a curve built on benchmark corpora predicts what a screen needs")
    print("  on institutional notes it never saw.")

    print("\n" + "=" * 78)
    print("POOLED: all 10 conditions")
    print("=" * 78)
    cAll = curve(stAll, "derived on all 10 conditions")

    print("\n" + "=" * 78)
    print("PER-MODEL BETWEEN-CONDITION VARIABILITY")
    print("=" * 78)
    m = st12.merge(stA, on=["model", "tier"], suffixes=("_p12", "_expA")) \
            .merge(stAll[["model", "s"]].rename(columns={"s": "s_pooled"}),
                   on="model")
    m = m.sort_values(["tier", "d_p12"])
    print(f"  {'model':<24}{'tier':>5}{'d p12':>9}{'d expA':>9}"
          f"{'SD p12':>9}{'SD expA':>9}{'SD pool':>9}")
    for r in m.itertuples():
        print(f"  {r.model:<24}{r.tier:>5}{r.d_p12:>+9.4f}{r.d_expA:>+9.4f}"
              f"{r.s_p12:>9.4f}{r.s_expA:>9.4f}{r.s_pooled:>9.4f}")
    print("\n  SD pool vs SD p12 is the point of Experiment A: 10 conditions")
    print("  instead of 6, and 4 of them from corpora the curve never saw.")

    c12.assign(basis="paper12").to_csv("expA_curve_heldout.csv", index=False)
    cAll.assign(basis="pooled10").to_csv("expA_curve_pooled.csv", index=False)
    print("\n-> expA_curve_heldout.csv, expA_curve_pooled.csv")


if __name__ == "__main__":
    main()
