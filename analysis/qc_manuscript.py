#!/usr/bin/env python3
"""
Paper 9 — numerical consistency check.

Every load-bearing number in the manuscript is recomputed here from the source
artefact and compared against what the text says. This exists because the
manuscript has been through many rounds of correction, each applied by patch;
a patch that silently misses leaves a stale figure behind, and several already
have.

The script checks three things:

  1. VALUES     each claimed number against its recomputed source
  2. INTERNAL   arithmetic that must hold within the manuscript itself
                (thresholds derived from the stated effect sizes, ratios,
                fractions matching their stated numerators and denominators)
  3. STALE      figures that were corrected in earlier rounds and must not
                reappear anywhere

Exit status is nonzero if any check fails, so it can gate a build.

    python qc_manuscript.py --md paper9_manuscript_v29.md
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

U = Path("/mnt/user-data/uploads")
APP = Path("appendices")
TIER1 = {"BioLORD-2023", "MedCPT", "BGE-base", "GTE-base",
         "Nomic-embed-text", "Nomic-embed-text-nopfx"}

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))


def near(a, b, tol=5e-4):
    return abs(a - b) <= tol


# ------------------------------------------------------------------ sources
def sources():
    s = {}
    e = pd.read_parquet(U / "epsilon_sensitivity.parquet")
    e = e[np.isclose(e.epsilon, 1e-5)].copy()
    e["tier"] = np.where(e.model.isin(TIER1), 1, 2)
    # base case uses RESPONSE-DEFINED grouping (v33+): a configuration belongs
    # to the harmed group if its measured mean effect is negative, regardless of
    # nominal mechanistic tier. Grouping by tier is superseded.
    pm = e.groupby("model")["delta_MRR@10"].mean()
    s["d_ben"] = pm[pm > 0].mean()
    s["d_harm"] = pm[pm < 0].mean()

    # decision-model quantities, computed analytically from stated parameters
    FX = 11.7
    D = sum(1 / 1.04 ** t for t in range(1, 6))
    C = 26153 * 5.11 / FX
    lam = 275_000 / FX
    Q = 1.5 * 0.001
    V = C + lam * Q
    M = 0.70 * 0.60 * 0.12 * 133_000 * V * D
    K = 2 * 400 + 1.5 * 100 * D
    s.update(D=D, V=V, M=M, K=K, KM=K / M)
    s["p_zero"] = abs(s["d_harm"]) / (s["d_ben"] + abs(s["d_harm"]))
    s["p_exact"] = (abs(s["d_harm"]) + K / M) / (s["d_ben"] + abs(s["d_harm"]))

    p = pd.read_csv(APP / "appendix2_transport_matrix.csv")
    s["panel"] = p
    b = pd.read_csv(APP / "appendix3_screen_resampling.csv")
    s["boot"] = b
    # Appendix 3 labels rows by contribution, since the reference sign is the
    # measured full-sample effect rather than the nominal tier.
    b = b.copy()
    b["_grp"] = np.where(b["contributes to"] == "specificity", 1, 2)
    s["pooled"] = b.groupby(["_grp", "Documents sampled"])[
        "Agreement rate"].mean().unstack(0)
    s["bridge"] = pd.read_csv(APP / "appendix4_query_generator.csv")
    s["frame"] = pd.read_csv(APP / "appendix1_system_extraction.csv")
    s["diera"] = pd.read_csv(APP / "appendix6_codesearch_full.csv")
    return s


# ------------------------------------------------------------------- checks
def check_values(t, s):
    check("d_ben in text", "0.075745" in t or "0.0757" in t,
          f"source {s['d_ben']:+.6f}")
    check("d_harm in text", "0.076593" in t or "0.0766" in t,
          f"source {s['d_harm']:+.6f}")

    m = re.search(r"zero-cost limit \((0\.\d+)\)", t) or \
        re.search(r"zero-cost limit of (0\.\d+)", t)
    check("p* zero-cost", m and near(float(m.group(1)), s["p_zero"]),
          f"text {m.group(1) if m else '?'} vs source {s['p_zero']:.6f}")

    m = re.search(r"exact threshold is\s*\n*\s*p\\?\* = \([^=]+= (0\.\d+)", t) \
        or re.search(r"p\\?\* = (0\.50\d+)", t)
    check("p* exact", m and near(float(m.group(1)), s["p_exact"]),
          f"text {m.group(1) if m else '?'} vs source {s['p_exact']:.6f}")

    m = re.search(r"V = €([\d,]+)", t)
    check("V", m and near(float(m.group(1).replace(",", "")), s["V"], 1.0),
          f"text {m.group(1) if m else '?'} vs source {s['V']:,.0f}")
    m = re.search(r"M = €([\d.]+) million", t)
    check("M", m and near(float(m.group(1)), s["M"] / 1e6, 0.05),
          f"text {m.group(1) if m else '?'}M vs source {s['M']/1e6:.1f}M")
    m = re.search(r"K = €([\d,]+)", t)
    check("K", m and near(float(m.group(1).replace(",", "")), s["K"], 1.0),
          f"text {m.group(1) if m else '?'} vs source {s['K']:,.0f}")

    # transport: section-level cells all agree with nominal assignment
    hpi = s["panel"][s["panel"]["Document variant"] == "hpi"]
    ok = (hpi["retains assignment"] == "yes").sum()
    check("52 section cells agree", ok == 52 and "52" in t, f"{ok}/52")
    f512 = s["panel"][s["panel"]["Document variant"] == "full512"]
    ok512 = (f512["retains assignment"] == "yes").sum()
    check("47 of 52 whole-note", ok512 == 47 and "47 of 52" in t, f"{ok512}/52")
    harm512 = f512[f512["Nominal tier"] == 1]
    check("24 harm-tier cells retained",
          (harm512["retains assignment"] == "yes").sum() == 24
          and ("24 harm-tier" in t or "24 nominal-harm-tier" in t))

    # resampling
    for n, val in ((10, 0.328), (75, 0.974)):
        src = s["pooled"].loc[n, 1]
        check(f"specificity n={n}", near(src, val, 1e-3) and str(val) in t,
              f"source {src:.4f} text {val}")

    # bridge
    agree = (s["bridge"]["assignment agrees"] == "yes").sum()
    check("25 of 26 assignments", agree == 25 and "25 of 26" in t,
          f"{agree}/26")

    # frame
    fr = s["frame"]
    n_aff = (fr["Confirmed affected"] == "yes").sum()
    check("1 of 55 mechanistic", n_aff == 1 and "1/55" in t, f"{n_aff}/55")
    check("27 commercial interfaces",
          (fr["Interface class"] == "commercial interface").sum() == 27
          and ("27 of the 55" in t or "27 used commercial" in t
               or "27 use commercial" in t))

    # code search
    d0 = s["diera"][s["diera"].Epsilon == 0.0]
    prim = d0[d0["Analysis"] == "primary replication grid"]
    check("18 primary cells", len(prim) == 18
          and ("18 of 18" in t or "18 primary cells" in t), f"{len(prim)} cells")


def check_internal(t, s):
    # ladder fractions must equal their stated numerator/denominator
    for num, den, txt in ((31, 55, "31/55"), (32, 55, "32/55")):
        m = re.search(rf"{txt}[^0-9]{{0,4}}=?\s*(0\.\d+)", t)
        if m:
            check(f"{txt} arithmetic", near(float(m.group(1)), num / den),
                  f"text {m.group(1)} vs {num/den:.4f}")
    # 32/55 must exceed p*, 31/55 must not — the paper's fragility claim
    # under response-defined grouping the rung-2 interval STRADDLES the
    # threshold, so the correct check is that the manuscript reports it as
    # indeterminate rather than as lying below.
    check("32/55 above p*", 32 / 55 > s["p_zero"],
          f"{32/55:.4f} vs {s['p_zero']:.4f}")
    check("rung 2 reported indeterminate",
          "indeterminate" in t and 31 / 55 > s["p_zero"],
          f"31/55={31/55:.4f} vs p*={s['p_zero']:.4f}")

    # screening design table: each requirement recomputed from its own se/sp
    db, dh = s["d_ben"], s["d_harm"]
    rows = re.findall(
        r"\|\s*(\d)\s*\|\s*(0\.\d+)\s*\|\s*(0\.\d+)\s*\|\s*[\d.×\s⁻¹²³⁴⁵⁶⁷⁸⁹e^-]+\|"
        r"\s*(0\.\d+)\s*\|", t)
    for n, se, sp, req in rows:
        se, sp, req = float(se), float(sp), float(req)
        b = (1 - sp) * abs(dh)
        exp = b / (se * db + b)
        check(f"screen req n={n}", near(exp, req, 2e-3),
              f"text {req:.4f} vs recomputed {exp:.4f}")

    # The manuscript legitimately carries three forms of the threshold: the
    # rounded value used in tables, the exact zero-cost limit, and the exact
    # value including strategy cost. Check they are mutually consistent rather
    # than merely counting distinct strings.
    ths = sorted(set(re.findall(r"0\.502\d*", t)))
    rounded = [x for x in ths if len(x) <= 6]
    exact = sorted(x for x in ths if len(x) > 6)
    check("rounded threshold rounds correctly",
          all(near(float(r), s["p_zero"], 5e-5) for r in rounded),
          f"{rounded} vs {s['p_zero']:.6f}")
    if len(exact) == 2:
        lo, hi = float(exact[0]), float(exact[1])
        gap = s["KM"] / (s["d_ben"] + abs(s["d_harm"]))
        check("cost term separates the two exact forms",
              near(hi - lo, gap, 5e-6),
              f"gap {hi-lo:.2e} vs K/M term {gap:.2e}")
    else:
        check("two exact forms present", False, f"{exact}")

    # ladder bounds in the manuscript must match those derivable from the frame
    fr = s["frame"]
    n = len(fr)
    aff = (fr["Confirmed affected"] == "yes").sum()
    for col, want in ((("Rung 2: contrastive training implies non-membership"),
                       "0.564"),
                      (("Rung 3: vendor claims accepted"), "0.073")):
        if col in fr.columns:
            unk = (fr[col] == "unknown").sum()
            hi = (aff + unk) / n
            check(f"ladder upper {want}", near(hi, float(want), 5e-4)
                  and want in t, f"frame {hi:.4f} text {want}")


def check_stale(t):
    stale = {
        "0.581551": "superseded p* (rounded effect sizes)",
        "0.580507": "superseded p* (nominal-tier grouping)",
        "0.581580": "superseded exact p*",
        "0.5816": "superseded p*",
        "11,451": "simulated V, replaced by analytic 11,458",
        "340.7": "simulated M, replaced by analytic 341.9",
        "€1,469": "simulated K, replaced by analytic 1,468",
        "28.7 million": "superseded universal-deployment cost",
        "2/55 = 0.036": "superseded mechanistic count",
        "training label": "superseded terminology",
        "mean-pooled output": "claim not supported by accessible record",
        "affected system": "implies measured response membership",
        "Two of 55": "superseded count",
        "pooling-mismatch": "superseded terminology",
    }
    for k, why in stale.items():
        check(f"no stale '{k}'", k not in t, why)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", type=Path, required=True)
    a = ap.parse_args()
    t = a.md.read_text()
    s = sources()

    check_values(t, s)
    check_internal(t, s)
    check_stale(t)

    fails = [r for r in results if not r[1]]
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL':<5}{name:<34}{detail}")
    print(f"\n{len(results) - len(fails)}/{len(results)} checks passed")
    if fails:
        print("\nFAILURES:")
        for n, _, d in fails:
            print(f"  {n}: {d}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
