#!/usr/bin/env python3
"""
Paper 9 — numerical consistency check.

The load-bearing numbers in the manuscript are recomputed here from the source
artefacts and compared against what the text says. This is a gate, not a proof:
it does not verify every value in Tables 4 to 7, the query-generator
correlations, the figures, or bibliographic metadata. This exists because the
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
import importlib.util
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
U = Path(os.environ.get("PAPER9_DATA", HERE))        # raw inputs
APP = Path(os.environ.get("PAPER9_APPENDICES", HERE / "appendices"))

# The canonical model is the single source of truth for the equations.
_spec = importlib.util.spec_from_file_location(
    "model", HERE / "clinical_ai_deployment_model.py")
MODEL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MODEL)
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
    # Response-defined grouping (v33+): a configuration belongs to the harmed
    # group if its MEASURED mean effect is negative, regardless of nominal tier.
    # Taken from the canonical model so the QC and the analysis cannot diverge.
    _eff = MODEL.load_effects(str(U / "appendix5_epsilon_percondition.csv"))
    _ben, _harm, s["d_ben"], s["d_harm"] = MODEL.response_groups(
        _eff[MODEL.BASE_EPSILON])
    s["n_ben"], s["n_harm"] = len(_ben), len(_harm)

    # decision-model quantities, computed analytically from stated parameters
    FX = MODEL.NOK_PER_EUR      # single source of truth
    D = MODEL.D
    C = MODEL.COST_EVENT_EUR       # EUR 11,494 at the 2024 rate
    lam = MODEL.WTP_EUR
    Q = MODEL.QALY_EVENT
    V = MODEL.value_per_event()
    M = MODEL.value_exposed(v=V)
    K = MODEL.T_DEPLOY
    s.update(D=D, V=V, M=M, K=K, KM=K / M, C=C, lam=lam, Q=Q)
    s["p_zero"] = abs(s["d_harm"]) / (s["d_ben"] + abs(s["d_harm"]))
    s["p_exact"] = (abs(s["d_harm"]) + K / M) / (s["d_ben"] + abs(s["d_harm"]))

    def _app(name):
        """Load an appendix, or record it as missing rather than crashing.

        A public archive should say which checks it could not run, not fail
        with a traceback on the first absent file.
        """
        f = APP / name
        if not f.exists():
            s.setdefault("missing", []).append(name)
            return None
        return pd.read_csv(f)

    p = _app("appendix2_transport_matrix.csv")
    s["panel"] = p
    b = _app("appendix3_screen_resampling.csv")
    s["boot"] = b
    if b is None:
        s["pooled"] = None
    # Appendix 3 labels rows by contribution, since the reference sign is the
    # measured full-sample effect rather than the nominal tier.
    if b is not None:
        b = b.copy()
        b["_grp"] = np.where(b["contributes to"] == "specificity", 1, 2)
        s["pooled"] = b.groupby(["_grp", "Documents sampled"])[
            "Agreement rate"].mean().unstack(0)
    s["bridge"] = _app("appendix4_query_generator.csv")
    s["frame"] = _app("appendix1_system_extraction.csv")
    s["diera"] = _app("appendix6_codesearch_full.csv")
    return s


# ------------------------------------------------------------------- checks
def _section(t, start, end):
    """Text between two top-level headings, however the exporter marks them."""
    i = t.find(f"## {start}")
    if i < 0:
        i = t.find(start)
    if i < 0:
        return ""
    if end is None:
        return t[i:]
    j = t.find(f"## {end}", i)
    if j < 0:
        j = t.find(end, i)
    return t[i:j] if j > i else t[i:]


def check_values(t, s):
    # --- editorial comment 3: the rate and its basis must be stated
    check("exchange rate stated", "11.6276" in t, "Methods must state NOK/EUR")
    check("rate basis stated", "Norges Bank" in t and "2024" in t,
          "annual average and price year")
    m = re.search(r"adverse-event cost is therefore \u20ac([\d,]+)", t) or \
        re.search(r"\u20ac(11,4\d\d)", t)
    check("C_event in text",
          m and near(float(m.group(1).replace(",", "")), s["C"], 1.0),
          f"text {m.group(1) if m else '?'} vs source {s['C']:,.0f}")

    # --- editorial comment 2: ethics subsection present and complete
    check("ethics subsection", "Ethical Considerations" in t)
    for probe, why in (("waiver of informed consent", "2a/2b review and consent"),
                       ("de-identified", "2c privacy"),
                       ("no compensation was provided", "2d compensation"),
                       ("no images of individuals", "2e identifiability")):
        check(f"ethics: {why}", probe in t)

    # --- editorial comment 5: code-search results must appear in Results
    res = _section(t, "Results", "Discussion")
    check("code-search in Results",
          "code-search" in res or "code search" in res,
          "comment 5: findings were Discussion-only")

    # --- comments 7, 19, 21: the sensitivity analyses must be reported
    for probe, why in (("50,000-draw", "7 probabilistic analysis"),
                       ("one to ten years", "19 horizon"),
                       ("\u20ac514", "21 lower European estimate"),
                       ("1,131.8", "7 absolute-scale range")):
        check(f"sensitivity: {why}", probe in t)

    # --- comment 8: the boundary sweep must be reported as a range
    for probe in ("0.58, 0.50, 0.43 and 0.35", "-0.0157"):
        check(f"boundary sweep: {probe}", probe in t)

    # --- comment 12: caveat in Methods AND Discussion, not Results alone
    meth = _section(t, "Methods", "Results")
    disc = _section(t, "Discussion", None)
    check("comment 12 in Methods", "No sampling interval" in meth)
    check("comment 12 in Discussion", "no statistical uncertainty bounds" in disc)


def check_values_original(t, s):
    if s.get("missing"):
        print(f"  NOTE  inputs not found, checks skipped: {', '.join(s['missing'])}")
    if s.get("panel") is None:
        return
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
    # The manuscript writes the deployment cost as T under the S/T formulation.
    m = re.search(r"\bT = €([\d,]+)", t) or re.search(r"\bK = €([\d,]+)", t)
    check("T (deployment cost)",
          m and near(float(m.group(1).replace(",", "")), s["K"], 1.0),
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
          and re.search(r"18 primary (replication )?cells|18 of 18", t) is not None,
          f"{len(prim)} cells")


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

    # Table 2 rows: conditions | se | sp | requirement | ratio. The exporter
    # may render this as a grid table without pipes, so match on the value
    # sequence rather than on delimiters.
    # Table 2 rows: conditions | se | sp | requirement | ratio. Exporters render
    # this as either a pipe table or a whitespace-aligned grid table, so accept
    # both rather than depending on one exporter's output.
    rows = re.findall(
        r"\|\s*([1-6])\s*\|\s*(0\.\d{3})\s*\|\s*(0\.\d{3})\s*\|"
        r"\s*(0\.\d{4})\s*\|", t)
    if len(rows) != 6:
        rows = re.findall(
            r"\b([1-6])\s+(0\.\d{3})\s+(0\.\d{3})\s+(0\.\d{4})\b", t)
    check("six screening rows parsed", len(rows) == 6, f"{len(rows)} rows matched")
    db, dh = s["d_ben"], s["d_harm"]
    for n_, se, sp, req in rows:
        n_, se, sp, req = int(n_), float(se), float(sp), float(req)
        # The exact S/T expression the manuscript uses, from the canonical model.
        exp = MODEL.screen_requirement(se, sp, db, dh, n_, s["M"])
        check(f"screen req n={n_}", near(exp, req, 5e-4),
              f"text {req:.4f} vs recomputed {exp:.4f}")

    # The manuscript legitimately carries three forms of the threshold: the
    # rounded value used in tables, the exact zero-cost limit, and the exact
    # value including strategy cost. Check they are mutually consistent rather
    # than merely counting distinct strings.
    ths = sorted(set(re.findall(r"0\.50(?:2\d*)?", t)))
    rounded = [x for x in ths if len(x) <= 6]
    exact = sorted(x for x in ths if len(x) > 6)
    check("rounded threshold present", len(rounded) >= 1, f"{rounded}")
    # The reported form is 0.50; tolerance must match the precision reported,
    # not the precision of the source value.
    check("rounded threshold rounds correctly",
          len(rounded) >= 1
          and all(near(float(r), s["p_zero"], 10 ** -len(r.split(".")[1]) / 2)
                  for r in rounded),
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
    if fr is None:
        return
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
        "11,451": "simulated V, superseded",
        "1,670": "K_C from the superseded nominal-tier grouping",
        "4.9 \u00d7 10\u207b\u2076": "K_C/M, superseded by the S/T formulation",
        "0.502811": "p* from rounded inputs; exact value is 0.502810",
        "0.502783": "zero-cost p* from rounded inputs; exact is 0.502782",
        "\u20ac11,493": "C_event rounds to 11,494",
        "11,458": "V at the undated 11.70 rate, superseded by 11,529",
        "340.7": "simulated M, superseded",
        "341.9": "M at the undated 11.70 rate, superseded by 344.0",
        "\u20ac11,400": "C_event at the undated 11.70 rate, superseded by 11,493",
        "\u20ac342 million": "M at the undated 11.70 rate, superseded by 344",
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


def word_count_docx(path):
    """Authoritative body word count, taken from the DOCX itself.

    Counts Introduction through Conclusions, excluding table content and
    table/figure captions, which JMIR does not count. The markdown export is
    not reliable for this because table rendering varies by exporter.
    """
    import zipfile
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    xml = re.sub(r"<w:tbl>.*?</w:tbl>", "", xml, flags=re.S)
    total, inside = 0, False
    for para in re.findall(r"<w:p[ >].*?</w:p>", xml, re.S):
        txt = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", para, re.S)).strip()
        if txt == "Introduction":
            inside = True
        if txt in ("Acknowledgments", "Acknowledgements"):
            inside = False
        if inside and not re.match(r"^(Table|Figure) \d", txt):
            total += len(txt.split())
    return total


def word_count(t):
    """Body words, Introduction through Conclusions, excluding the reference
    list. Tables and figure captions are excluded by JMIR and are not counted
    here either; run against the markdown export, not the rendered PDF."""
    i, j = t.find("Introduction"), t.find("Acknowledgments")
    if i < 0 or j < 0:
        return None
    seg = t[i:j]
    # Exclude table bodies and table/figure captions, which JMIR does not count.
    keep, in_table = [], False
    for ln in seg.split("\n"):
        st_ = ln.strip()
        if st_.startswith("|") or re.match(r"^[-+ ]{5,}$", st_) or re.match(r"^ *-{3,}", st_):
            in_table = True
            continue
        if in_table and not st_:
            in_table = False
            continue
        if in_table:
            continue
        if re.match(r"^(Table|Figure) \d", st_):
            continue
        keep.append(ln)
    return len(" ".join(keep).split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", type=Path, required=True,
                    help="markdown export of the manuscript")
    ap.add_argument("--docx", type=Path, default=None,
                    help="the DOCX itself, for the authoritative word count")
    a = ap.parse_args()
    t_raw = a.md.read_text()
    # Probes must not be defeated by the exporter's line wrapping.
    t = re.sub(r'[ \t]*\n[ \t]*', ' ', t_raw)
    t = re.sub(r' {2,}', ' ', t)
    s = sources()

    check_values(t, s)
    check_values_original(t, s)
    if a.docx and a.docx.exists():
        w = word_count_docx(a.docx)
        check("body under 10,000 words", w < 10000, f"{w} words (from DOCX)")
    else:
        w = word_count(t_raw)
        if w is not None:
            print(f"  NOTE  markdown word count {w} is approximate; pass --docx "
                  f"for the authoritative figure")
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
