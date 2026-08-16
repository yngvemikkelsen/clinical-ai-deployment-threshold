#!/usr/bin/env python3
"""
Paper 9 - Experiment A, step 7: bound the query-generation substitution.

The external corpora are credentialed-access, so queries had to be generated
locally with qwen2.5:14b-instruct rather than GPT-4o, with three further
deviations documented in Methods. The obvious reviewer question is whether the
external result rides on that substitution.

This answers it on MTSamples - a PUBLIC corpus for which the companion study's
GPT-4o queries already exist - by running the identical 13-configuration panel
twice, once per query set, and comparing what actually matters: whether the
configurations rank the same way.

WHAT THIS TESTS
---------------
Rank stability of dMRR@10 across query derivation, not query equivalence. Two
models generating from the same extracted metadata will phrase differently; low
lexical overlap is expected and is not failure. The benchmark is the companion
line of work's own finding of Kendall tau 0.59-0.90 across query derivations
(mean 0.76, all P<=.004).

WHAT IT DOES NOT TEST
---------------------
Whether the EXTERNAL result would change. MTSamples is not one of the external
corpora, and no GPT-4o queries exist for those - that is why the substitution
was necessary. What is bounded is whether the query model reorders
configurations on a corpus where both sets are available.

Also reported: agreement on the quantity the paper actually uses, which is the
SIGN of dMRR, not its rank. A query set that reorders configurations but assigns
every one to the same subgroup leaves every conclusion in this paper intact.

Usage:
    python expA_07_bridge.py --prepare        # writes corpus + GPT-4o query CSVs
    python expA_02_queries_v2.py --corpus mtsamples
    python expA_07_bridge.py --run            # panel under both query sets
    python expA_07_bridge.py --compare
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

N_DOCS = 100
CORP = Path("./expA_corpora")
QUER = Path("./expA_queries")
OUT = Path("./expA_bridge")
QUERY_MODEL = "qwen2.5-14b-instruct"


# ------------------------------------------------------------------ prepare
def prepare(csv_path: Path, json_path: Path):
    """Write MTSamples into the panel's expected shape, plus the GPT-4o queries.

    The companion study's queries are positional lists of 100 with no document
    identifier, so alignment is by row order against the first 100 rows of the
    500-row sample. Verified by spot-checking specialty against query content.
    """
    d = pd.read_csv(csv_path).head(N_DOCS).reset_index(drop=True)
    q = json.load(open(json_path))["MTSamples"]
    if not (len(q["keyword"]) == len(q["natural_language"]) == N_DOCS):
        raise SystemExit("GPT-4o query lists are not both length 100")

    CORP.mkdir(exist_ok=True)
    QUER.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)

    corpus = pd.DataFrame({
        "doc_id": [f"mts_{i:03d}" for i in range(len(d))],
        # the panel reads text_hpi for the primary variant; MTSamples documents
        # are the unit the companion study encoded, so both columns hold the
        # same text and no section extraction is applied
        "text_full": d["text"].astype(str),
        "text_hpi": d["text"].astype(str),
        "hpi_source": "whole_document",
        "n_chars_full": d["text"].astype(str).str.len(),
        "n_chars_hpi": d["text"].astype(str).str.len(),
        "meta_specialty": d["specialty"].astype(str).str.strip(),
        "meta_note_type": "Clinical Note",
        "meta_primary_diagnosis": d["description"].astype(str).str.strip(),
        "meta_secondary": "none",
        "meta_demographics": "adult",
    })
    f = CORP / f"mtsamples_n{N_DOCS}_seed42_variants.csv"
    corpus.to_csv(f, index=False)
    print(f"corpus -> {f}  ({len(corpus)} docs, "
          f"median {int(corpus.n_chars_full.median()):,} chars)")

    gpt = pd.DataFrame({
        "doc_id": corpus["doc_id"],
        "query_nl": q["natural_language"],
        "query_keyword": q["keyword"],
        "md_primary_diagnosis": corpus["meta_primary_diagnosis"],
        "md_json_parsed": True,
        "registry_diagnosis": "",
    })
    g = QUER / "mtsamples_queries_gpt4o.csv"
    gpt.to_csv(g, index=False)
    print(f"gpt4o  -> {g}")
    print("\nnext: python expA_02_queries_v2.py --corpus mtsamples")
    print("      (~17 min; writes mtsamples_queries_"
          f"{QUERY_MODEL}.csv)")


# ---------------------------------------------------------------------- run
def run_panel():
    """Panel on MTSamples under each query set. Imports the panel module so the
    encoding, ZCA and MRR are byte-identical to the main analysis."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("pnl", "expA_04_panel.py")
    pnl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pnl)

    docs = pd.read_csv(CORP / f"mtsamples_n{N_DOCS}_seed42_variants.csv")
    sets = {}
    for tag, fn in (("gpt4o", "mtsamples_queries_gpt4o.csv"),
                    ("local", f"mtsamples_queries_{QUERY_MODEL}.csv")):
        p = QUER / fn
        if not p.exists():
            raise SystemExit(f"missing {p}")
        sets[tag] = pd.read_csv(p)

    rows = []
    for cfg in pnl.PANEL:
        de = pnl.encode(list(docs["text_hpi"].astype(str)), cfg,
                        cfg.d_prefix, cfg.max_len, bs=8)
        W, mu = pnl.fit_zca(de)
        dz = pnl.apply_zca(de, W, mu)
        for tag, qdf in sets.items():
            for fmt in ("nl", "keyword"):
                qe = pnl.encode(list(qdf[f"query_{fmt}"].astype(str)), cfg,
                                cfg.q_prefix, cfg.max_len,
                                query_side=True, bs=8)
                base = pnl.mrr_at_k(qe, de)
                zca = pnl.mrr_at_k(pnl.apply_zca(qe, W, mu), dz)
                rows.append(dict(query_set=tag, query_format=fmt,
                                 model=cfg.name, tier=cfg.tier,
                                 baseline_MRR10=base, zca_MRR10=zca,
                                 delta_MRR10=zca - base))
        print(f"  {cfg.name:<24} done")
    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "mtsamples_both_query_sets.csv", index=False)
    print(f"-> {OUT / 'mtsamples_both_query_sets.csv'}")
    return df


# ------------------------------------------------------------------ compare
def compare(df=None):
    if df is None:
        df = pd.read_csv(OUT / "mtsamples_both_query_sets.csv")
    print("\n" + "=" * 84)
    print("RANK STABILITY OF dMRR@10 ACROSS QUERY DERIVATION")
    print("=" * 84)
    print(f"{'format':<12}{'n':>4}{'Kendall tau':>13}{'P':>9}{'Spearman':>11}"
          f"{'vs 0.59-0.90':>15}")
    for fmt in sorted(df.query_format.unique()):
        a = df[(df.query_set == "gpt4o") & (df.query_format == fmt)] \
            .set_index("model")["delta_MRR10"]
        b = df[(df.query_set == "local") & (df.query_format == fmt)] \
            .set_index("model")["delta_MRR10"]
        i = a.index.intersection(b.index)
        t, p = kendalltau(a[i], b[i])
        s, _ = spearmanr(a[i], b[i])
        band = "inside" if 0.59 <= t <= 0.90 else ("above" if t > 0.90 else "BELOW")
        print(f"{fmt:<12}{len(i):>4}{t:>13.3f}{p:>9.4f}{s:>11.3f}{band:>15}")

    print("\n" + "=" * 84)
    print("SUBGROUP AGREEMENT - the quantity the paper actually uses")
    print("=" * 84)
    print(f"{'format':<12}{'query set':<10}{'tier1 mean':>12}{'tier2 mean':>12}"
          f"{'all correct':>13}{'p*':>9}")
    for fmt in sorted(df.query_format.unique()):
        for tag in ("gpt4o", "local"):
            g = df[(df.query_set == tag) & (df.query_format == fmt)]
            t1, t2 = g[g.tier == 1].delta_MRR10, g[g.tier == 2].delta_MRR10
            ok = int((t1 < 0).sum() + (t2 > 0).sum())
            ps = (abs(t1.mean()) / (t2.mean() + abs(t1.mean()))
                  if t2.mean() + abs(t1.mean()) > 0 else np.nan)
            print(f"{fmt:<12}{tag:<10}{t1.mean():>+12.4f}{t2.mean():>+12.4f}"
                  f"{ok:>9}/{len(g)}{ps:>9.4f}")

    print("\n  Rank stability inside or above the benchmark band bounds the")
    print("  substitution. But subgroup agreement is the stronger test: if both")
    print("  query sets assign every configuration to the same subgroup, the")
    print("  query model cannot have produced this paper's conclusions, whatever")
    print("  it does to the ordering within a subgroup.")

    dis = []
    for fmt in sorted(df.query_format.unique()):
        a = df[(df.query_set == "gpt4o") & (df.query_format == fmt)] \
            .set_index("model")
        b = df[(df.query_set == "local") & (df.query_format == fmt)] \
            .set_index("model")
        for m in a.index.intersection(b.index):
            if np.sign(a.loc[m, "delta_MRR10"]) != np.sign(b.loc[m, "delta_MRR10"]):
                dis.append((fmt, m, a.loc[m, "delta_MRR10"], b.loc[m, "delta_MRR10"]))
    if dis:
        print("\n  SIGN DISAGREEMENTS:")
        for f, m, x, y in dis:
            print(f"    {f:<10}{m:<24}gpt4o {x:+.4f}  local {y:+.4f}")
    else:
        print("\n  No sign disagreements in any cell.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--csv", type=Path, default=Path("mtsamples_sample.csv"))
    ap.add_argument("--json", type=Path, default=Path("metadata_queries.json"))
    a = ap.parse_args()
    if a.prepare:
        prepare(a.csv, a.json)
    if a.run:
        compare(run_panel())
    elif a.compare:
        compare()
    if not any((a.prepare, a.run, a.compare)):
        ap.print_help()


if __name__ == "__main__":
    main()
