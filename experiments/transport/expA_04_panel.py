#!/usr/bin/env python3
"""
Paper 9 - Experiment A, step 4: the 13-model panel on the external corpora.

Produces the conditions that extend Paper 12's screening curve from 6 observed
to 6 + N. For each condition it measures corpus-only ZCA whitening at the
companion's default epsilon and reports dMRR@10 per model - the quantity the
screening derivation consumes.

CONDITIONS
----------
    2 corpora  x  3 document variants  x  2 query formats  =  12

    corpora   er_reason (UCSF ED provider notes)
              mimic_discharge (BIDMC discharge summaries)
    variants  hpi        section extract, 173/317 tokens median
              full       whole note, each model's NATIVE max_length
              full512    whole note, every model capped at 512 tokens
    formats   natural language, keyword

WHY THREE VARIANTS
------------------
Paper 12's corpora are ~400 tokens, so every model encoded every document
whole. These are 1,685 and 2,443 tokens median, and the panel's max_length is
not uniform: 512 for the nine BERT-scale models, 2048 for BioMistral, 4096 for
E5-Mistral and Phi-3 (Paper 3 Multimedia Appendix 2; Paper 12 Methods).

That matters because the split is not random with respect to tier. Tier 1 -
BioLORD, MedCPT, BGE, GTE, Nomic, Nomic-nopfx - is entirely BERT-scale at 512.
Tier 2 contains all four LLM-scale models. So on a full-note variant with
native limits, tier 2 models see whole documents while tier 1 sees 20-31% of
them: a systematic advantage to tier 2 unrelated to whitening, running in the
same direction as the effect being measured.

    hpi      no confound - everything fits for every model
    full     faithful to the published encoding, confound present
    full512  identical text for every model, deviates from published encoding

If full512 behaves like full, the asymmetry is immaterial and can be reported
as such. If it does not, the confound has been measured rather than argued
about.

CONFIGURATION
-------------
Verbatim from Paper 3 Multimedia Appendix 2 (doi:10.2196/94241) and Paper 12
Table 1 (doi:10.2196/99639): pooling, prefixes, dtype, max_length, right-side
truncation, L2 normalisation on all dense output. MedCPT is a dual encoder -
queries and documents go through different checkpoints.

ZCA
---
    W = U(Lam + eps I)^(-1/2) U^T,  x' = W(x - mu_D)/||W(x - mu_D)||
W and mu_D are fit on DOCUMENT embeddings only; queries are transformed after
with the same W. eps = 1e-5, the companion's default. Fit on all 100 documents
(the deployment-realistic protocol), not five-fold CV - see the manuscript's
Limitations on why the CV estimates are retained only as sensitivity.

Usage:
    python expA_04_panel.py --probe                 # 2 models, 1 condition
    python expA_04_panel.py --variant hpi
    python expA_04_panel.py                         # all 12 conditions
"""
from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

SEED = 42
EPSILON = 1e-5
CORP = Path("./expA_corpora")
QUER = Path("./expA_queries")
OUT = Path("./expA_panel")
QUERY_MODEL = "qwen2.5-14b-instruct"

DEV = ("cuda" if torch.cuda.is_available() else
       "mps" if torch.backends.mps.is_available() else "cpu")


@dataclass
class Cfg:
    name: str
    ckpt: str
    pool: str                 # mean | cls | eos
    tier: int                 # 1 harmed, 2 benefits (Paper 12 Table 3)
    max_len: int = 512
    dtype: str = "fp32"
    q_prefix: str = ""
    d_prefix: str = ""
    ckpt_query: str = ""      # dual encoder only


# Paper 3 Appendix 2 + Paper 12 Table 1. OpenAI-emb3-small and BM25 are excluded:
# the former has no hidden states and was dropped from Paper 12's panel, the
# latter is not an embedding model and cannot be whitened.
PANEL = [
    Cfg("BioBERT", "dmis-lab/biobert-v1.1", "mean", 2),
    Cfg("ClinicalBERT", "medicalai/ClinicalBERT", "mean", 2),
    Cfg("BERT-base-uncased", "bert-base-uncased", "mean", 2),
    Cfg("BioLORD-2023", "FremyCompany/BioLORD-2023", "mean", 1),
    Cfg("MedCPT", "ncbi/MedCPT-Article-Encoder", "cls", 1,
        ckpt_query="ncbi/MedCPT-Query-Encoder"),
    Cfg("BGE-base", "BAAI/bge-base-en-v1.5", "mean", 1),
    Cfg("GTE-base", "thenlper/gte-base", "mean", 1),
    Cfg("Nomic-embed-text", "nomic-ai/nomic-embed-text-v1.5", "mean", 1,
        q_prefix="search_query: ", d_prefix="search_document: "),
    Cfg("Nomic-embed-text-nopfx", "nomic-ai/nomic-embed-text-v1.5", "mean", 1),
    Cfg("E5-Mistral-7B", "intfloat/e5-mistral-7b-instruct", "eos", 2,
        max_len=4096, dtype="fp16",
        q_prefix='Instruct: "Given a clinical note, retrieve the most relevant '
                 'clinical document."\n'),
    Cfg("E5-Mistral-7B-ablation", "intfloat/e5-mistral-7b-instruct", "mean", 2,
        max_len=4096, dtype="fp16"),
    Cfg("Phi-3-mini", "microsoft/Phi-3-mini-4k-instruct", "mean", 2,
        max_len=4096, dtype="fp16"),
    Cfg("BioMistral-7B", "BioMistral/BioMistral-7B", "mean", 2,
        max_len=2048, dtype="fp16"),
]

VARIANTS = {"hpi": ("text_hpi", None),        # (column, max_len override)
            "full": ("text_full", None),
            "full512": ("text_full", 512)}


# ------------------------------------------------------------------ encoding
def pool(out, mask, how):
    h = out.last_hidden_state
    if how == "cls":
        return h[:, 0]
    if how == "eos":
        idx = mask.sum(1) - 1
        return h[torch.arange(h.size(0), device=h.device), idx]
    m = mask.unsqueeze(-1).expand(h.size()).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)


def encode(texts, cfg: Cfg, prefix: str, max_len: int, query_side=False, bs=8):
    ckpt = cfg.ckpt_query if (query_side and cfg.ckpt_query) else cfg.ckpt
    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "right"          # Appendix 2: right for all models
    kw = {"trust_remote_code": True}
    if cfg.dtype == "fp16":
        kw["torch_dtype"] = torch.float16
    model = AutoModel.from_pretrained(ckpt, **kw).to(DEV).eval()

    texts = [prefix + t for t in texts]
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt").to(DEV)
        with torch.no_grad():
            p = pool(model(**enc), enc["attention_mask"], cfg.pool)
        out.append(p.detach().cpu().float().numpy())
    E = np.concatenate(out, 0)
    del model
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return E / np.linalg.norm(E, axis=1, keepdims=True).clip(1e-9)   # L2, App 2


# ----------------------------------------------------------------- ZCA + MRR
def fit_zca(D, eps=EPSILON):
    mu = D.mean(0)
    Dc = D - mu
    U, Lam, _ = np.linalg.svd(np.cov(Dc, rowvar=False))
    return U @ np.diag(1.0 / np.sqrt(Lam + eps)) @ U.T, mu


def apply_zca(X, W, mu):
    Y = (X - mu) @ W.T
    return Y / np.linalg.norm(Y, axis=1, keepdims=True).clip(1e-9)


def mrr_at_k(Q, D, k=10):
    """Known-item: query i's target is document i. Paper 12's convention -
    higher cosine is better, rank counts strictly-better competitors."""
    S = Q @ D.T
    correct = np.diag(S)[:, None]
    rank = (S > correct).sum(1) + 1
    rr = np.where(rank <= k, 1.0 / rank, 0.0)
    return float(rr.mean())


# --------------------------------------------------------------------- panel
def load(corpus):
    d = pd.read_csv(next(CORP.glob(f"{corpus}_n*_variants.csv")))
    q = pd.read_csv(QUER / f"{corpus}_queries_{QUERY_MODEL}.csv")
    m = d.merge(q, on="doc_id", how="inner")
    if len(m) != len(d):
        raise SystemExit(f"{corpus}: {len(d)} docs but {len(m)} matched queries")
    return m


def run(corpora, variants, panel, bs=8):
    rows = []
    for corpus in corpora:
        df = load(corpus)
        print(f"\n=== {corpus}: {len(df)} documents ===")
        for cfg in panel:
            t0 = time.time()
            # queries are metadata-derived and shared across variants
            qe = {f: encode(list(df[f"query_{f}"]), cfg, cfg.q_prefix,
                            cfg.max_len, query_side=True, bs=bs)
                  for f in ("nl", "keyword")}
            for var in variants:
                col, override = VARIANTS[var]
                ml = override or cfg.max_len
                de = encode(list(df[col].astype(str)), cfg, cfg.d_prefix, ml, bs=bs)
                W, mu = fit_zca(de)
                dz = apply_zca(de, W, mu)
                for f in ("nl", "keyword"):
                    base = mrr_at_k(qe[f], de)
                    zca = mrr_at_k(apply_zca(qe[f], W, mu), dz)
                    rows.append(dict(corpus=corpus, variant=var, query_format=f,
                                     model=cfg.name, tier=cfg.tier,
                                     max_len_used=ml,
                                     baseline_MRR10=base, zca_MRR10=zca,
                                     delta_MRR10=zca - base))
            print(f"  {cfg.name:<24}{time.time()-t0:>6.1f}s")
    return pd.DataFrame(rows)


def report(df):
    print("\n" + "=" * 92)
    print("dMRR@10 BY CONDITION AND TIER (corpus-only ZCA, eps=1e-5)")
    print("=" * 92)
    print(f"{'corpus':<18}{'variant':<9}{'format':<9}{'tier1 mean':>12}"
          f"{'tier2 mean':>12}{'t1 all<0':>10}{'t2 all>0':>10}{'p*':>9}")
    for (c, v, f), g in df.groupby(["corpus", "variant", "query_format"]):
        t1 = g[g.tier == 1].delta_MRR10
        t2 = g[g.tier == 2].delta_MRR10
        ps = (abs(t1.mean()) / (t2.mean() + abs(t1.mean()))
              if t2.mean() + abs(t1.mean()) > 0 else np.nan)
        print(f"{c:<18}{v:<9}{f:<9}{t1.mean():>+12.4f}{t2.mean():>+12.4f}"
              f"{str(bool((t1 < 0).all())):>10}{str(bool((t2 > 0).all())):>10}"
              f"{ps:>9.4f}")
    print("\n  Paper 12 benchmark, full-corpus fit at the same epsilon:")
    print("    d_harm -0.0867  d_ben +0.0627  p* 0.5816")
    print("\n  't1 all<0' and 't2 all>0' are the two-tier structure. Experiment B")
    print("  found it does NOT replicate under independent fine-tuning in code")
    print("  search. If it also fails here, tier membership is neither model-")
    print("  stable nor corpus-stable, and the paper's empirical core becomes")
    print("  that membership is measurable only locally.")
    print("\n  full vs full512 isolates the truncation asymmetry: tier 1 is all")
    print("  BERT-scale at 512 while tier 2 holds every LLM-scale model, so the")
    print("  native-limit variant advantages tier 2 for reasons unrelated to")
    print("  whitening. If the two agree, that confound is immaterial.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true",
                    help="2 models, hpi only - checks wiring, writes nothing")
    ap.add_argument("--variant", choices=list(VARIANTS), default=None)
    ap.add_argument("--corpus", choices=["er_reason", "mimic_discharge"],
                    default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"device {DEV} | epsilon {EPSILON} | {len(PANEL)} configurations")

    corpora = [a.corpus] if a.corpus else ["er_reason", "mimic_discharge"]
    variants = [a.variant] if a.variant else list(VARIANTS)
    panel = PANEL[:2] if a.probe else PANEL
    if a.probe:
        corpora, variants = corpora[:1], ["hpi"]

    df = run(corpora, variants, panel, a.batch_size)
    report(df)
    if a.probe:
        print("\nprobe only - nothing written")
        return
    OUT.mkdir(exist_ok=True)
    f = OUT / "expA_panel_results.csv"
    df.to_csv(f, index=False)
    print(f"\n-> {f}  ({len(df)} rows = "
          f"{df.model.nunique()} models x {len(df)//df.model.nunique()} conditions)")


if __name__ == "__main__":
    main()
