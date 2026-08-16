#!/usr/bin/env python3
"""
Paper 9 - Experiment B: second empirical instance of a sign-changing correction.

Replicates Diera, Galke & Scherp (ESANN 2025, doi 10.14428/esann/2025.ES2025-58;
arXiv 2411.17538) at epsilon = 0 to obtain the SIGN SPLIT they describe in prose
but never tabulate.

WHY THIS RUN EXISTS
-------------------
Their Table 3 reports dMRR at the BEST epsilon per model, where every value is
positive. The sign flip appears only in their text: standard ZCA (eps = 0)
"greatly improves the base CodeBERT and Code Llama results, but in the case of
fine-tuned CodeBERT and CodeT5+, it decreased the ranking performance on most
datasets." No numbers are given for that case.

Paper 9 needs those numbers. They yield d_ben and d_harm in a domain unrelated
to clinical retrieval, from an independent group's code and data, and therefore
a SECOND deployment threshold computed identically:

    p* = |d_harm| / (d_ben + |d_harm|)

That converts "general theorem + one case" into demonstrated generality.

SECONDARY OUTPUT, ARGUABLY MORE IMPORTANT
-----------------------------------------
Diera conclude that per-model epsilon tuning makes the correction beneficial for
everyone. Paper 12's epsilon sweep shows Tier 1 harmed at ALL six epsilon values
tested. Running their full epsilon grid here establishes the contrast precisely:
whether tuning rescues the harmed subgroup is DOMAIN-DEPENDENT. That answers the
"why is there no per-site epsilon-tuning arm?" question a reviewer will raise.

CHANGES FROM THE ORIGINAL REPO
------------------------------
1. Batched embedding extraction (their loop is one sequence at a time).
2. Vectorised MRR (their loop is O(n^2) in Python; the Python split is 22,176
   pairs = 4.9e8 scalar distance calls).
3. Their exact rank convention is preserved: rank = #{distances <= correct},
   ties counted against. Do NOT switch to Paper 12's convention - comparability
   with their published Table 2 is the validation gate.
4. Sweeps a full epsilon grid including 0 in one pass, writes tidy parquet.

VALIDATION GATE
---------------
Step 1 reproduces their Table 2 baselines (no whitening). If those do not match
to within ~0.01 MRR, STOP: the pipeline differs from theirs and nothing
downstream is interpretable.

    Table 2 baseline MRR (from the paper)
                 CodeBERT  FT-CodeBERT  CodeT5+  CodeLlama
    ruby            0.006        0.547    0.705      0.047
    javascript      0.002        0.427    0.638      0.026
    go              0.002        0.619    0.757      0.031
    java            0.000        0.395    0.595      0.015
    python          0.001        0.500    0.721      0.017
    php             0.000        0.248    0.537      0.009
    r               0.011          n/a    0.045      0.024

USAGE
-----
    # 0. their repo + deps
    git clone https://github.com/drndr/code_isotropy.git && cd code_isotropy
    pip install numpy torch pandas transformers IsoScore datasets info_nce pyarrow

    # 1. fine-tune CodeBERT per language (REQUIRED - it is one of the two
    #    harmed models). This is the expensive step.
    for L in ruby javascript go java python php; do
        python fine_tune.py --lang $L --train_batch_size 32 \
            --learning_rate 5e-5 --num_train_epochs 5 --num_of_accumulation_steps 1
    done

    # 2. embeddings (this script, batched)
    python paper9_expB_diera.py embed --all

    # 3. evaluate across the epsilon grid and derive the threshold
    python paper9_expB_diera.py evaluate
    python paper9_expB_diera.py threshold

NOTES ON COST
-------------
Test-set sizes: ruby 1,261 | javascript 6,483 | go 14,291 | java 26,909 |
python 22,176 | php 28,391 | r 1,070. Roughly 100k sequences, doubled for
code+doc. CodeBERT and CodeT5+ are ~110-125M and cheap. Code Llama 7B dominates.
A ruby + r + javascript pilot (~8.8k pairs) validates the whole pipeline in
minutes and is worth running before committing to the full grid.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

LANGS = ["ruby", "javascript", "go", "java", "python", "php", "r"]
PILOT_LANGS = ["ruby", "r", "javascript"]
MODELS = {
    "codebert": ("microsoft/codebert-base", False),
    "codebert_ft": ("microsoft/codebert-base", True),
    "codet5p": ("Salesforce/codet5p-110m-embedding", False),
    "codellama": ("codellama/CodeLlama-7b-hf", False),
}
# Diera Table 1: contrastive pre-training yes/no. This is the tier axis.
CONTRASTIVE = {"codebert": False, "codebert_ft": True,
               "codet5p": True, "codellama": False}

EPSILONS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

EMB = Path("./embeddings")
OUT = Path("./paper9_expB")
OUT.mkdir(exist_ok=True)

# Diera Table 2, for the validation gate
TABLE2 = {
    ("codebert", "ruby"): 0.006, ("codebert", "javascript"): 0.002,
    ("codebert", "go"): 0.002, ("codebert", "java"): 0.000,
    ("codebert", "python"): 0.001, ("codebert", "php"): 0.000,
    ("codebert", "r"): 0.011,
    ("codebert_ft", "ruby"): 0.547, ("codebert_ft", "javascript"): 0.427,
    ("codebert_ft", "go"): 0.619, ("codebert_ft", "java"): 0.395,
    ("codebert_ft", "python"): 0.500, ("codebert_ft", "php"): 0.248,
    ("codet5p", "ruby"): 0.705, ("codet5p", "javascript"): 0.638,
    ("codet5p", "go"): 0.757, ("codet5p", "java"): 0.595,
    ("codet5p", "python"): 0.721, ("codet5p", "php"): 0.537,
    ("codet5p", "r"): 0.045,
    ("codellama", "ruby"): 0.047, ("codellama", "javascript"): 0.026,
    ("codellama", "go"): 0.031, ("codellama", "java"): 0.015,
    ("codellama", "python"): 0.017, ("codellama", "php"): 0.009,
    ("codellama", "r"): 0.024,
}


def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load_statcodesearch(path="./statcodesearch/test_statcodesearch.jsonl"):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            if "input" not in d:
                continue
            parts = d["input"].split("[CODESPLIT]", 1)
            if len(parts) == 2:
                rows.append({"func_documentation_tokens": parts[0].strip(),
                             "func_code_tokens": parts[1].strip()})
    return pd.DataFrame(rows)


def load_lang(lang):
    if lang == "r":
        return load_statcodesearch()
    from datasets import load_dataset
    try:
        return load_dataset("code_search_net", lang, trust_remote_code=True)["test"]
    except TypeError:
        return load_dataset("code_search_net", lang)["test"]


# ---------------------------------------------------------------- embeddings
def _texts(ds, kind):
    col = "func_documentation_tokens" if kind == "doc" else "func_code_tokens"
    return [(" ".join(x).strip() if not isinstance(x, str) else x.strip())
            for x in ds[col]]


def embed(model_key, lang, batch_size=32, device=None):
    """Batched re-implementation. Pooling matches the original exactly."""
    from transformers import AutoModel, AutoTokenizer
    ckp, is_ft = MODELS[model_key]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(ckp, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(ckp, trust_remote_code=True)
    if is_ft:
        sd = torch.load(f"./models/codebert_{lang}.pth", map_location="cpu")
        model.load_state_dict(sd)
    model.to(device).eval()

    ds = load_lang(lang)
    for kind in ("code", "doc"):
        texts = _texts(ds, kind)
        out = []
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], padding=True, truncation=True,
                      max_length=256, return_tensors="pt").to(device)
            with torch.no_grad():
                o = model(**enc)
            if ckp == "Salesforce/codet5p-110m-embedding":
                pooled = o if isinstance(o, torch.Tensor) else o[0]
                pooled = pooled.squeeze()
            else:
                # mean pool over real tokens only (original used no padding, so
                # an unmasked mean was equivalent; with batching it is not)
                h = o.last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            out.append(pooled.detach().cpu().float().numpy())
        arr = np.concatenate(out, 0)
        EMB.mkdir(exist_ok=True)
        suf = "_finetuned" if is_ft else ""
        base = "codebert" if model_key == "codebert_ft" else model_key
        np.save(EMB / f"{kind}_embs_{base}_{lang}{suf}.npy", arr)
        print(f"  {model_key}/{lang}/{kind}: {arr.shape}")
    del model
    torch.cuda.empty_cache()


def load_embs(model_key, lang):
    _, is_ft = MODELS[model_key]
    suf = "_finetuned" if is_ft else ""
    base = "codebert" if model_key == "codebert_ft" else model_key
    c = EMB / f"code_embs_{base}_{lang}{suf}.npy"
    d = EMB / f"doc_embs_{base}_{lang}{suf}.npy"
    if not (c.exists() and d.exists()):
        return None, None
    return np.load(c), np.load(d)


# ---------------------------------------------------------------- ZCA + MRR
def zca(X, eps):
    """Identical to Diera's zca_features(): fit and apply on the same matrix."""
    Xc = X - X.mean(0)
    Sigma = np.cov(Xc, rowvar=False)
    U, Lam, _ = np.linalg.svd(Sigma)
    W = U @ np.diag(1.0 / np.sqrt(Lam + eps)) @ U.T
    return Xc @ W.T


def mrr(doc, code):
    """Vectorised. Preserves Diera's rank convention: cosine DISTANCE, and
    rank = #{distances <= correct distance}, i.e. ties counted against."""
    d = doc / np.linalg.norm(doc, axis=1, keepdims=True).clip(1e-9)
    c = code / np.linalg.norm(code, axis=1, keepdims=True).clip(1e-9)
    dist = 1.0 - (d @ c.T)                       # [n_queries x n_codes]
    correct = np.diag(dist)[:, None]
    ranks = (dist <= correct).sum(1)             # >= 1 by construction
    return float(np.mean(1.0 / ranks))


def evaluate(langs):
    rows = []
    for mk in MODELS:
        for lang in langs:
            if mk == "codebert_ft" and lang == "r":
                continue                          # no fine-tuning data for R
            code, doc = load_embs(mk, lang)
            if code is None:
                print(f"  skip {mk}/{lang}: embeddings missing")
                continue
            base = mrr(doc, code)
            for eps in EPSILONS:
                m = mrr(zca(doc, eps), zca(code, eps))
                rows.append(dict(model=mk, lang=lang, epsilon=eps,
                                 contrastive=CONTRASTIVE[mk],
                                 baseline_MRR=base, zca_MRR=m,
                                 delta_MRR=m - base))
            print(f"  {mk}/{lang}: baseline {base:.3f}")
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "diera_replication.parquet", index=False)
    return df


def validation_gate(df):
    b = df[df.epsilon == EPSILONS[0]][["model", "lang", "baseline_MRR"]].drop_duplicates()
    print(f"\n{'model':<14}{'lang':<12}{'ours':>8}{'paper':>8}{'diff':>8}  gate")
    worst = 0.0
    for _, r in b.iterrows():
        ref = TABLE2.get((r.model, r.lang))
        if ref is None:
            continue
        diff = abs(r.baseline_MRR - ref); worst = max(worst, diff)
        print(f"{r.model:<14}{r.lang:<12}{r.baseline_MRR:>8.3f}{ref:>8.3f}"
              f"{diff:>8.3f}  {'ok' if diff < 0.01 else 'MISMATCH'}")
    print(f"\nlargest deviation from Diera Table 2: {worst:.4f}")
    if worst >= 0.01:
        print("STOP. Pipeline does not reproduce the published baselines;")
        print("nothing downstream is interpretable.")
    return worst < 0.01


def threshold(df):
    print("\n" + "=" * 78)
    print("SIGN SPLIT AT eps = 0 (the case Diera describe but do not tabulate)")
    print("=" * 78)
    z = df[df.epsilon == 0.0]
    print(f"{'model':<14}{'contrastive':>12}{'mean dMRR':>11}{'SD across langs':>17}{'n':>4}")
    for mk, g in z.groupby("model"):
        print(f"{mk:<14}{str(CONTRASTIVE[mk]):>12}{g.delta_MRR.mean():>+11.4f}"
              f"{g.delta_MRR.std(ddof=1):>17.4f}{len(g):>4}")
    ben = z[~z.contrastive].delta_MRR.mean()
    harm = z[z.contrastive].delta_MRR.mean()
    print(f"\n  d_ben  (non-contrastive) {ben:+.4f}")
    print(f"  d_harm (contrastive)     {harm:+.4f}")
    if ben > 0 > harm:
        p = abs(harm) / (ben + abs(harm))
        print(f"  SECOND THRESHOLD p* = |d_harm|/(d_ben+|d_harm|) = {p:.4f}")
        print("  (clinical retrieval, same construction: 0.5816)")
    else:
        print("  NO SIGN SPLIT at eps=0 in this replication. Report as such:")
        print("  it would mean the two-tier structure is not reproduced")
        print("  independently, and the generality claim must be withdrawn.")

    print("\n" + "=" * 78)
    print("DOES EPSILON TUNING RESCUE THE HARMED SUBGROUP?")
    print("=" * 78)
    print("  Diera: yes (their Table 3, best-eps, all positive).")
    print("  Paper 12: no - Tier 1 negative at all six eps tested.")
    print(f"\n{'model':<14}" + "".join(f"{e:>10.0e}" for e in EPSILONS))
    for mk, g in df.groupby("model"):
        cells = "".join(f"{g[g.epsilon == e].delta_MRR.mean():>+10.3f}"
                        for e in EPSILONS)
        print(f"{mk:<14}{cells}")
    harmed = df[df.contrastive]
    per_eps = harmed.groupby("epsilon").delta_MRR.mean()
    rescued = [e for e in EPSILONS if per_eps.get(e, -1) > 0]
    print(f"\n  epsilon values at which the contrastive group is NOT harmed: "
          f"{rescued if rescued else 'none'}")
    print("  If nonempty, tuning rescues in code search but not in clinical")
    print("  retrieval, and that domain contrast is the reportable result.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["embed", "evaluate", "threshold"])
    ap.add_argument("--all", action="store_true", help="all 7 languages")
    ap.add_argument("--pilot", action="store_true", help="ruby, r, javascript only")
    ap.add_argument("--model", default=None)
    ap.add_argument("--lang", default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    a = ap.parse_args()
    set_seed()
    langs = PILOT_LANGS if a.pilot else (LANGS if a.all else [a.lang or "ruby"])

    if a.cmd == "embed":
        for mk in ([a.model] if a.model else list(MODELS)):
            for lang in langs:
                if mk == "codebert_ft" and lang == "r":
                    continue
                print(f"embedding {mk}/{lang}")
                embed(mk, lang, a.batch_size)
    else:
        f = OUT / "diera_replication.parquet"
        df = pd.read_parquet(f) if (f.exists() and a.cmd == "threshold") \
            else evaluate(langs)
        if a.cmd == "evaluate":
            validation_gate(df)
        else:
            threshold(df)


if __name__ == "__main__":
    main()
