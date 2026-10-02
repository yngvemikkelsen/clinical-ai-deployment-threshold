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

Paper 9 needs those numbers as a BOUNDARY TEST, not as a generality claim. The
framework applies only where an intervention's effect changes sign across
response-defined subgroups, and whether that condition holds in a given domain
is empirical. This run asks whether it holds in semantic code search.

WHAT THE RUN FOUND
------------------
It holds, but only at low regularisation. At eps = 0, three of four
configurations have a positive mean effect and CodeT5+ is negative, so
response-defined groups exist. CodeT5+ turns positive between eps = 1e-3 and
eps = 1e-2; above that the harmed group is empty and no threshold is defined.

The PUBLISHED grouping does not reproduce. Fine-tuned CodeBERT, which their
account places in the harmed group, is positive at every epsilon tested.

The clinical comparison is reported as a harmed group that is NON-EMPTY at every
one of the six clinical epsilon values (six to nine of 13 configurations). It is
NOT reported as "the clinical harmed group stayed negative": under grouping by
measured sign at each value, that statement is true by construction.

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
Step 1 reproduces their Table 2 baselines (no whitening) for the RELEASED
checkpoints only: CodeBERT, CodeT5+ and Code Llama on the six CodeSearchNet
languages, 18 cells. If those do not match to within 0.01 MRR, STOP.

Two sets of cells sit outside the gate by construction and are printed
separately rather than silently dropped:

  * fine-tuned CodeBERT. The published checkpoint was not released, so it is
    fine-tuned here with the authors' own fine_tune.py at their published
    settings (InfoNCE, lr 5e-5, batch 32, 5 epochs). Independent fine-tuning
    does not reproduce a published checkpoint exactly; this one exceeds their
    baseline in all six languages, by 0.013 to 0.177 MRR. Baseline agreement
    against a published value is not an applicable criterion for it.
  * R. Outside the published primary grid. CodeBERT and Code Llama on R both
    reproduce within 0.01 through this same pipeline and corpus, but CodeT5+
    on R gives 0.405 against a published 0.045. That cell is reported and
    carries no claim; excluding it does not change the epsilon at which
    CodeT5+ changes sign.

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

# Gate scope. The gate compares against Diera Table 2 and is meaningful only for
# checkpoints they released, on the languages in their primary grid.
GATED_MODELS = ("codebert", "codet5p", "codellama")
PRIMARY_LANGS = ("ruby", "javascript", "go", "java", "python", "php")

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
    """Gate the pipeline on the released-checkpoint cells only.

    Scope is the point. codebert_ft is fine-tuned locally because the published
    checkpoint was not released, and R is outside the published primary grid, so
    neither is held to a published-baseline criterion. Both are still printed.
    """
    b = df[df.epsilon == EPSILONS[0]][["model", "lang", "baseline_MRR"]].drop_duplicates()
    gated, ungated = [], []
    for _, r in b.iterrows():
        ref = TABLE2.get((r.model, r.lang))
        if ref is None:
            continue
        row = (r.model, r.lang, r.baseline_MRR, ref, abs(r.baseline_MRR - ref))
        in_scope = r.model in GATED_MODELS and r.lang in PRIMARY_LANGS
        (gated if in_scope else ungated).append(row)

    print("\nGATED - released checkpoints, six CodeSearchNet languages")
    print(f"{'model':<14}{'lang':<12}{'ours':>8}{'paper':>8}{'diff':>9}  gate")
    worst = 0.0
    for mk, lang, ours, ref, diff in gated:
        worst = max(worst, diff)
        print(f"{mk:<14}{lang:<12}{ours:>8.3f}{ref:>8.3f}{diff:>9.4f}"
              f"  {'ok' if diff < 0.01 else 'MISMATCH'}")
    print(f"\n  cells gated: {len(gated)}    largest deviation: {worst:.4f}")
    ok = worst < 0.01
    print("  PASS." if ok else
          "  STOP. Pipeline does not reproduce the released-checkpoint baselines;\n"
          "  nothing downstream is interpretable.")

    if ungated:
        print("\nNOT GATED - reported in full; no claim rests on their baseline agreement")
        print(f"{'model':<14}{'lang':<12}{'ours':>8}{'paper':>8}{'diff':>9}  reason")
        for mk, lang, ours, ref, diff in ungated:
            why = ("locally fine-tuned, checkpoint not released"
                   if mk == "codebert_ft" else "outside published primary grid")
            print(f"{mk:<14}{lang:<12}{ours:>8.3f}{ref:>8.3f}{diff:>9.4f}  {why}")
    return ok


def threshold(df):
    """Group by MEASURED SIGN at each epsilon, as the manuscript does.

    Grouping by training objective is the published claim under test, not the
    framework's rule, so it is reported separately and not used for p*.
    """
    print("\n" + "=" * 78)
    print("CONFIGURATION MEANS BY EPSILON")
    print("=" * 78)
    print(f"{'model':<14}{'contrastive':>12}" + "".join(f"{e:>10.0e}" for e in EPSILONS))
    for mk, g in df.groupby("model"):
        cells = "".join(f"{g[g.epsilon == e].delta_MRR.mean():>+10.4f}" for e in EPSILONS)
        print(f"{mk:<14}{str(CONTRASTIVE[mk]):>12}{cells}")

    print("\n" + "=" * 78)
    print("RESPONSE-DEFINED GROUPING AT EACH EPSILON (the framework's own rule)")
    print("=" * 78)
    print(f"{'epsilon':>10}{'n_ben':>7}{'n_harm':>8}{'d_ben':>12}{'d_harm':>12}{'p*':>12}")
    for e in EPSILONS:
        means = df[df.epsilon == e].groupby("model").delta_MRR.mean()
        ben, harm = means[means > 0], means[means < 0]
        if len(ben) == 0 or len(harm) == 0:
            print(f"{e:>10.0e}{len(ben):>7}{len(harm):>8}{'-':>12}{'-':>12}{'undefined':>12}")
            continue
        db, dh = ben.mean(), harm.mean()
        print(f"{e:>10.0e}{len(ben):>7}{len(harm):>8}{db:>+12.4f}{dh:>+12.4f}"
              f"{abs(dh) / (db + abs(dh)):>12.4f}")
    print("\n  A threshold exists only while the harmed group is non-empty.")
    print("  Clinical comparator under the same rule: non-empty at all six values")
    print("  tested, six to nine of 13 configurations, p* from 0.3163 to 0.6993.")

    print("\n" + "=" * 78)
    print("DOES THE PUBLISHED MECHANISTIC GROUPING REPRODUCE?")
    print("=" * 78)
    print("  Published claim: contrastively trained configurations are harmed at eps = 0.")
    z = df[df.epsilon == 0.0].groupby("model").delta_MRR.mean()
    for mk in sorted(z.index):
        exp = "harmed" if CONTRASTIVE[mk] else "benefited"
        got = "benefited" if z[mk] > 0 else "harmed"
        print(f"  {mk:<14} published {exp:<10} measured {got:<10} {z[mk]:+.4f}"
              f"   {'agrees' if exp == got else 'DOES NOT AGREE'}")


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
