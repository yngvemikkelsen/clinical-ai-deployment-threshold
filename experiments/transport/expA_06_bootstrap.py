#!/usr/bin/env python3
"""
Paper 9 - Experiment A, step 6: empirical screen accuracy by local resampling.

WHY THIS EXISTS
---------------
The screening curve in the manuscript derives se(n) and sp(n) from BETWEEN-
CONDITION variability - the spread of dMRR across corpus x query-format cells.
That is a proxy, and it was never justified as one. A deploying site does not
sample conditions. It samples documents from its own corpus, fits the transform
on them, scores retrieval, and reads the sign. The variability that governs its
error rate is within-corpus sampling variability, which is a different quantity.

This script measures the thing directly. For each configuration on each external
corpus:

    1. the full-sample dMRR@10 defines the reference sign (the "truth" a
       perfectly-powered site would observe)
    2. draw B subsamples of n documents without replacement
    3. refit corpus-only ZCA on the subsample exactly as a site would - W and
       mu_D from those n documents only, never the full corpus
    4. score the n paired queries against those n documents
    5. record whether sign(dMRR_subsample) == sign(dMRR_full)

    se(n) = P(correct sign | truly benefits)   over tier 2
    sp(n) = P(correct sign | truly harmed)     over tier 1

with Wilson intervals on both. Feeding those into the inverted screening
condition gives an EMPIRICALLY OBSERVED requirement curve, against which the
derived one can be compared.

This upgrades the evidence chain from

    derived screen accuracy -> external effect-size transport

to

    derived screen accuracy -> independently observed screen accuracy
                            -> external decision threshold

Three outcomes, all informative. Agreement validates the derivation. Better than
derived means the published rule is demonstrably conservative. Worse means local
measurement characteristics themselves need calibration - which is the paper's
own proposition, so it is a finding rather than a failure.

NOTE ON THE RESAMPLING UNIT
---------------------------
Documents, not conditions. n here is the number of documents a site scores, not
the number of corpus-format cells. The manuscript's n_cond and this n are
different axes and must not be conflated in the writeup; the mapping between
them is empirical and is reported by this script, not assumed.

REQUIRES cached embeddings. The panel script discards them after computing
dMRR, so run expA_04_panel.py with --cache-embeddings first (see --help), or
point --emb at a directory of {corpus}_{model}_{docs,q_nl,q_keyword}.npy.

Usage:
    python expA_06_bootstrap.py --emb ./expA_emb --out expA_bootstrap.csv
    python expA_06_bootstrap.py --emb ./expA_emb --n-grid 10,20,30,50,75 -B 500
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EPSILON = 1e-5
SEED = 42
TIER1 = {"BioLORD-2023", "MedCPT", "BGE-base", "GTE-base",
         "Nomic-embed-text", "Nomic-embed-text-nopfx"}


# ------------------------------------------------------------------ ZCA, MRR
# Corpus-only ZCA is W = U(Lam + eps I)^(-1/2) U^T with W and mu_D fit on the
# documents. Computing that from the d x d covariance is O(d^3) and, for a
# subsample of n <= 75 documents, almost entirely null space: the covariance has
# rank <= n-1. At d=4096 the naive route costs 60 s per draw, which makes the
# bootstrap infeasible.
#
# The transform decomposes exactly. With U_r spanning the data,
#     W = U_r (Lam_r + eps I)^(-1/2) U_r^T + eps^(-1/2) (I - U_r U_r^T)
# so W x = eps^(-1/2) x + U_r [(Lam_r + eps)^(-1/2) - eps^(-1/2)] U_r^T x,
# computable from the thin SVD of the centred data in O(n^2 d). Verified
# identical to the full construction to 2.6e-14 with matching MRR; 25 ms per
# draw at d=4096, a 2400-fold reduction.


def fit_zca(D, eps=EPSILON):
    """Thin-SVD ZCA. Returns (Vt, scale, mu) - the in-span correction, not W."""
    n = len(D)
    mu = D.mean(0)
    Dc = D - mu
    _, S, Vt = np.linalg.svd(Dc, full_matrices=False)
    lam = S ** 2 / max(n - 1, 1)
    scale = 1.0 / np.sqrt(lam + eps) - 1.0 / np.sqrt(eps)
    return Vt, scale, mu


def apply_zca(X, P, eps=EPSILON):
    Vt, scale, mu = P
    Xc = X - mu
    Y = Xc / np.sqrt(eps) + (Xc @ Vt.T * scale) @ Vt
    return Y / np.linalg.norm(Y, axis=1, keepdims=True).clip(1e-9)


def mrr_at_k(Q, D, k=10):
    S = Q @ D.T
    correct = np.diag(S)[:, None]
    rank = (S > correct).sum(1) + 1
    return float(np.where(rank <= k, 1.0 / rank, 0.0).mean())


def delta(Q, D):
    """dMRR@10 for corpus-only ZCA, fit and applied on exactly these rows."""
    P = fit_zca(D)
    return mrr_at_k(apply_zca(Q, P), apply_zca(D, P)) - mrr_at_k(Q, D)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# ---------------------------------------------------------------- bootstrap
def run(emb_dir: Path, n_grid, B, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    corpora = sorted({p.name.split("__")[0] for p in emb_dir.glob("*__*__docs.npy")})
    if not corpora:
        raise SystemExit(f"no embeddings in {emb_dir} "
                         "(expect {corpus}__{model}__docs.npy)")
    for corpus in corpora:
        models = sorted(p.name.split("__")[1]
                        for p in emb_dir.glob(f"{corpus}__*__docs.npy"))
        print(f"\n{corpus}: {len(models)} configurations")
        for model in models:
            D = np.load(emb_dir / f"{corpus}__{model}__docs.npy")
            for fmt in ("nl", "keyword"):
                f = emb_dir / f"{corpus}__{model}__q_{fmt}.npy"
                if not f.exists():
                    continue
                Q = np.load(f)
                N = len(D)
                d_full = delta(Q, D)
                if d_full == 0:
                    print(f"  {model}/{fmt}: full-sample delta is zero, skipped")
                    continue
                ref = np.sign(d_full)
                for n in n_grid:
                    if n > N:
                        continue
                    hits = 0
                    for _ in range(B):
                        idx = rng.choice(N, size=n, replace=False)
                        # a site sees only its own n documents: the transform is
                        # fit on them and retrieval is scored against them alone
                        if np.sign(delta(Q[idx], D[idx])) == ref:
                            hits += 1
                    lo, hi = wilson(hits, B)
                    rows.append(dict(corpus=corpus, model=model,
                                     tier=1 if model in TIER1 else 2,
                                     query_format=fmt, n_docs=n, B=B,
                                     delta_full=d_full, correct=hits,
                                     p_correct=hits / B, ci_lo=lo, ci_hi=hi))
            print(f"  {model:<24} full-sample delta "
                  f"{d_full:+.4f}  ({'tier 1' if model in TIER1 else 'tier 2'})")
    return pd.DataFrame(rows)


def report(df, d_ben_ext, d_harm_ext, derived=None):
    print("\n" + "=" * 92)
    print("EMPIRICAL SCREEN ACCURACY BY LOCAL RESAMPLING")
    print("=" * 92)
    print(f"  {'n_docs':>7}{'se':>8}{'se 95% CI':>18}{'sp':>8}{'sp 95% CI':>18}"
          f"{'min prevalence':>17}")
    out = []
    for n, g in df.groupby("n_docs"):
        t2, t1 = g[g.tier == 2], g[g.tier == 1]
        se, sp = t2.p_correct.mean(), t1.p_correct.mean()
        se_ci = wilson(int(t2.correct.sum()), int(t2.B.sum()))
        sp_ci = wilson(int(t1.correct.sum()), int(t1.B.sum()))
        b = (1 - sp) * abs(d_harm_ext)
        ps = b / (se * d_ben_ext + b)
        out.append(dict(n_docs=n, se=se, se_lo=se_ci[0], se_hi=se_ci[1],
                        sp=sp, sp_lo=sp_ci[0], sp_hi=sp_ci[1],
                        p_screen_star=ps))
        print(f"  {n:>7}{se:>8.3f}  [{se_ci[0]:.3f}, {se_ci[1]:.3f}]"
              f"{sp:>8.3f}  [{sp_ci[0]:.3f}, {sp_ci[1]:.3f}]{ps:>17.4f}")

    print("\n  se = P(correct sign | truly benefits), over tier 2")
    print("  sp = P(correct sign | truly harmed), over tier 1")
    print("  Wilson intervals pool draws across configurations within tier.")

    if derived:
        print("\n  against the DERIVED curve (between-condition variability):")
        print(f"  {'n_cond':>7}{'derived requirement':>22}   "
              f"{'closest n_docs':>15}{'observed requirement':>22}")
        for nc, req in derived.items():
            near = min(out, key=lambda r: abs(r["p_screen_star"] - req))
            print(f"  {nc:>7}{req:>22.4f}   {near['n_docs']:>15}"
                  f"{near['p_screen_star']:>22.4f}")
        print("\n  The two axes are DIFFERENT: n_cond counts corpus-format cells,")
        print("  n_docs counts documents a site scores. This table maps between")
        print("  them empirically. Do not report them as the same quantity.")

    print("\n  Reading the outcome:")
    print("    observed requirement <= derived  -> the published rule is")
    print("      conservative for these settings, which is the safe direction")
    print("    observed requirement  > derived  -> between-condition variability")
    print("      understated local sampling error, and the derived curve is")
    print("      optimistic. That is a finding, not a failure: it is the paper's")
    print("      own proposition that measurement characteristics need local")
    print("      calibration.")
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("expA_bootstrap.csv"))
    ap.add_argument("--n-grid", default="10,20,30,50,75")
    ap.add_argument("-B", type=int, default=500)
    ap.add_argument("--d-ben", type=float, default=0.1419,
                    help="external tier 2 mean, from the panel run")
    ap.add_argument("--d-harm", type=float, default=-0.1010,
                    help="external tier 1 mean, from the panel run")
    a = ap.parse_args()

    n_grid = [int(x) for x in a.n_grid.split(",")]
    df = run(a.emb, n_grid, a.B)
    df.to_csv(a.out, index=False)
    print(f"\n-> {a.out}  ({len(df)} rows)")

    derived = {1: 0.2165, 2: 0.1020, 3: 0.0495,
               4: 0.0246, 5: 0.0125, 6: 0.0065}
    summ = report(df, a.d_ben, a.d_harm, derived)
    summ.to_csv(str(a.out).replace(".csv", "_summary.csv"), index=False)
    print(f"-> {str(a.out).replace('.csv', '_summary.csv')}")


if __name__ == "__main__":
    main()
