#!/usr/bin/env python3
"""
Paper 9 - Experiment A, step 1: sample the two external corpora.

Extends the screening curve from 6 observed conditions to 10 by adding two
INSTITUTIONAL corpora that differ from Paper 12's benchmark set in institution,
genre, and de-identification regime:

    MIMIC-IV-Note v2.2  discharge summaries      (BIDMC)
    ER-Reason v1.0.0    ED provider notes        (UCSF)

against Paper 12's MTSamples / PMC-Patients / synthetic.

DESIGN
------
100 documents per corpus, matching Paper 12's working scale. Random sample at a
fixed seed, minimum 400 characters (the eligibility filter used in Paper 19),
no stratification. Stratifying would be defensible on its own terms but would
break comparability: the six original conditions are unstratified, so a change
in the screening curve could not then be attributed to the corpora rather than
to the sampling.

Also emits the metadata fields Paper 12's query prompt expects, taken from
registry columns where they exist. These support the STRUCTURED-METADATA
sensitivity; the primary run uses LLM extraction from note text, faithful to
the published protocol (Multimedia Appendix 1).

DUA
---
Both corpora are PhysioNet credentialed-access. Nothing here transmits note
text anywhere. Query generation (step 2) runs locally on Ollama for the same
reason - the published protocol used GPT-4o, which is not permissible here.
That model substitution is the one deliberate deviation and is bridged in
step 3 against the existing GPT-4o queries on a public corpus.

Usage:
    python expA_01_sample.py --check      # counts only, no sampling
    python expA_01_sample.py
"""
from __future__ import annotations

import argparse
import os
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_DOCS = 100
MIN_CHARS = 400

# PhysioNet root. Both corpora are credentialed-access and must be
# obtained directly; set PHYSIONET_ROOT to wherever they are held.
PN = Path(os.environ.get("PHYSIONET_ROOT",
                         Path.home() / "physionet.org" / "files"))
ER_CSV = PN / "er-reason/1.0.0/er_reason.csv"
MIMIC_DIS = PN / "mimic-iv-note/2.2/note/discharge.csv.gz"
OUT = Path("./expA_corpora")


# ---------------------------------------------------------------- ER-Reason
ER_COLS = ["encounterkey", "ED_Provider_Notes_Text", "primarychiefcomplaintname",
           "primaryeddiagnosisname", "sex", "Age", "acuitylevel", "eddisposition"]


def er_reason(check_only=False):
    """One row per ED encounter; each note type in its own column.

    ED_Provider_Notes_Text is used rather than the discharge summary: it keeps
    the genre distinct from the MIMIC sample, so the two external corpora test
    two different documentation types rather than two samples of one.
    """
    df = pd.read_csv(ER_CSV, usecols=ER_COLS, low_memory=False)
    s = df["ED_Provider_Notes_Text"]
    df = df[s.notna()].copy()
    df["text"] = df["ED_Provider_Notes_Text"].astype(str)
    df["n_chars"] = df["text"].str.len()
    ok = df[df["n_chars"] >= MIN_CHARS]
    print(f"ER-Reason: {len(s):,} encounters | {s.notna().sum():,} with ED note "
          f"| {len(ok):,} at >={MIN_CHARS} chars")
    print(f"  chars: median {int(ok.n_chars.median()):,} "
          f"IQR {int(ok.n_chars.quantile(.25)):,}-{int(ok.n_chars.quantile(.75)):,}")
    if check_only:
        return None
    smp = ok.sample(n=N_DOCS, random_state=SEED).reset_index(drop=True)
    return pd.DataFrame({
        "doc_id": smp["encounterkey"].astype(str),
        "text": smp["text"],
        "n_chars": smp["n_chars"],
        # registry metadata for the structured-metadata sensitivity
        "meta_specialty": "Emergency Medicine",
        "meta_note_type": "ED Provider Note",
        "meta_primary_diagnosis": smp["primaryeddiagnosisname"].fillna(
            smp["primarychiefcomplaintname"]).fillna("Unknown"),
        "meta_secondary": "none",
        "meta_demographics": (smp["Age"].astype(str) + "y " +
                              smp["sex"].fillna("unknown").astype(str)),
    })


# ------------------------------------------------------------ MIMIC-IV-Note
def mimic_note(check_only=False):
    """Discharge summaries. ~331k rows; read in chunks to stay in memory."""
    keep, n_tot, n_ok = [], 0, 0
    rng = np.random.default_rng(SEED)
    with gzip.open(MIMIC_DIS, "rt") as f:
        for chunk in pd.read_csv(f, usecols=["note_id", "subject_id", "hadm_id",
                                             "note_type", "text"],
                                 chunksize=20_000, low_memory=False):
            n_tot += len(chunk)
            c = chunk[chunk["text"].notna()].copy()
            c["n_chars"] = c["text"].astype(str).str.len()
            c = c[c["n_chars"] >= MIN_CHARS]
            n_ok += len(c)
            # reservoir-style: keep a random 2% so the pool stays small but
            # the eventual sample is still drawn from the whole file
            if len(c):
                keep.append(c.sample(frac=0.02, random_state=SEED))
    pool = pd.concat(keep, ignore_index=True)
    print(f"MIMIC-IV-Note: {n_tot:,} discharge summaries | {n_ok:,} at "
          f">={MIN_CHARS} chars | pool {len(pool):,}")
    print(f"  chars: median {int(pool.n_chars.median()):,} "
          f"IQR {int(pool.n_chars.quantile(.25)):,}-{int(pool.n_chars.quantile(.75)):,}")
    if check_only:
        return None
    smp = pool.sample(n=N_DOCS, random_state=SEED).reset_index(drop=True)
    return pd.DataFrame({
        "doc_id": smp["note_id"].astype(str),
        "hadm_id": smp["hadm_id"],
        "text": smp["text"].astype(str),
        "n_chars": smp["n_chars"],
        # MIMIC discharge.csv carries no diagnosis fields. The structured
        # sensitivity would need a join to mimiciv/hosp/diagnoses_icd on
        # hadm_id; hadm_id is retained here so that join stays possible.
        "meta_specialty": "Unknown",
        "meta_note_type": "Discharge Summary",
        "meta_primary_diagnosis": "Unknown",
        "meta_secondary": "none",
        "meta_demographics": "adult",
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report counts and length distributions, sample nothing")
    a = ap.parse_args()

    for p in (ER_CSV, MIMIC_DIS):
        if not p.exists():
            raise SystemExit(f"not found: {p}")

    OUT.mkdir(exist_ok=True)
    er = er_reason(a.check)
    mi = mimic_note(a.check)
    if a.check:
        return

    for name, df in (("er_reason", er), ("mimic_discharge", mi)):
        f = OUT / f"{name}_n{N_DOCS}_seed{SEED}.csv"
        df.to_csv(f, index=False)
        print(f"\n{name}: {len(df)} docs -> {f}")
        print(f"  chars median {int(df.n_chars.median()):,} "
              f"min {int(df.n_chars.min()):,} max {int(df.n_chars.max()):,}")

    print("\nNOTE these are far longer than Paper 12's corpora (median ~400 "
          "tokens).\nAt a 512-token ceiling only the leading portion is encoded. "
          "Paper 12's\nchunking analysis found First-N equivalent to Full and "
          "Last-N worse by\n3-8 points, so head truncation is defensible - but "
          "the length difference\nis a real property of institutional notes and "
          "belongs in the writeup.")
    print("\nDO NOT commit expA_corpora/ - it contains credentialed note text.")


if __name__ == "__main__":
    main()
