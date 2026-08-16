#!/usr/bin/env python3
"""
Paper 9 - Experiment A, step 1b: emit BOTH document variants.

WHY TWO VARIANTS
----------------
Paper 12's corpora have median ~400 tokens. The external corpora have median
6,622 chars (ER-Reason ED notes, ~1,650 tokens) and 9,941 chars (MIMIC
discharge summaries, ~2,500 tokens). At the 512-token encoder ceiling that is
the leading 31% and 20% of each document respectively.

If the screening curve fails to transport, a single full-note run cannot
separate two explanations: the corpora are institutional, or four-fifths of
each document was never encoded. Running both variants removes the confound.

    full   - whole note, head-truncated by the encoder at 512 tokens.
             Faithful to Paper 12's encoding. Its chunking analysis found
             First-N equivalent to Full and Last-N worse by 3-8 points, so
             head truncation is defensible; the length difference is not
             thereby removed.
    hpi    - the History of Present Illness / ED course section only, which
             is length-comparable to Paper 12's corpora and is the section a
             clinical retrieval system would actually target.

20 conditions rather than 10. Panel compute is trivial either way.

Section extraction is heuristic and its yield is reported per corpus. Documents
where no section is found fall back to the leading 2,500 characters, flagged in
the `hpi_source` column so the fallback share is auditable rather than silent.

Usage:
    python expA_01b_variants.py --check
    python expA_01b_variants.py
"""
from __future__ import annotations

import argparse
import os
import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd

SEED, N_DOCS, MIN_CHARS = 42, 100, 400
FALLBACK_CHARS = 2500
# a section floor, not the note floor: MIN_CHARS=400 is right for a whole
# note and far too high for an HPI paragraph (smoke test showed valid
# 120-160 char sections being discarded to fallback)
MIN_SECTION_CHARS = 80

# PhysioNet root. Both corpora are credentialed-access and must be
# obtained directly; set PHYSIONET_ROOT to wherever they are held.
PN = Path(os.environ.get("PHYSIONET_ROOT",
                         Path.home() / "physionet.org" / "files"))
ER_CSV = PN / "er-reason/1.0.0/er_reason.csv"
MIMIC_DIS = PN / "mimic-iv-note/2.2/note/discharge.csv.gz"
OUT = Path("./expA_corpora")

# Two corpora, two document structures - established by probing the real text,
# not assumed:
#   MIMIC discharge summaries are newline-delimited with colon-terminated
#     headers. "Brief Hospital Course:" / "History of Present Illness:" fire on
#     299/300 documents.
#   ER-Reason ED notes contain NO newlines at all. The whole note is one line
#     with runs of spaces as separators, and headers carry no colon. A ^-
#     anchored, colon-requiring pattern matches 0/300. Header frequencies over
#     200 notes: "ED Course" 200, "Medical Decision" 200, "Chief Complaint" 198,
#     "Physical Exam" 198, but "History of Present Illness" only 4.
#
# So the narrative section is corpus-specific. Headers are matched on a
# whitespace boundary rather than line start, and the colon is optional.
HPI_HEADS = [
    r"history of present illness", r"history of the present illness", r"hpi",
    r"brief hospital course",                      # MIMIC
    r"ed course", r"emergency department course",  # ER-Reason, 200/200
    r"medical decision making", r"medical decision",
    r"history",                                    # ER-Reason narrative opener
    r"chief complaint", r"presenting complaint", r"reason for visit",
]
NEXT_HEADS = [
    r"past medical history", r"pmh", r"medical history", r"surgical history",
    r"past surgical history", r"review of systems", r"ros",
    r"physical exam", r"physical examination", r"medications", r"allergies",
    r"allergies/contraindications", r"family history", r"social history",
    r"socioeconomic history", r"assessment", r"plan", r"impression",
    r"labs", r"laboratory", r"imaging", r"vital signs", r"interpreter used",
    r"discharge medications", r"discharge diagnosis", r"pertinent results",
    r"major surgical", r"disposition", r"procedures",
]
# Verified against real text from both corpora. The colon is optional (ER-Reason
# headers have none) and the trailing separator must accept a newline (MIMIC has
# "Brief Hospital Course:\n"), which an explicit [ \t] class cannot.
# A header must be FOLLOWED BY a colon, two or more spaces, or a newline.
# Requiring only "optional colon + whitespace" matches prose: in an ER-Reason
# note, "with history of COPD" fired the "history" pattern and captured the rest
# of the sentence. Single-space continuations are therefore excluded.
_B = r"(?:^|\s)"
_A = r"(?:\s*:\s*|[ \t]{2,}|\s*\n)"
_HPI_PATS = [re.compile(_B + r"(" + h + r")" + _A, re.I | re.M) for h in HPI_HEADS]
_STOP = re.compile(_B + r"(" + "|".join(NEXT_HEADS + HPI_HEADS) + r")" + _A,
                   re.I | re.M)


def extract_hpi(text: str):
    """Return (section_text, source_tag). Falls back to the leading chars."""
    for pat in _HPI_PATS:
        m = pat.search(text)
        if not m:
            continue
        start = m.end()
        m2 = _STOP.search(text, start)
        body = text[start:m2.start()] if m2 else text[start:start + FALLBACK_CHARS]
        body = re.sub(r"\s+", " ", body).strip()
        if len(body) >= MIN_SECTION_CHARS:
            return body, f"section:{m.group(1).lower()}"
    return text[:FALLBACK_CHARS], "fallback"


def add_variants(df):
    out = df.copy()
    got = [extract_hpi(t) for t in out["text"]]
    out["text_full"] = out["text"]
    out["text_hpi"] = [g[0] for g in got]
    out["hpi_source"] = [g[1] for g in got]
    out["n_chars_full"] = out["text_full"].str.len()
    out["n_chars_hpi"] = out["text_hpi"].str.len()
    return out.drop(columns=["text"])


ER_COLS = ["encounterkey", "ED_Provider_Notes_Text", "primarychiefcomplaintname",
           "primaryeddiagnosisname", "sex", "Age"]


def er_reason(check_only=False):
    df = pd.read_csv(ER_CSV, usecols=ER_COLS, low_memory=False)
    df = df[df["ED_Provider_Notes_Text"].notna()].copy()
    df["text"] = df["ED_Provider_Notes_Text"].astype(str)
    df = df[df["text"].str.len() >= MIN_CHARS]
    print(f"ER-Reason: {len(df):,} eligible ED notes")
    if check_only:
        s = df.sample(n=min(300, len(df)), random_state=SEED)
        tags = [extract_hpi(t)[1] for t in s["text"]]
        print("  section yield (300-doc probe):",
              pd.Series(tags).str.split(":").str[0].value_counts().to_dict())
        return None
    smp = df.sample(n=N_DOCS, random_state=SEED).reset_index(drop=True)
    smp = add_variants(smp)
    smp["doc_id"] = smp["encounterkey"].astype(str)
    smp["meta_specialty"] = "Emergency Medicine"
    smp["meta_note_type"] = "ED Provider Note"
    smp["meta_primary_diagnosis"] = smp["primaryeddiagnosisname"].fillna(
        smp["primarychiefcomplaintname"]).fillna("Unknown")
    smp["meta_secondary"] = "none"
    smp["meta_demographics"] = (smp["Age"].astype(str) + "y " +
                                smp["sex"].fillna("unknown").astype(str))
    return smp[["doc_id", "text_full", "text_hpi", "hpi_source",
                "n_chars_full", "n_chars_hpi", "meta_specialty",
                "meta_note_type", "meta_primary_diagnosis", "meta_secondary",
                "meta_demographics"]]


def mimic_note(check_only=False):
    keep, n_tot = [], 0
    with gzip.open(MIMIC_DIS, "rt") as f:
        for chunk in pd.read_csv(f, usecols=["note_id", "subject_id", "hadm_id",
                                             "text"],
                                 chunksize=20_000, low_memory=False):
            n_tot += len(chunk)
            c = chunk[chunk["text"].notna()].copy()
            c["text"] = c["text"].astype(str)
            c = c[c["text"].str.len() >= MIN_CHARS]
            if len(c):
                keep.append(c.sample(frac=0.02, random_state=SEED))
    pool = pd.concat(keep, ignore_index=True)
    print(f"MIMIC-IV-Note: {n_tot:,} discharge summaries | pool {len(pool):,}")
    if check_only:
        s = pool.sample(n=min(300, len(pool)), random_state=SEED)
        tags = [extract_hpi(t)[1] for t in s["text"]]
        print("  section yield (300-doc probe):",
              pd.Series(tags).str.split(":").str[0].value_counts().to_dict())
        return None
    smp = pool.sample(n=N_DOCS, random_state=SEED).reset_index(drop=True)
    smp = add_variants(smp)
    smp["doc_id"] = smp["note_id"].astype(str)
    smp["meta_specialty"] = "Unknown"
    smp["meta_note_type"] = "Discharge Summary"
    smp["meta_primary_diagnosis"] = "Unknown"
    smp["meta_secondary"] = "none"
    smp["meta_demographics"] = "adult"
    return smp[["doc_id", "hadm_id", "text_full", "text_hpi", "hpi_source",
                "n_chars_full", "n_chars_hpi", "meta_specialty",
                "meta_note_type", "meta_primary_diagnosis", "meta_secondary",
                "meta_demographics"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    for p in (ER_CSV, MIMIC_DIS):
        if not p.exists():
            raise SystemExit(f"not found: {p}")
    OUT.mkdir(exist_ok=True)

    er, mi = er_reason(a.check), mimic_note(a.check)
    if a.check:
        print("\nsection yield is the number to watch: a high fallback share "
              "means\nthe hpi variant is really 'leading 2,500 chars' and "
              "should be named so.")
        return

    for name, df in (("er_reason", er), ("mimic_discharge", mi)):
        f = OUT / f"{name}_n{N_DOCS}_seed{SEED}_variants.csv"
        df.to_csv(f, index=False)
        src = df["hpi_source"].str.split(":").str[0].value_counts()
        print(f"\n{name} -> {f}")
        print(f"  full: median {int(df.n_chars_full.median()):,} chars "
              f"(~{int(df.n_chars_full.median()/4):,} tokens)")
        print(f"  hpi : median {int(df.n_chars_hpi.median()):,} chars "
              f"(~{int(df.n_chars_hpi.median()/4):,} tokens)")
        print(f"  hpi source: {src.to_dict()}")
        frac = (df.n_chars_hpi / df.n_chars_full).median()
        print(f"  hpi is {100*frac:.0f}% of the full note (median)")

    print("\n20 conditions: 2 corpora x 2 variants x 2 query formats, "
          "against\nPaper 12's 6. Query generation (step 2) runs on the "
          "'hpi' text for the\nhpi variant and 'full' for the full variant - "
          "but note the queries are\nderived from METADATA, not body text, so "
          "one query set serves both\nvariants. Only the documents differ.")
    print("\nDO NOT commit expA_corpora/ - credentialed note text.")


if __name__ == "__main__":
    main()
