#!/usr/bin/env python3
"""
Paper 9 - Experiment A, step 2 (v2): metadata-derived queries, generated locally.

COMPLETE FILE. Every fix established during testing is baked in here rather than
applied as a patch - an earlier round of patch-then-replace lost all of them.
Filename is versioned so it can never overwrite a working copy.

Reproduces Paper 12's protocol (doi:10.2196/99639, Multimedia Appendix 1). Both
corpora are PhysioNet credentialed-access, so note text cannot go to a cloud API
and queries are generated locally.

DEVIATIONS FROM APPENDIX 1, ALL FORCED, ALL DELIBERATE
------------------------------------------------------
1. qwen2.5:14b-instruct in place of GPT-4o. Required by the DUA. qwen3:30b was
   tried first and rejected: it emits reasoning regardless of /no_think or
   think=False, which lands in the response field and consumes the whole token
   budget before any answer. Every field came back "Unknown".
2. JSON mode ("format": "json") on the extraction stage. Without it prose leaks
   into the structured output and parsing fails.
3. One added instruction - "Use ONLY the values given..." - plus normalising a
   missing age to Appendix 1's own 'adult' default. Without both, the model
   supplies a plausible age that was never in the metadata ("Female" -> "A
   45-year-old female"), breaking the protocol's guarantee that queries derive
   only from extracted metadata.
4. Output cleaning: truncate at the first non-Latin character (qwen2.5
   occasionally breaks into Chinese mid-generation), drop underscores, drop a
   trailing incomplete sentence rather than emitting a fragment.

Step 3 bridges the combined effect against the published GPT-4o queries on a
public corpus.

PROTOCOL, VERBATIM FROM APPENDIX 1
----------------------------------
Two stages. Stage 1 extracts structured metadata from the note. Stage 2 receives
ONLY that metadata, never the document text, "by design, to limit lexical
leakage from document into query". Temperature 0.0 extract / 0.3 generate, up to
3 retries with exponential backoff, and on persistent failure the primary
diagnosis string is used as the query. Placeholder defaults: specialty
'Unknown', note type 'Clinical Note', primary diagnosis 'Unknown', secondary
'none', demographics 'adult'.

ONE QUERY SET SERVES BOTH DOCUMENT VARIANTS. Queries derive from metadata, not
body text. Stage 1 runs on the FULL note for both variants so the query set is
identical and only the documents differ.

Usage:
    ollama serve                                   # separate terminal
    python expA_02_queries_v2.py --probe           # 3 docs, prints, writes nothing
    python expA_02_queries_v2.py --corpus mimic_discharge
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

MODEL = "qwen2.5:14b-instruct"
OLLAMA = "http://localhost:11434/api/generate"
TEMP_EXTRACT, TEMP_QUERY = 0.0, 0.3
NUM_PREDICT_EXTRACT, NUM_PREDICT_NL, NUM_PREDICT_KW = 300, 150, 60
RETRIES = 3
EXTRACT_CHARS = 6000
CORP = Path("./expA_corpora")
OUT = Path("./expA_queries")

EXTRACT_PROMPT = """Extract metadata from this clinical note. Return ONLY a JSON object with exactly these keys:
"specialty", "note_type", "primary_diagnosis", "secondary_diagnoses", "age", "sex"
Use "Unknown" for any field not stated. "secondary_diagnoses" must be a list of strings.
No prose, no code fences, JSON only.

NOTE:
{text}

JSON:"""

_GUARD = ("Use ONLY the values given. Do not infer, add, or invent any detail "
          "not present above, including age.\n")

NL_PROMPT = ("Based ONLY on this metadata, write a natural language clinical "
             "question (1-2 sentences):\n"
             "Specialty: {specialty} | Note type: {note_type} | Diagnosis: "
             "{primary_diagnosis} | Other: {secondary} | Patient: {demographics}\n"
             + _GUARD + "Query:")

KW_PROMPT = ("Based ONLY on this metadata, output 3-6 clinical search keywords "
             "(space-separated):\n"
             "Specialty: {specialty} | Note type: {note_type} | Diagnosis: "
             "{primary_diagnosis} | Other: {secondary} | Patient: {demographics}\n"
             + _GUARD + "Keywords:")


def strip_think(s: str) -> str:
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S | re.I)
    return re.sub(r"</?think>", "", s, flags=re.I).strip()


def ollama(prompt: str, temperature: float, num_predict: int,
           json_mode: bool = False) -> str:
    last = None
    for attempt in range(RETRIES):
        try:
            payload = {"model": MODEL, "prompt": prompt, "stream": False,
                       "think": False,
                       "options": {"temperature": temperature,
                                   "num_predict": num_predict}}
            if json_mode:
                payload["format"] = "json"
            r = requests.post(OLLAMA, timeout=300, json=payload)
            r.raise_for_status()
            j = r.json()
            out = strip_think(j.get("response", "") or "")
            # some builds route reasoning to a separate field and leave
            # response empty - fall back rather than silently returning ""
            if not out and j.get("thinking"):
                out = strip_think(j["thinking"])
            if not out:
                raise RuntimeError(f"empty response "
                                   f"(done_reason={j.get('done_reason')})")
            return out
        except Exception as e:                                  # noqa: BLE001
            last = e
            body = getattr(getattr(e, "response", None), "text", "")
            print(f"    ollama attempt {attempt+1}/{RETRIES}: "
                  f"{type(e).__name__}: {e}{' | ' + body[:150] if body else ''}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"ollama failed after {RETRIES} attempts: {last}")


def extract_metadata(text: str, note_type_hint: str) -> dict:
    raw = ollama(EXTRACT_PROMPT.format(text=text[:EXTRACT_CHARS]),
                 TEMP_EXTRACT, NUM_PREDICT_EXTRACT, json_mode=True)
    m = re.search(r"\{.*\}", raw, re.S)
    d = {}
    if m:
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            d = {}
    sec = d.get("secondary_diagnoses") or []
    if isinstance(sec, str):
        sec = [sec]
    age, sex = str(d.get("age", "")).strip(), str(d.get("sex", "")).strip()
    age_ok = bool(re.search(r"\d", age)) and age.lower() != "unknown"
    sex_ok = bool(sex) and sex.lower() != "unknown"
    if age_ok and sex_ok:
        demo = f"{age} {sex}"
    elif sex_ok:
        demo = f"adult {sex}"       # no number for the model to anchor on
    elif age_ok:
        demo = age
    else:
        demo = "adult"              # Appendix 1 default
    return {
        "specialty": d.get("specialty") or "Unknown",
        "note_type": d.get("note_type") or note_type_hint or "Clinical Note",
        "primary_diagnosis": d.get("primary_diagnosis") or "Unknown",
        "secondary": ", ".join(str(x) for x in sec) if sec else "none",
        "demographics": demo,
        "_parsed": bool(m and d),
    }


def clean_query(s: str, keyword: bool) -> str:
    s = strip_think(s)
    # qwen2.5 occasionally breaks into Chinese mid-generation
    m = re.search(r"[^\x00-\x7F]", s)
    if m:
        s = s[:m.start()]
    s = s.replace("_", " ")
    s = re.sub(r"^\s*(query|keywords)\s*:\s*", "", s, flags=re.I)
    s = s.strip().strip('"').strip()
    s = s.split("\n")[0].strip()
    if keyword:
        s = re.sub(r"[,;]", " ", s)
        s = " ".join(s.split()[:6])
    else:
        parts = [p for p in re.split(r"(?<=[.?!])\s+", s) if p.strip()]
        if parts and not re.search(r"[.?!]$", parts[-1]):
            parts = parts[:-1] or parts        # never return empty
        s = " ".join(parts[:2]).strip()
    return s


def run(corpus: str, limit=None, probe=False):
    f = next(CORP.glob(f"{corpus}_n*_variants.csv"))
    df = pd.read_csv(f)
    if limit:
        df = df.head(limit)
    print(f"{corpus}: {len(df)} documents from {f.name} | model {MODEL}")

    rows, t0 = [], time.time()
    for i, r in df.iterrows():
        md = extract_metadata(str(r["text_full"]),
                              str(r.get("meta_note_type", "")))
        nl = clean_query(ollama(NL_PROMPT.format(**md), TEMP_QUERY,
                                NUM_PREDICT_NL), False)
        kw = clean_query(ollama(KW_PROMPT.format(**md), TEMP_QUERY,
                                NUM_PREDICT_KW), True)
        rows.append({
            "doc_id": r["doc_id"],
            "query_nl": nl or md["primary_diagnosis"],
            "query_keyword": kw or md["primary_diagnosis"],
            "md_specialty": md["specialty"], "md_note_type": md["note_type"],
            "md_primary_diagnosis": md["primary_diagnosis"],
            "md_secondary": md["secondary"],
            "md_demographics": md["demographics"],
            "md_json_parsed": md["_parsed"],
            "registry_diagnosis": r.get("meta_primary_diagnosis", ""),
        })
        if probe:
            print(f"\n--- doc {i} ---")
            print(f"  extracted: {md}")
            print(f"  registry : {r.get('meta_primary_diagnosis','')}")
            print(f"  nl : {nl}")
            print(f"  kw : {kw}")
        elif (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(df)}  {el/60:.1f} min  "
                  f"eta {el/(i+1)*(len(df)-i-1)/60:.1f} min")

    q = pd.DataFrame(rows)
    if probe:
        return q

    OUT.mkdir(exist_ok=True)
    o = OUT / f"{corpus}_queries_{MODEL.replace(':', '-')}.csv"
    q.to_csv(o, index=False)
    print(f"\n-> {o}")
    print(f"  JSON parsed        : {q.md_json_parsed.sum()}/{len(q)}")
    print(f"  diagnosis 'Unknown': {(q.md_primary_diagnosis=='Unknown').sum()}/{len(q)}")
    print(f"  nl length median   : {int(q.query_nl.str.len().median())} chars")
    print(f"  kw words median    : {int(q.query_keyword.str.split().str.len().median())}")
    inv = sum(1 for _, x in q.iterrows()
              if (mm := re.search(r"(\d+)-year-old", str(x.query_nl)))
              and mm.group(1) not in str(x.md_demographics))
    print(f"  invented ages      : {inv}/{len(q)}  (want 0)")
    print(f"  non-ASCII in kw    : {q.query_keyword.str.contains(r'[^\x00-\x7F]').sum()}")

    reg = q[q.registry_diagnosis.notna() & (q.registry_diagnosis != "") &
            (q.registry_diagnosis != "Unknown")]
    if len(reg):
        def hit(a, b):
            return bool({w for w in re.findall(r"[a-z]{4,}", str(a).lower())} &
                        {w for w in re.findall(r"[a-z]{4,}", str(b).lower())})
        agree = sum(hit(x.md_primary_diagnosis, x.registry_diagnosis)
                    for _, x in reg.iterrows())
        print(f"  vs registry dx     : {agree}/{len(reg)} token overlap")
        print("    loose check only - the note records presentation, the registry")
        print("    records the coded final diagnosis, so disagreement is expected")
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None,
                    choices=["er_reason", "mimic_discharge", "mtsamples"])
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
    except Exception:
        raise SystemExit("ollama not reachable - run `ollama serve`")
    for c in ([a.corpus] if a.corpus else ["er_reason", "mimic_discharge"]):
        run(c, limit=3 if a.probe else None, probe=a.probe)


if __name__ == "__main__":
    main()
