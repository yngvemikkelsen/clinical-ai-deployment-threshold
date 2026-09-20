# Paper 9 — JMIR AI ms#109863, major revision package

Everything here is regenerated against the revised manuscript and verified.
`scripts/qc_manuscript.py` is the release gate: it recomputes every
load-bearing number from source and fails on any disagreement.

## Contents

paper9_ms109863_rev1.docx        revised manuscript (clean copy for upload)
paper9_response_to_editor.txt    point-by-point response; plain text, for
                                 pasting into the notification field

appendices/
  MMA1_system_extraction_frame.docx
  MMA2_transport_matrix.docx
  MMA3_screen_resampling.docx
  MMA4_query_generation.docx
  MMA5_decision_model_parameters.docx   UPDATED
  MMA6_codesearch_replication.docx
  MMA7_CHEERS2022_checklist.docx
  MMA8_CHEERS-AI_checklist.docx         UPDATED
  MMA9_sensitivity_analyses.docx        NEW

figures/   600 dpi PNG for upload, PDF as the vector original
  fig1_rev20260919.png / .pdf           REGENERATED
  fig2_rev20260919.png / .pdf           REGENERATED
  fig3_rev20260919.png                  unchanged

scripts/
  clinical_ai_deployment_model.py   canonical model; stdlib only; self-verifying
  paper9_sensitivity.py             economic sensitivities -> Tables S6-S10
  paper9_boundary_sweep.py          classification boundary -> Table S11
  make_figures.py                   Figures 1-3; imports the canonical model
  make_appendices.py                appendix generation
  qc_manuscript.py                  release gate, 71 checks; exits 0 on the
                                    current package
  docx2csv.py                       extracts appendix tables for QC
  appendix5_parameters.csv          regenerated parameter table

## What changed in this revision

MMA5   C_event EUR 11,422 -> EUR 11,494 with the conversion stated; the
       "reported without reconciliation" note replaced by the actual
       comparison with Durand 2024 and Laroche 2025; lambda annotated as
       nominal and unindexed; per-condition values rounded to 4 decimals.
MMA2   Header-frequency denominator removed rather than guessed; the
       unquantified "contributes negligibly" covariance claim replaced by what
       was actually done.
MMA9   NEW. Tables S6-S11: the economic and classification sensitivity
       analyses. Built as its own appendix rather than appended to MMA5,
       because the wide tables need their own page setup.
PSA    The adverse-event cost Gamma scale was a stale literal (4569) against
       a declared mean implying 4597. It is now derived as C_EVENT/2.5, so
       the declared and sampled means cannot diverge. Corrected M range:
       EUR 38.3M to 1,131.8M.
FX     The 2026 year-to-date rate is removed. An incomplete year has no
       annual average, and the figure could not be sourced. The sensitivity
       now uses the completed 2024 and 2025 averages only.
Tbl 6  The minimum-prior-probability column is removed: those values used
       the zero-cost form with institutional magnitudes, and the document
       axis carries no condition count, so no exact value exists per row.
MMA8   AI 10 no longer claims the resampling establishes a minimum
       measurement intensity, which the revision explicitly rejects.
Fig 1  right-hand box now reads "documents in joint fit/evaluation sample
       per condition"; boxes resized so the text fits.
Fig 2  y-axis "minimum prior probability of benefit"; panel B x-axis
       "documents in joint fit/evaluation sample"; the external curve now
       uses the same exact S/T equation as the benchmark curve with the
       institutional effect magnitudes.
Model  the legacy prevalence prior, EVPI/EVPPI and "hard minimum four
       conditions" are gone; the screening threshold carries S and T
       separately rather than folding an assumed prevalence into one cost.

## Reproducing

    cd scripts
    python3 clinical_ai_deployment_model.py          # base case + verification
    python3 paper9_sensitivity.py                    # Tables S6-S10
    python3 paper9_boundary_sweep.py                 # Table S11
    python3 make_figures.py                          # Figures 1-3
    python3 qc_manuscript.py --md ms.md --docx ../paper9_ms109863_rev1.docx

The model and boundary sweep need only the standard library; the
sensitivity script needs numpy and scipy; the figures need matplotlib and
pandas; the QC script needs pandas.

## One open item, for the author

Reference 27 (Siverskog) is an applied economic evaluation reporting EVPI,
not a methodological authority for value-of-information analysis. Either
soften the sentence to "is used in health economic evaluation" or add a
standard methodological source. This is a citation-quality judgement, not a
factual error.

MMA2's header-frequency sentence previously read "4 of 200 sampled notes"
against an analytic sample of 100 per corpus. Rather than guess a denominator,
the count is removed and the sentence now reads "a small minority of the
sampled notes". If a separate 200-note header-discovery set does exist, put
the exact figure back and say which set it refers to.

## Known validator notes

MMA9 is generated by pandoc. A strict OOXML validator reports four cosmetic
complaints about pandoc's own default style definitions and one about
settings.xml. Word and LibreOffice both open and render the file correctly;
the content, tables and page setup are unaffected. Every other appendix
validates cleanly.
