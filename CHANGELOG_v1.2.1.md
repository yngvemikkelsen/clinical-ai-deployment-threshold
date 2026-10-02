# v1.2.1 — JMIR AI ms#109863, minor revision (Decision B)

Corrections arising from the three editorial comments on the major revision, plus
defects found while verifying them. No analysis was re-run and no reported
numerical result changed.

## Editorial comment 1 — name the evaluated technique in the abstract

- `build/mma_src/a8_cheersai.txt` — CHEERS-AI Item 2 AI elaboration moves from
  partly reported to reported, since its stated reason was that ZCA whitening was
  named in Methods rather than the abstract. Summary counts now 29 / 5 / 4.
  Item 1 remains partly reported: it concerns the title, which is unchanged.

## Editorial comment 2 — truncation rule

- `experiments/transport/expA_02_queries_v2.py` — the implemented criterion is
  **non-ASCII**, not non-Latin; the docstring said otherwise. The pattern is now a
  module constant `NON_ASCII` used by both the cleaner and the post-clean check, so
  the two cannot drift. Documented that cleaning is applied inline, that raw
  pre-truncation generations are therefore not retained, and that generation runs at
  temperature 0.3 with no seed, so a re-run does not reproduce the original output.
- The same file **did not parse on Python 3.11** — a backslash inside an f-string
  expression, which requires 3.12. Fixed by the same constant.
- `build/mma_src/a4_protocol.txt` — non-ASCII wording; added a `random seed` row;
  removed the "non-ASCII in keyword output" row, which was computed on cleaned text
  and was zero by construction.
- `appendices/appendix4_generation_protocol.txt` — same wording correction.

## Editorial comment 3 — code-search subgroup structure

- `scripts/make_appendices.py` — `appendix6()` carried a published-baseline dict of
  only 18 entries and a comment asserting that no published baseline exists for the
  fine-tuned CodeBERT. That is wrong: Diera Table 2 publishes it for all six
  CodeSearchNet languages, and publishes R for the three released checkpoints. The
  dict is now complete (27 cells), rows are labelled by gate scope, and the absolute
  difference is rounded rather than carrying float noise.
- `appendices/appendix6_baseline_replication.csv` — 18 → 27 rows. Every previously
  present value is unchanged.
- `experiments/replication/paper9_expB_diera.py` — the validation gate applied the
  published-baseline criterion to a locally fine-tuned checkpoint and to an
  extension language, so a fresh clone printed STOP. It is now scoped to the 18
  released-checkpoint cells, where the largest deviation is 0.0010; the other nine
  cells are printed with their deviations and the reason they are not gated.
  `threshold()` grouped by training objective rather than measured sign, reporting a
  p\* the paper does not contain and comparing it to a stale clinical value of
  0.5816. It now groups by measured sign at each regularisation value, as the
  manuscript does, and reports the mechanistic grouping separately as the published
  claim under test. The header's withdrawn "demonstrated generality" and
  "tuning-rescue is domain-dependent" framing is replaced by what the run found.

## Defects found while verifying the above

- `appendices/MMA5_decision_model_parameters.docx` — the parameter table omitted
  **FX, S and T**: the exchange rate and the two costs the S/T correction turns on.
  The CSV has carried them since the major revision; the document was never rebuilt.
  Added.
- `build/build_mmas.js` — resolved `appendices/` against the caller's working
  directory but `mma_src/` against `build/`, so there was no directory from which
  the build could run, and the output path was absolute to the authoring machine.
  All three now resolve from the repository root, with `PAPER9_MMA_OUT` overriding
  the output directory.
- The same script had drifted from the submitted document convention, emitting
  "Multimedia Appendix N." titles and "Table N.M" numbering where the submitted
  appendices use bare titles and "Table SM". Restored, and `build/mma_src/a2_legend.txt`
  realigned to the committed wording.

## Reproducibility status of the appendices

After these changes, Multimedia Appendices 1, 2, 3, 4, 6, 7 and 8 are byte-faithful
rebuilds from `build/build_mmas.js` and the files in `appendices/`.

**Appendix 5 is not.** Its committed document carries hand-rounded values
(-0.0766 where the CSV holds -0.076593, 0.4240 where it holds 0.42395634920634917)
at mixed precision that no formatter in the build reproduces. The three added rows
were therefore inserted into the committed document rather than regenerated, and
rebuilding Appendix 5 from source would replace its rounded values with raw floats.
This is recorded rather than silently left for a later reader to discover.
