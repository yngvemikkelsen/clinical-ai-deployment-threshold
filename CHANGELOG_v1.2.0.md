# Changes in v1.2.0

This release supersedes v1.1.0 and accompanies the revised manuscript submitted
to JMIR AI (ms#109863). **v1.1.0 reproduces the response-defined base case
correctly but does not reproduce the revised screening algebra, the corrected
probabilistic analysis, or the stated exchange rate**, and its decision model
carries machinery the revised manuscript no longer uses.

## Why

Three defects were found while preparing the revision. Each is corrected here.

**The screening cost embedded an assumed prevalence.** v1.1.0 combined the fixed
cost of scoring the screen with the cost of deploying the correction into a
single strategy cost, weighting the latter by the expected rate at which the
screen selects a system. That rate depends on the prevalence p, so the combined
figure assumed a value for the quantity the manuscript states is not identified,
and the displayed inversion for the screening threshold was not exact. The two
components are now carried separately: S, the fixed screening cost, incurred
whichever way the screen reads; and T, the deployment cost, incurred only on the
systems the screen selects.

    p*_screen = [(1 − sp)(h + t) + s] / [se(d_ben − t) + (1 − sp)(h + t)]

with s = S/M, t = T/M and h = |d_harm|.

**The probabilistic analysis sampled from the wrong distribution.** The
adverse-event cost was drawn as Gamma(2.5, 4569) against a declared mean
implying a scale of 4597, so the sampled and declared means diverged. The scale
is now derived from the declared mean rather than written as a literal, which
makes the divergence impossible.

**The exchange rate was implicit and undated.** No rate, basis or date appeared
anywhere, and the rate in use corresponded to no stated period.

| | v1.1.0 | v1.2.0 |
|---|---|---|
| screening cost | single K_C with p folded in | S and T carried separately |
| adverse-event cost | EUR 11,422, implicit rate | EUR 11,494 at 11.6276 NOK/EUR |
| exchange rate | not stated | Norges Bank annual average 2024 |
| PSA gamma scale | 4569 literal | derived as C_event / 2.5 |
| PSA value exposed, 95% | EUR 38.0M–1,124.8M | EUR 38.3M–1,131.8M |
| p\* exact | 0.502811 | 0.502810 |
| p\* zero-cost | 0.502783 | 0.502782 |
| six-condition requirement | 0.0711 | 0.0712 |

The two threshold values and the screening requirement move in the sixth and
fourth decimal respectively because v1.1.0 computed them from group means
rounded to six figures. The per-configuration table now carries full precision,
so every derived quantity follows from the archived data rather than from a
rounded intermediate.

## Decision model

`model/clinical_rag_he_model_v5.py` is **removed** and replaced by
`scripts/clinical_ai_deployment_model.py`.

The v5 model grew around an earlier analysis and retained machinery the revised
manuscript does not use. It is deleted rather than kept, because running it
returns results that contradict the paper.

The replacement is standard library only, and deliberately omits:

- **any prevalence prior.** Earlier versions carried a Beta prior derived from a
  k = 2 of 55 mechanistic count. The manuscript states that the publication
  frame does not identify a distribution for the response-defined prevalence,
  that the no-assumption bound is 0 to 1, and that the 1/55 count is descriptive
  and is not entered into the model.
- **population strategy probabilities, EVPI and EVPPI.** Each requires a
  distribution for p.
- **any "minimum viable" or "hard minimum" condition count.** The revised paper
  reports 1 to 6 conditions as a prior-dependent design trade-off.
- **the zero-cost screening formula**, superseded by the exact S/T expression.

The script computes every load-bearing value and verifies each against the
value printed in the manuscript, exiting non-zero on any disagreement.

## Analysis code

- `scripts/paper9_sensitivity.py` — **new.** One-way sensitivity on the
  illustrative parameters, a 50,000-draw probabilistic analysis, the
  time-horizon and discount-rate analysis, the exchange-rate analysis, and the
  published European adverse-event cost scenarios. Imports the canonical model
  for all effect sizes, costs and threshold functions rather than maintaining
  its own base case. The exchange-rate analysis uses the completed 2024 and 2025
  Norges Bank annual averages; an incomplete year has no annual average and is
  not reported as one.
- `scripts/paper9_boundary_sweep.py` — **new.** Sweeps the rule assigning a
  configuration to the harmed group over ±0.05 in mean ΔMRR@10 and recomputes
  the threshold at each step, using the exact S/T requirement. Imports M, T and
  S from the canonical model.
- `scripts/make_figures.py` — Figure 2 now derives the screen accuracies and the
  requirement from the canonical model instead of carrying literals, and the
  external curve uses the same exact S/T equation as the benchmark curve with
  the institutional effect magnitudes read from the transport matrix, asserting
  they match the values reported in the manuscript. Axis labels corrected:
  panel A reads "minimum prior probability of benefit", panel B "documents in
  joint fit/evaluation sample". Figure 1's equation is set on two lines and its
  right-hand box resized so the text fits.
- `scripts/make_appendices.py` — the parameter table gains rows for the exchange
  rate and for S and T separately, replacing the single strategy-cost row. Data
  paths are resolved relative to the script rather than to an absolute location.
- `scripts/qc_manuscript.py` — recomputes Table 2 with the exact S/T expression
  from the canonical model rather than reimplementing it, asserts that the six
  screening rows are actually parsed before checking them, takes the
  authoritative word count from the DOCX, and accepts either pipe or grid table
  rendering. Its own description is corrected from "recomputes every
  load-bearing number" to a gate over the principal ones.
- `scripts/docx2csv.py` — **new.** Extracts appendix tables to CSV so the gate
  can run against the submitted appendix documents.

## Appendices

**New: Multimedia Appendix 9**, the economic and classification sensitivity
analyses — Tables S6 to S11, generated by `paper9_sensitivity.py` and
`paper9_boundary_sweep.py`.

`appendix5_parameters.csv` regenerated: the adverse-event cost row corrected to
EUR 11,494 with its conversion stated; the note claiming the Norwegian figure
was reported without reconciliation to the European estimates replaced by the
comparison actually performed; rows added for the exchange rate and for S and T;
the willingness-to-pay row annotated as applied at nominal policy value without
indexation.

Multimedia Appendix 5's per-condition table now displays four decimals rather
than raw floating-point values, and its sensitivity and specificity rows
describe the two design axes separately rather than collapsing them.

Multimedia Appendix 8, item AI 10, no longer states that the resampling analysis
specifies a minimum measurement intensity. The revised manuscript states the
opposite: because the same documents fit the transform and evaluate retrieval,
the experiment does not identify separate requirements for either task.

Multimedia Appendix 2 no longer reports a header frequency against a denominator
the appendix does not establish, and no longer describes a form-field label's
contribution to the document covariance as negligible, which was not quantified.

## Figures

Figures 1 and 2 regenerated at 600 dpi. The superseded set
(`figure1_evidence_sequence`, `figure2_screening_constraints`,
`figure3_response_heatmap`) is removed; the current files are
`fig1_rev20260919`, `fig2_rev20260919` and `fig3_rev20260919`.

## Repository layout

Analysis code consolidated under `scripts/`. `analysis/` retains only
`fix_workbook_a50.py`. The three scripts previously duplicated across both
directories are gone from `analysis/`, so there is one authoritative copy of
each.

The README is rewritten against the current numbers. The version in v1.1.0
reported p\* = 0.5805, a screening requirement falling from 0.217 to 0.007, and
a decision model producing value-of-information results — all superseded.
