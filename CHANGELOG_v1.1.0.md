# Changes in v1.1.0

This release supersedes v1.0.0. **Results in v1.0.0 do not reproduce the
submitted manuscript** and should not be used.

## Why

v1.0.0 grouped configurations by *nominal mechanistic tier* — a classification
by training objective. The manuscript's theorem defines the benefited and harmed
subgroups by *measured response*. One configuration
(E5-Mistral-7B-ablation, benchmark mean −0.016) sits in the nominal benefit tier
while measuring negative, so the two groupings disagree.

Reassigning it, as the theorem requires, changes the base case:

| | v1.0.0 (nominal tier) | v1.1.0 (response-defined) |
|---|---|---|
| d_ben | +0.0627 | +0.0757 |
| d_harm | −0.0867 | −0.0766 |
| p\* | 0.5805 | 0.5028, reported as 0.50 |
| p\* at ε = 1e-7 | 1.0607 — not a probability | 0.6993 |
| six-condition screening requirement | 0.007 | 0.071 |

The screening requirement moves an order of magnitude because that configuration
has the panel's largest between-condition SD (0.200); adding it to the harmed
group lowers six-condition specificity from 0.996 to 0.936, and the requirement
is dominated by the false-positive term at low prior probability.

## Analysis code

- `make_appendices.py` — regularisation sweep now groups by measured sign at
  each ε, and records group sizes, which change across the sweep. System A51
  recoded from affected to unknown: its primary article names a model family for
  context embeddings but does not report the extraction step, so it does not meet
  the stated criterion. System A50's coding basis narrowed to what the primary
  source supports. Appendix 2 gains a benchmark-response-group column so the
  48-of-52 agreement figure is auditable. Appendix 3's tier column relabelled,
  since the reference sign is the measured full-sample effect rather than the
  nominal tier.
- `make_figures.py` — Figure 2's threshold line and screening curve updated to
  the response-defined values; Figure 3 annotates the one configuration whose
  benchmark response group differs from its nominal tier; PNG export raised to
  600 dpi.
- `qc_manuscript.py` — base case switched to response-defined grouping. The
  rung-2 check is inverted: under the new threshold the proxy interval straddles
  rather than falls below, so the correct check is that the manuscript reports it
  as decision-indeterminate.

## Appendices

Regenerated: `appendix2_transport_matrix.csv`, `appendix3_screen_resampling.csv`,
`appendix5_parameters.csv`, `appendix5_epsilon_sweep.csv`,
`appendix5_epsilon_percondition.csv`, `appendix1_coding_definitions.txt`.

New: `appendix2_legend.txt` (corpus construction, section extraction, and why a
section extract is the primary analysis) and `appendix4_generation_protocol.txt`
(query-generation protocol with prompts verbatim, the four deviations, and the
grounding check).

No sampling interval is now assigned to p\*: the configuration panel is purposive
rather than a probability sample, and the response groups are defined from the
same measured effects, so a conventional interval conditional on that
classification would not have a frequentist interpretation for any deployment
population. Structural sensitivity from the regularisation sweep (0.32 to 0.70)
is reported instead.

## Build

`build/build_mmas.js` and `build/mma_src/` generate the eight multimedia
appendix documents as submitted. Node with the `docx` package is required.
