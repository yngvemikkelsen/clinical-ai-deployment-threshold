# Clinical AI Deployment Threshold
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21990809.svg)](https://doi.org/10.5281/zenodo.21990809)
Replication materials for:

**Mikkelsen Y. Deployment-Specific Benefit and Harm in Clinical Artificial
Intelligence: Decision-Analytic Derivation and Multi-Institutional Evaluation of
a Deployment Threshold. 2026.**

## Overview

Clinical AI operates in a health-care environment heterogeneous in biology,
clinical practice, treatment effects, and documentation. Where an intervention's
effect changes sign across settings, its benefit–harm balance is
deployment-specific, and the variable that determines which side a given
deployment falls on is a property of the deployed system rather than of the
patient population.

This repository contains the decision model, the empirical work, and the
analysis code behind three claims:

1. **A deployment threshold.** An intervention with sign-changing effects is
beneficial across a setting only where the prevalence of the benefited,
response-defined subgroup exceeds

    p* = (|d_harm| + T/M) / (d_ben + |d_harm|)

    Empirically T/M = 4.3 × 10⁻⁶, so the exact threshold of 0.502810 is
indistinguishable from its zero-cost limit of 0.502782, and both are reported
as 0.50. Across the six observed regularisation settings, recomputing the
response-defined groups at each gives thresholds from 0.32 to 0.70.

2. **Published evidence cannot establish whether the condition is met.** No
system in a 55-system extraction frame had been evaluated under the
intervention, so the response-defined fraction is identified only under explicit
classification assumptions. Under no assumption the bound is the trivial
interval 0–1.

3. **Local screening has two independent requirements.** Cross-condition
evidence breadth sets the prior probability a site must be able to assert,
falling from 0.2101 at one scored condition to 0.0712 at six. Within-site
sample size sets whether the screen is reliable at all: specificity was 0.328
at ten documents, so a site more often concludes that a harmed system benefits.

Grouping is by **measured response**, not by nominal training objective. The two
disagree for one configuration, and that disagreement moves the threshold; see
`CHANGELOG_v1.1.0.md`.

## Repository structure

```
.
├── scripts/
│   ├── clinical_ai_deployment_model.py  canonical decision model; self-verifying
│   ├── paper9_sensitivity.py            economic sensitivity analyses
│   ├── paper9_boundary_sweep.py         classification-boundary sweep
│   ├── make_appendices.py               generates the appendices
│   ├── make_figures.py                  generates figures 1–3
│   ├── qc_manuscript.py                 numerical consistency gate
│   ├── docx2csv.py                      extracts appendix tables for the gate
│   ├── appendix5_parameters.csv         generated parameter table
│   └── appendix5_sensitivity_tables.md  generated sensitivity tables
├── experiments/
│   ├── transport/                       multi-institutional evaluation
│   │   ├── expA_01_sample.py            corpus sampling
│   │   ├── expA_01b_variants.py         section extraction, document variants
│   │   ├── expA_02_queries_v2.py        local metadata-derived query generation
│   │   ├── expA_04_panel.py             13-configuration panel, ZCA, MRR@10
│   │   ├── expA_05_curve.py             screening curve, held-out and pooled
│   │   ├── expA_06_bootstrap.py         local-screen resampling
│   │   └── expA_07_bridge.py            query-generator bridging check
│   └── replication/                     code-search boundary test
│       ├── paper9_expB_diera.py         replication driver
│       └── paper9_expB_colab.py         fine-tuning and embedding extraction
├── analysis/
│   └── fix_workbook_a50.py              extraction-frame coding correction
├── appendices/                          multimedia appendices 1–9, and the
│                                        generated CSV/TXT they are built from
├── build/                               appendix build tooling
├── figures/                             figures 1–3, 600 dpi PNG and vector PDF
├── frame/                               55-system extraction frame
├── results/                             aggregate outputs, no patient data
├── CHANGELOG_v1.1.0.md
├── README.md
└── LICENSE
```

## Data availability

**No patient-level data are redistributed.** The two institutional corpora are
PhysioNet credentialed-access resources and must be obtained directly:

| Corpus             | Source                                 | Access                  |
| ------------------ | -------------------------------------- | ----------------------- |
| MIMIC-IV-Note v2.2 | Beth Israel Deaconess Medical Center   | PhysioNet, credentialed |
| ER-Reason v1.0.0   | University of California San Francisco | PhysioNet, credentialed |

`results/` contains aggregate outputs only: effect sizes, retrieval metrics,
resampling counts, and threshold derivations. Sampled note text, generated
queries derived from those notes, and document embeddings are **not** included,
as all three are derivative of credentialed data.

The benchmark corpora and their published queries come from the companion study
and are available in its own repository
(github.com/yngvemikkelsen/clinical-rag-retrieval-benchmark).

## Reproducing key results

### Decision model and thresholds

```
python scripts/clinical_ai_deployment_model.py
```

Standard library only. The script computes the response-defined groups, the
thresholds, the screen operating characteristics by number of scored
conditions, the exact screening requirement with the fixed screening cost S and
the conditional deployment cost T carried separately, and the monetary scale. It
then verifies each against the value printed in the manuscript and exits
non-zero on any disagreement.

Expected: 6 benefited and 7 harmed configurations, d_ben = +0.075745,
d_harm = −0.076593, p\* = 0.502782 zero-cost and 0.502810 exact,
V = €11,529, M = €344.0 million, T = €1,468, screening requirements 0.2101 to
0.0712 for 1 to 6 conditions, and a regularisation range of 0.3163 to 0.6993.

The script deliberately omits a prevalence prior, population strategy
probabilities, EVPI and EVPPI, and any "minimum viable" condition count. Each
requires a distribution for the response-defined prevalence, which the
publication frame does not identify. Its docstring states this explicitly.

### Economic sensitivity analyses

```
python scripts/paper9_sensitivity.py
```

Requires numpy and scipy. One-way sensitivity on the illustrative parameters, a
50,000-draw probabilistic analysis, the time-horizon and discount-rate analysis,
the exchange-rate analysis, and the published European adverse-event cost
scenarios. Writes the tables reproduced in Multimedia Appendix 9.

### Classification-boundary sweep

```
python scripts/paper9_boundary_sweep.py
```

Standard library only. Sweeps the rule that assigns a configuration to the
harmed group and recomputes the threshold at each step.

Expected: thresholds 0.58, 0.50, 0.43 and 0.35 across ±0.05 in mean ΔMRR@10,
with grouping stable between −0.0157 and +0.0369.

### Screening curve and design table

```
python experiments/transport/expA_05_curve.py \
    --p12 results/epsilon_sensitivity.parquet \
    --expa results/expA_panel_results.csv
```

### Local-screen resampling

```
python experiments/transport/expA_06_bootstrap.py --emb <embedding dir>
```

Requires cached embeddings, which are not redistributed. Run
`expA_04_panel.py --variant hpi --cache-embeddings` first, having obtained the
corpora from PhysioNet.

Expected: specificity 0.328 at ten documents rising to 0.974 at 75.

### Code-search boundary replication

```
git clone https://github.com/drndr/code_isotropy.git
python experiments/replication/paper9_expB_diera.py evaluate --all
```

Expected: baselines reproduce published values to within 0.001 across 18 of 18
primary cells; the sign structure does not reproduce at zero regularisation.

### Appendices, figures, and the consistency gate

```
python scripts/make_appendices.py
python scripts/make_figures.py
python scripts/qc_manuscript.py --md <manuscript>.md --docx <manuscript>.docx
```

`qc_manuscript.py` is a gate, not a proof. It recomputes the principal
load-bearing numbers from their source artefacts, checks that superseded values
have not reappeared, and fails on any mismatch. It does not verify every value
in every table, the query-generator correlations, the figures, or bibliographic
metadata, and it cannot judge whether a claim is warranted by its evidence.

## Notes on the extraction frame

`frame/p_tier2_wide_frame_extraction_audit_v2.xlsx` supersedes an earlier
version. One system's audit note recorded a coding rationale that the analysis
established was invalid; the corrected note states the basis actually supported
by the primary article, and a `Coding provenance` column records the superseded
rationale so the change is auditable. Tier codings are unchanged in all frames.

Every coding in the frame is **mechanistic**: it records an implementation
characteristic expected to correspond to the response-defined subgroup, not a
measured response. No system in the frame was evaluated under the intervention.
The counts therefore support proxy-based partial identification under stated
assumptions and do not bound response-defined prevalence.

## Citation

```
@article{mikkelsen2026deployment,
  title={Deployment-Specific Benefit and Harm in Clinical Artificial
         Intelligence: Decision-Analytic Derivation and Multi-Institutional
         Evaluation of a Deployment Threshold},
  author={Mikkelsen, Yngve},
  year={2026}
}
```

## Related work

This repository is one component of a programme on retrieval failure in clinical
AI:

- Mikkelsen Y. Clinical Context Variables Collectively Rival Model Choice in
Embedding-Based Retrieval. *JMIR Med Inform.* 2026;14:e94241.
[Repository](https://github.com/yngvemikkelsen/clinical-rag-retrieval-benchmark)
- Mikkelsen Y. Effects of Model Choice, Corpus Context, and Post Hoc Correction
on Layer-Level Embedding Degradation in Clinical Document Retrieval.
*JMIR Med Inform.* 2026;14:e99639.

## License

Code: MIT License. Aggregate results: CC BY 4.0. No patient-level data are
included; corpus access is governed by the respective PhysioNet data use
agreements.

## Contact

Yngve Mikkelsen, MD MSc DBA

ORCID: 0000-0003-1543-3805
