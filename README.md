# Clinical AI Deployment Threshold

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
   beneficial across a setting only where affected-subgroup prevalence exceeds
   p\* = (|d_harm| + K/M) / (d_ben + |d_harm|). Empirically K/M = 4.3 × 10⁻⁶, so
   the threshold is indistinguishable from its zero-cost limit of 0.5805.

2. **Published evidence cannot establish whether the condition is met.** No
   system in a 55-system extraction frame had been evaluated under the
   intervention, so the affected fraction is identified only under explicit
   classification assumptions. Under no assumption the bound is the trivial
   interval 0–1.

3. **Local screening has two independent requirements.** Cross-condition
   evidence breadth sets the prevalence a site must be able to assert, falling
   from 0.217 at one scored condition to 0.007 at six. Within-site sample size
   sets whether the screen is reliable at all: specificity was 0.328 at ten
   documents, so a site more often concludes that a harmed system benefits.

## Repository structure

```
.
├── model/
│   └── clinical_rag_he_model_v5.py     decision model, PSA, EVPI/EVPPI
├── experiments/
│   ├── transport/                      multi-institutional evaluation
│   │   ├── expA_01_sample.py           corpus sampling
│   │   ├── expA_01b_variants.py        section extraction, document variants
│   │   ├── expA_02_queries_v2.py       local metadata-derived query generation
│   │   ├── expA_04_panel.py            13-configuration panel, ZCA, MRR@10
│   │   ├── expA_05_curve.py            screening curve, held-out and pooled
│   │   ├── expA_06_bootstrap.py        local-screen resampling
│   │   └── expA_07_bridge.py           query-generator bridging check
│   └── replication/                    code-search boundary test
│       ├── paper9_expB_diera.py        replication driver
│       └── paper9_expB_colab.py        fine-tuning and embedding extraction
├── analysis/
│   ├── make_appendices.py              generates all six appendices
│   ├── make_figures.py                 generates figures 1-3
│   ├── qc_manuscript.py                numerical consistency check
│   └── fix_workbook_a50.py             extraction-frame coding correction
├── appendices/                         multimedia appendices 1-6
├── figures/                            figures 1-3, PNG and vector PDF
├── frame/                              55-system extraction frame
├── results/                            aggregate outputs, no patient data
├── README.md
└── LICENSE
```

## Data availability

**No patient-level data are redistributed.** The two external corpora are
PhysioNet credentialed-access resources and must be obtained directly:

| Corpus | Source | Access |
|---|---|---|
| MIMIC-IV-Note v2.2 | Beth Israel Deaconess Medical Center | PhysioNet, credentialed |
| ER-Reason v1.0.0 | University of California San Francisco | PhysioNet, credentialed |

`results/` contains aggregate outputs only: effect sizes, retrieval metrics,
resampling counts, and threshold derivations. Sampled note text, generated
queries derived from those notes, and document embeddings are **not** included,
as all three are derivative of credentialed data.

The benchmark corpora and their published queries come from the companion study
and are available in its own repository
(github.com/yngvemikkelsen/clinical-rag-retrieval-benchmark).

## Reproducing key results

### Decision model, thresholds, and value of information

```
pip install numpy scipy pandas
python model/clinical_rag_he_model_v5.py
```

Expected: p\* = 0.5805 (zero-cost) and 0.580536 (exact); partial expected value
of perfect information below €0.01 for every parameter entering M and for K;
robustness across the regularisation sweep, screening design, and prior
scenarios.

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

Expected: specificity 0.328 at ten documents rising to 0.974 at 75; net effect
negative at every sample size examined at the documented mechanistic fraction.

### Code-search boundary replication

```
git clone https://github.com/drndr/code_isotropy.git
python experiments/replication/paper9_expB_diera.py evaluate --all
```

Expected: baselines reproduce published values to within 0.001 across 18 of 18
primary cells; the sign structure does not reproduce at zero regularisation.

### Appendices, figures, and consistency check

```
python analysis/make_appendices.py
python analysis/make_figures.py
python analysis/qc_manuscript.py --md <manuscript>.md
```

`qc_manuscript.py` recomputes every load-bearing number in the manuscript from
its source artefact and fails on any mismatch or on any superseded value
reappearing. It does not check whether a claim is warranted by its evidence,
only whether the numbers agree.

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
- Mikkelsen Y. Effects of Model Choice, Corpus Context, and Post Hoc Correction on Layer-Level Embedding Degradation in Clinical Document Retrieval: Experimental Study. *JMIR Med Inform.* 2026;14:e99639.
  [Repository] (https://github.com/yngvemikkelsen/clinical-embedding-layer-analysis)

## License

Code: MIT License. Aggregate results: CC BY 4.0. No patient-level data are
included; corpus access is governed by the respective PhysioNet data use
agreements.

## Contact

Yngve Mikkelsen, MD MSc DBA

ORCID: 0000-0003-1543-3805
