# INSPIRE Cardiac-Surgery ML — Analysis Code (most-updated)

Consolidated, most-current analysis code supporting the manuscript
**"Intraoperative physiologic trajectory phenotypes and operative-period prediction of
adverse outcomes after cardiac surgery"** (`Lee_CardiacML_MainText_Edited.docx`) and its
supplement (`clustering_supplementaryfigures_v2.docx`).

Only **code** is included here — no patient data or intermediate spreadsheets. See
**Data availability** below.

---

## Data availability
All analyses use the publicly available **INSPIRE** dataset (v1.3) on PhysioNet:
https://physionet.org/content/inspire/1.3/ (PhysioNet Credentialed Health Data License).
Scripts expect the derived CSVs (e.g., `model_combined_dataset.csv`,
`model_combined_dataset_clustered.csv`, `dynamic_features_vitals.csv`) in the working
directory; these are produced by the data-pipeline notebooks from the raw INSPIRE tables.

---

## Folder layout & mapping to the manuscript

### `01_data_pipeline/` — raw INSPIRE → modeling tables
| File | What it does |
|------|--------------|
| `timeline_builder.ipynb` | Builds the per-patient intraoperative timeline and dynamic vital-sign / drug features (operation start→end only; outcomes and postoperative data excluded from predictors — the leakage safeguard described in Methods). |
| `postward_vitals_combine.py` | Concatenates the sharded post-op ward-vitals CSVs used only by the complications analysis. |

### `02_main_analysis/` — primary results (Table 1, Table 2, Figures 1–3)
| File | Manuscript output |
|------|-------------------|
| `clustering_and_prediction_NCR.ipynb` | Core pipeline: PCA → k-means (k=2) phenotypes; Random-Forest prediction of extended ICU stay and in-hospital mortality. Produces **Table 2**, cluster figures, and feature pipeline (412→292→224 features). |
| `postoperative_complications_and_ORs.ipynb` | Post-op complication definitions (AKI/KDIGO, sustained hypotension, metabolic acidosis, hyperlactatemia, respiratory depression), cluster χ² tests and **odds ratios (Table 1 / Figure 1 legend)**. |
| `figure1_pipeline_schematic.py` | Renders the **Figure 1** study/pipeline schematic (vector PDF/PNG). |

### `03_revision_analyses/` — revision additions (Supplementary Tables/Figures)
| File | Addresses | Supplement item |
|------|-----------|-----------------|
| `01_missing_data_MICE_TableS3.ipynb` | Missingness patterns + MCAR/MAR/MNAR classification (Sections 1–2, used as-is). | text/Methods |
| `01b_regenerate_TableS3_canonical.py` | **Canonical MICE sensitivity** — reproduces the S5 feature pipeline/split and the correct outcome (extended ICU = `icu_los_min ≥ 4320`), then MICE-imputes vital columns. **Supersedes Section 3 of the notebook** (see note). | **Table S3** |
| `02_clustering_robustness_GMM_PCA_FigS1.ipynb` | GMM vs k-means, PC-sensitivity, PCA loadings. | **Figure S1**, PCA loading table |
| `03_temporal_split_and_leakage_audit.ipynb` | Chronological 70/30 split; temporal-window correlation / leakage audit. | **Table S5** (temporal rows) |
| `04_calibration_bootstrap_and_DCA_TableS4.ipynb` | Bootstrap calibration CIs (slope/intercept/ECE), reliability diagrams, decision-curve / net-benefit analysis. | **Table S4**, Figure 2 / net benefit |
| `05_model_comparison_temporal_TableS5.ipynb` | RF vs L2-logistic vs gradient boosting; DeLong CIs; repeated holdout. | **Table S5** |
| `05b_regenerate_TableS5.py` | Rebuilds the model-comparison CSV. | **Table S5** |
| `06_interpretability_SHAP_ICE_FigS2S3.ipynb` | SHAP summary/beeswarm and ICE/partial-dependence curves. | **Figures S2–S3** |
| `07_confounding_complexity_proxy.ipynb` | Pre-operative complexity-proxy model (ASA, age, emergency status; AUROC 0.748 for mortality) vs intraoperative RF. | Results (incremental value) |
| `09_leakage_safeguard_audit.ipynb` | Confirms features are derived only from the intraoperative window. | Methods (leakage safeguard) |
| `10_thresholds_and_brier_skill_TableS6.ipynb` | Threshold-specific sensitivity/specificity/PPV/NPV and Brier Skill Score. | **Table S6**, BSS |

---

## Note on Table S3 (important)
The original `01_missing_data_MICE_TableS3.ipynb` Section 3 built its baseline on a
different feature set and used `icu_admit` (any ICU admission) as the ICU outcome, which
produced an inflated ICU AUROC (0.903) inconsistent with the rest of the paper.
**`01b_regenerate_TableS3_canonical.py` is the corrected, canonical version**: it uses the
identical feature pipeline, split (seed 42), and outcome definition (extended ICU stay =
`icu_los_min ≥ 4320 min`) as `05_model_comparison_temporal_TableS5.ipynb`. Its "Primary
pipeline" baseline reproduces the headline model (mortality AUROC 0.866, extended-ICU
AUROC 0.774) and MICE changes discrimination by ≤0.02, confirming robustness. Use `01b`
for the published Table S3. Sections 1–2 of the notebook (missingness patterns and
mechanism classification) remain valid and are unaffected.

---

## Environment
Developed with Python 3.13 (Anaconda `inspire` env). Install with:

```bash
pip install -r requirements.txt
```

Run notebooks from a directory containing the derived INSPIRE CSVs, or point the path
setup at your INSPIRE root. Scripts are deterministic (`random_state = 42`).
