# Learning Algorithms for Predicting Postoperative Adverse Outcomes in Cardiac Procedures

## Overview
This project, centered on the `NCR_code.ipynb` notebook, implements an **interpretable machine learning pipeline** to study adverse surgical outcomes. By analyzing dynamic intraoperative signals from **2,747 cardiothoracic surgeries** (January 2011 – December 2020) in the **INSPIRE** dataset, we identify surgical trajectory phenotypes and predict postoperative complications such as extended ICU stays and in-hospital mortality.

## Project Scientific Framework
The methodology focuses on reframing intraoperative physiology as a set of dynamic trajectories that encode hidden risk phenotypes. Key innovation includes the direct **coupling of vasopressor dosing with hemodynamic response (MAP)**, allowing for the extraction of mechanistically grounded risk profiles.

### Key Methodology
- **Unsupervised Phenotyping**: Silhouette-guided K-means clustering (k=2) revealed two stable intraoperative archetypes: **High-risk** and **Low-risk**.
- **Drug-Hemodynamic Coupling**: Advanced feature engineering to capture:
    - **Lag**: Time delay between drug administration and BP response.
    - **Dose-Response Slope**: Magnitude of BP change per unit drug.
    - **Responsiveness Index**: Mean ΔMAP/Δdose across all increments.
- **Supervised Learning**: 100-tree Random Forest models used for predicting binary outcomes (Extended ICU Stay, Mortality).

## Detailed Results

### Surgical Trajectory Phenotypes
Clustering stratified patients into two distinct courses with striking differences in postoperative outcomes (n = 2,747):

| Complication / Endpoint | Low-risk (n=520) | High-risk (n=2,227) | Odds Ratio | p-value |
| :--- | :--- | :--- | :--- | :--- |
| **Acute Kidney Injury (AKI)** | 4.2% | 25.4% | 7.7 | 3.4 x 10⁻²⁶ |
| **Sustained Hypotension** | 3.9% | 27.4% | 9.4 | 1.6 x 10⁻³⁰ |
| **Metabolic Acidosis** | 36.7% | 75.1% | 5.2 | 9.8 x 10⁻⁶⁴ |
| **Respiratory Depression** | 7.7% | 54.6% | 14.4 | 2.8 x 10⁻⁸³ |
| **Extended ICU Stay (>3 days)** | 8.1% | 31.7% | 5.3 | 2.2 x 10⁻²⁷ |
| **In-hospital Mortality** | 2.1% | 5.1% | 1.8 | 0.0049 |

### Supervised Prediction Performance
The models demonstrated clinically actionable accuracy for predicting adverse clinical endpoints:

| Model | AUROC | Balanced Accuracy | Brier Score (Uncalibrated) |
| :--- | :--- | :--- | :--- |
| **Extended ICU Stay** | 0.77 | 0.78 | 0.17 |
| **In-hospital Mortality** | 0.87 | 0.63 | 0.043 |

### Mechanistic Insights (Interpretability)
-   **Autonomic Variability**: Protective effects were observed with "optimal" variabilities in **Heart Rate** and **Body Temperature**, likely indicating intact sympathetic-parasympathetic afferent-efferent communication.
-   **Dobutamine "Valley"**: Partial dependence analysis for mortality revealed a minimum risk "valley" for dobutamine dosing between **26.3 - 47.4 µg/kg** (approx. 2.5 - 5.0 µg/kg/min).
-   **Myocardial Responsiveness**: The **Dobutamine response lag** emerged as a top feature for mortality, suggesting that physiological latency is a biomarker for cardiac reserve.

## Financial Impact
Enhanced triaging could lead to significant resource conservation. For this cohort (n = 2,747), accurate identification of at-risk patients translates to potential cost avoidance of **~$726,710** based on ICU length of stay differences.

## Requirements & Data Dependencies
Requires standard scientific Python stack (`pandas`, `numpy`, `scipy`, `scikit-learn`, `imblearn`, `statsmodels`).
Input files from INSPIRE v1.3:
-   `patient_timeline_events_labeled.csv`, `parameters.csv`, `schema.csv`, and extracted clinical datasets (`vitals`, `labs`, `medications`, `operations`).

> [!IMPORTANT]
> **Total Dose Escalation**: Notebook logic multiplies all total dose measures by **5** to adjust for the 5-minute data resolution (correcting µg/kg/min rates to actual total dose).

## Authors
**Chanseo Lee** et al., Yale School of Medicine.
Based on the manuscript: *Learning algorithms harness dynamic intraoperative signals to predict postoperative adverse outcomes in cardiac procedures under anesthesia.*
