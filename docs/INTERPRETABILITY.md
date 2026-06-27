# Clinical Interpretability: SHAP & LIME 

Deep neural models in oncology must provide actionable, interpretable reasoning for their predictions. Rather than acting as a black-box oracle, the Cross-Attention Framework utilizes **SHapley Additive exPlanations (SHAP)** and **Local Interpretable Model-agnostic Explanations (LIME)** to provide multi-level biological validation.

## 1. Interpretability Pipeline Workflow

This flowchart details how we extract post-hoc actionable intelligence from the trained model to map genomic drivers directly back to underlying cancer biology.

```mermaid
flowchart TD
    classDef explain fill:#fdcb6e,stroke:#ffeaa7,stroke-width:2px,color:#2d3436;

    Model[Trained Cross-Attention Model] --> Data[Scaffold-Blind Test Set]
    Data --> SHAP[SHAP DeepExplainer]:::explain
    Data --> LIME[LIME Tabular Explainer]:::explain

    SHAP --> S_Global[Aggregate Marginal Contributions]:::explain
    S_Global --> S_Plot[Global Beeswarm / Bar Plots<br>Identify Key Genomic Drivers]

    LIME --> L_Perturb[Perturb Patient Genomic Profile]:::explain
    L_Perturb --> L_Local[Fit Local Surrogate Ridge Model]:::explain
    L_Local --> L_Plot[Patient-Specific Waterfall Plots<br>Identify Individual Resistance Factors]
```

## 2. Global Interpretability (SHAP)

SHAP values compute the marginal contribution of each genomic feature across all possible coalitions in the input space. We use `SHAP DeepExplainer` to calculate global feature attribution over the validation set.

| SHAP Global Importance Beeswarm | SHAP Feature Importance (Bar) |
| :---: | :---: |
| ![SHAP Global Importance Beeswarm](assets/shap_beeswarm.png) | ![SHAP Bar Plot](assets/shap_bar.png) |

**Biological Insights:**
- **The Beeswarm Plot (Left):** Isolates the specific genomic mutations (e.g., TP53, BRAF, KRAS) driving overarching global drug resistance across the entire cohort. Red dots on the right indicate that the presence of that mutation heavily drives resistance (higher $IC_{50}$).
- **The Bar Plot (Right):** Shows the absolute mean impact on model output across the top canonical oncogenes, regardless of directionality.

## 3. Localized Patient-Specific Explainability (LIME)

While SHAP provides global cohort insights, clinicians need to know *why* a drug was recommended for a *specific patient*. We deploy LIME to fit local surrogate ridge-regression models around a single patient's genomic profile.

| Local LIME Patient-Specific Analysis | Patient-Specific SHAP Waterfall |
| :---: | :---: |
| ![LIME Local Explanation](assets/lime_patient.png) | ![SHAP Waterfall](assets/shap_waterfall.png) |

**Clinical Utility:**
- **LIME Local Explanations (Left):** Validates that the local Cross-Attention layer correctly conditions the prediction solely on the patient's unique multi-omics perturbation profile. It shows the oncologist exactly which specific mutations in this specific patient are driving the sensitivity.
- **SHAP Waterfall (Right):** Traces the exact mathematical accumulation of a single prediction from the base-value (expected cohort mean) to the final predicted $IC_{50}$. This serves as an auditable mathematical trail for regulatory compliance.

---

[⬅ Return to Main README](../README.md)
