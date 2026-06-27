# Cross-Attention Fusion of Genomic and Chemical Representations for Robust Drug Sensitivity Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

Official repository for **"Cross-Attention Fusion of Genomic and Chemical Representations for Robust Drug Sensitivity Prediction"**. 

This research proposes a highly interpretative, uncertainty-aware Deep Learning framework. By fusing pharmacogenomic features (GDSC databases) with SMILES-derived chemical graphs, the architecture decodes the non-linear interaction between a patient's tumor biology and an anticancer drug's structural chemistry.

---

## 🔬 1. The Architecture: Fusing Genomics and Chemistry

<p align="center">
  <img src="docs/paper_figures/architecture.jpg" alt="Deep Learning Pipeline Architecture" width="100%">
</p>

Our framework abandons simple feature concatenation in favor of **Dynamic Cross-Attention**. 
- **Genomic Profiles** (Gene expression and mutations) act as the *Query*.
- **Chemical Embeddings** (RDKit-processed drug representations) act as the *Key* and *Value*.

This forces the model to actively "attend" only to the specific genetic markers that are biologically relevant to the input drug's unique chemical structure. The output is processed by dual Transformer and BiLSTM streams to capture both global context and localized genomic sequences.

---

## 📈 2. Robust Training and Validation

<p align="center">
  <img src="docs/paper_figures/training_curves.png" alt="Training Convergence Curves" width="100%">
</p>

To guarantee clinical realism, the model is trained using **Murcko Scaffold-blind splitting**. This ensures no chemical derivatives of the test set exist in the training set, explicitly preventing data leakage.
- **Superior Convergence:** As shown above, the Dual-Stream Cross-Attention mechanism enables rapid optimization, achieving an exceptional **Validation R² of 0.9958** at epoch 49 before triggering early stopping.

---

## 🎯 3. Epistemic Uncertainty Quantification

<p align="center">
  <img src="docs/paper_figures/uncertainty_plots.png" alt="Monte Carlo Dropout Uncertainty Analysis" width="100%">
</p>

A clinical model must know when it is guessing. We apply **Monte Carlo (MC) Dropout** (50 inference passes) to calculate predictive variance for unseen compounds.
- **Error Correlation:** As absolute error increases, the model's self-reported uncertainty symmetrically increases (slope = 0.47).
- **Calibration:** The model reliably flags novel, out-of-distribution chemical structures with high epistemic uncertainty, allowing oncologists to defer to clinical judgment when the AI is unsure.

---

## 🧠 4. Global Biomarker Discovery (SHAP)

<p align="center">
  <img src="docs/paper_figures/shap_bar.png" alt="SHAP Global Bar Chart" width="45%">
  <img src="docs/paper_figures/shap_beeswarm.png" alt="SHAP Global Beeswarm" width="45%">
</p>

By integrating SHapley Additive exPlanations, the "black box" is rendered fully transparent. 
- **The Bar Chart** proves that historical sensitivity metrics (`log_ic50_mean_pos`) and `Tissue Type` are the dominant global drivers of drug resistance.
- **The Beeswarm Plot** maps the directional impact: High values of specific genomic features systematically drive the predicted $IC_{50}$ higher, directly aligning with established oncological resistance pathways.

---

## 👤 5. Patient-Level Interpretability (LIME & SHAP)

<p align="center">
  <img src="docs/paper_figures/shap_waterfall.png" alt="Patient SHAP Waterfall" width="70%">
</p>

<p align="center">
  <img src="docs/paper_figures/lime_comparison.png" alt="LIME Patient Comparison" width="100%">
</p>

For individual patient cases, the model provides localized reasoning for every prediction:
- **Waterfall Analysis:** Traces the exact mathematical shift from the baseline expected response ($0.253$) to the specific patient prediction ($0.263$), explicitly quantifying how much weight was placed on the `Tissue Type` (+0.05).
- **Dynamic Interaction (LIME):** The LIME comparison explicitly proves that the relative importance of biological features dynamically shifts depending on the *exact drug* the patient is receiving, successfully validating the efficacy of the Cross-Attention layer.

---

## 📚 Deep Dive Documentation

For extensive technical specifics, please refer to our dedicated documentation modules:
- 📊 **[Extracted Notebook Plots](docs/SUPPLEMENTARY_FIGURES.md):** 114+ intermediate visualizations extracted directly from the research notebooks.
- 🔬 **[Exploratory Data Analysis (EDA)](docs/EDA.md):** GDSC dataset breakdown and Murcko splitting logic.
- 🧠 **[Mathematical Architecture](docs/ARCHITECTURE.md):** Rigorous formulation of the Dual-Stream Cross-Attention and Attention Pooling.

---

## 🚀 Quick Start & Installation

```bash
git clone https://github.com/Panchadip-128/Cross-Attention-Fusion-based-Drug-Sensitivity-Detection.git
cd Cross-Attention-Fusion-based-Drug-Sensitivity-Detection
pip install -r requirements.txt
```

**Run Training:**
```bash
python scripts/train.py --epochs 200 --batch_size 8192 --lr 1e-3
```

**Run Automated Testing (PyTest):**
```bash
pytest tests/ -v
```
