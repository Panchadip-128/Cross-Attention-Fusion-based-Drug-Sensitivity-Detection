# Cross-Attention Fusion of Genomic and Chemical Representations for Robust Drug Sensitivity Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

Official repository for **"Cross-Attention Fusion of Genomic and Chemical Representations for Robust Drug Sensitivity Prediction"**. 

This research proposes a highly interpretative, uncertainty-aware Deep Learning framework. By fusing pharmacogenomic features (GDSC databases) with SMILES-derived chemical graphs, the architecture decodes the non-linear interaction between a patient's tumor biology and an anticancer drug's structural chemistry.

---

## 🔬 Methodology

### 1. Cross-Attention Architecture
<p align="center">
  <img src="docs/paper_figures/architecture.jpg" alt="Deep Learning Pipeline Architecture" width="90%">
</p>
Our framework utilizes **Dynamic Cross-Attention** to fuse genomic profiles (Query) with chemical graph embeddings (Key/Value). This explicitly forces the model to attend to genetic markers that are biologically relevant to the specific input drug, processed via dual Transformer and BiLSTM streams.

### 2. Murcko Scaffold-Blind Splitting
To simulate true clinical utility and prevent chemical data leakage, we utilize rigorous Murcko Scaffold-blind splitting. The model is evaluated on chemical scaffolds it has *never* seen during training.

---

## 📈 Key Results

### 1. Robust Predictive Performance
<p align="center">
  <img src="docs/paper_figures/training_curves.png" alt="Training Convergence" width="80%">
</p>
Despite the strict out-of-distribution validation, the dual-stream architecture achieves rapid convergence, reaching a peak **Validation R² of 0.9958**.

### 2. Global & Local Interpretability (SHAP & LIME)
<table>
  <tr>
    <td align="center"><b>Global Biomarker Discovery (SHAP)</b></td>
    <td align="center"><b>Patient-Level Precision (LIME)</b></td>
  </tr>
  <tr>
    <td><img src="docs/paper_figures/shap_beeswarm.png" alt="SHAP Beeswarm" width="100%"></td>
    <td><img src="docs/paper_figures/lime_comparison.png" alt="LIME Comparison" width="100%"></td>
  </tr>
  <tr>
    <td>Identifies `log_ic50_mean_pos` and `Tissue Type` as the absolute dominant drivers of drug resistance across the entire cohort.</td>
    <td>Proves the Cross-Attention mechanism dynamically shifts feature importance for every unique patient-drug interaction.</td>
  </tr>
</table>

### 3. Epistemic Uncertainty Quantification
<p align="center">
  <img src="docs/paper_figures/uncertainty_plots.png" alt="MC Dropout Uncertainty" width="80%">
</p>
Using **Monte Carlo (MC) Dropout**, the model reliably flags novel, out-of-distribution structures. The strong correlation between predictive variance and absolute error ensures the model knows when it is guessing.

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
