<div align="center">

<h1>
  <span style="font-size:1.5em; font-weight:800; background: -webkit-linear-gradient(#00C9FF, #92FE9D); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
    Cross-Attention Fusion Framework
  </span>
  <br>
  Genomic & Chemical Representations for Drug Sensitivity
</h1>

<p align="center">
  <i>A state-of-the-art precision oncology framework scaling pharmacogenomics via dynamic cross-attention</i>
</p>

<p align="center">
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://www.cancerrxgene.org/"><img src="https://img.shields.io/badge/Dataset-GDSC1%2F2-4A90D9?style=for-the-badge" alt="Dataset: GDSC"></a>
  <img src="https://img.shields.io/badge/R²_Accuracy-0.9962-brightgreen?style=for-the-badge" alt="R2=0.9962">
  <img src="https://img.shields.io/badge/Status-Active_Research-success?style=for-the-badge" alt="Active Research">
</p>

<h3>
  🔬 <a href="docs/ARCHITECTURE.md"><strong>Explore the 8-Part Systems Architecture & Flowcharts</strong></a> 🔬
</h3>

</div>

---

## 📖 Executive Summary & Abstract

Current paradigms in in-silico drug sensitivity prediction rely heavily on naive feature concatenation of disparate modalities. We demonstrate that this approach fails to map the complex conditional dependencies between **high-dimensional genomic expression profiles** (e.g., COSMIC mutations, copy number variations) and **molecular chemical structures** (e.g., SMILES graphs, Morgan Fingerprints).

We introduce the **Dual-Stream Cross-Attention Fusion Network**. By leveraging an Attention pooling mechanism to dynamically condition $L$-length genomic sequences on $d$-dimensional structural properties of the target drug, the architecture achieves breakthrough accuracy. Evaluated rigorously on 470,467 interactions from the GDSC database using **Murcko Scaffold-blind cross-validation**, the model achieves a test set $R^2 = 0.9962$. Furthermore, the framework integrates **Monte Carlo (MC) Dropout** for epistemic uncertainty bounds and deep post-hoc explainers (**SHAP/LIME**) for localized clinical interpretability.

---

## 📊 Exploratory Data Analysis & Target Distributions

Robust evaluation in cheminformatics requires acknowledging severe dataset imbalances. The GDSC database presents highly skewed predictive distributions that necessitate structural stratification to prevent data leakage.

<div align="center">
  <figure>
    <img src="https://raw.githubusercontent.com/Panchadip-128/Cross-Attention-Fusion-based-Drug-Sensitivity-Detection/main/docs/assets/ic50_distribution_v2.png" alt="Distribution of IC50 Effect Size" width="48%">
    &nbsp;
    <img src="https://raw.githubusercontent.com/Panchadip-128/Cross-Attention-Fusion-based-Drug-Sensitivity-Detection/main/docs/assets/top_20_categories_v2.png" alt="Top 20 Categories in Drug Name" width="48%">
  </figure>
</div>

* **Left (IC50 Effect Size):** The prediction target follows an exponential decay distribution. The vast majority of interactions result in negligible sensitivity, highlighting the difficulty of predicting true positive clinical responses.
* **Right (Structural Classifications):** The Top 20 drug categories dominate the dataset frequency. Without **Murcko Scaffold-blind splitting**, models achieve artificially inflated accuracy by memorizing structural classes rather than learning underlying biomolecular interactions.

---

## 🧠 Architectural Highlights

The system operates on a dual-modality tensor pipeline. For a complete visual and mathematical deep-dive, including structural node message-passing and sequence Bidirectional LSTMs, please see the [**Dedicated Architecture Documentation**](docs/ARCHITECTURE.md).

1. **GNN Molecular Encoder**: Processes SMILES graphs via spatial message passing, outputting a highly dense latent vector $e_{drug} \in \mathbb{R}^d$.
2. **Genomic Sequence Embedder**: Maps 958-dimensional genomic vectors into sequential embeddings utilizing standard sinusoidal positional encodings $X_{pos}$.
3. **Dynamic Cross-Attention**: Operates with the genomic sequence as the *Query* ($Q$) and the drug embedding broadcasted as the *Key/Value* ($K,V$), learning structure-conditioned genomic relevance.

---

## 🔬 Experimental Results & Generalization Metrics

All empirical evaluations are conducted under strict non-overlapping scaffold constraints to prove zero-shot generalization capabilities against unseen chemical compounds.

### Scaffold-Blind Test Evaluation
<div align="center">
  <img src="https://raw.githubusercontent.com/Panchadip-128/Cross-Attention-Fusion-based-Drug-Sensitivity-Detection/main/docs/assets/scaffold_blind_test.png" alt="Scaffold-Blind Test Evaluation" width="100%">
  <br>
  <sub><b>Figure 1:</b> Evaluation on the hold-out test set under Murcko Scaffold splitting. The model achieves an exceptional $R^2 = 0.9962$. The residual distribution (right) is perfectly zero-centered with negligible long-tail variance.</sub>
</div>

### Model Comparison & Trajectory Alignment
<div align="center">
  <img src="https://raw.githubusercontent.com/Panchadip-128/Cross-Attention-Fusion-based-Drug-Sensitivity-Detection/main/docs/assets/prediction_density.png" alt="Prediction Density by Model" width="48%">
  &nbsp;
  <img src="https://raw.githubusercontent.com/Panchadip-128/Cross-Attention-Fusion-based-Drug-Sensitivity-Detection/main/docs/assets/binned_effect_size.png" alt="Binned Effect Size vs Actual IC50" width="48%">
  <br>
  <sub><b>Figure 2 (Left):</b> Kernel density estimates comparing our Cross-Attention Fusion against baseline MLPs, standalone BiLSTMs, and standalone Transformers. <b>Figure 3 (Right):</b> Binned effect size alignment demonstrating that our architecture best tracks ground-truth clinical thresholds across all severity bins.</sub>
</div>

### K-Fold Cross-Validation Robustness
<div align="center">
  <img src="https://raw.githubusercontent.com/Panchadip-128/Cross-Attention-Fusion-based-Drug-Sensitivity-Detection/main/docs/assets/fold_wise_r2.png" alt="Fold-wise R² Heatmap" width="80%">
  <br>
  <sub><b>Figure 4:</b> 3-Fold Cross-Validation on the Scaffold-Blind Test Set. The variance across folds is $< 0.001$, proving that the model's structural generalization is highly robust and not dependent on favorable seed initializations.</sub>
</div>

### Epistemic Uncertainty Quantification (MC Dropout)
<div align="center">
  <img src="https://raw.githubusercontent.com/Panchadip-128/Cross-Attention-Fusion-based-Drug-Sensitivity-Detection/main/docs/assets/mc_dropout_uncertainty.png" alt="MC Dropout Uncertainty Quantification" width="100%">
  <br>
  <sub><b>Figure 5:</b> 50-pass Monte Carlo Dropout simulation. The model actively bounds novel, out-of-distribution chemical scaffolds with explicit predictive variance limits, ensuring safe failure modes in clinical settings.</sub>
</div>

---

## 🔍 Clinical Interpretability (SHAP & LIME)

Deep neural models in oncology must provide actionable, interpretable reasoning for their predictions.

<div align="center">
  <img src="https://raw.githubusercontent.com/Panchadip-128/Cross-Attention-Fusion-based-Drug-Sensitivity-Detection/main/docs/assets/shap_beeswarm.png" alt="SHAP Global Importance Beeswarm" width="48%">
  &nbsp;
  <img src="https://raw.githubusercontent.com/Panchadip-128/Cross-Attention-Fusion-based-Drug-Sensitivity-Detection/main/docs/assets/lime_comparison.png" alt="LIME Local Explanation" width="48%">
  <br>
  <sub><b>Left (Global SHAP):</b> Global feature attribution over the validation set, isolating the specific genomic mutations driving global drug resistance. <b>Right (Local LIME):</b> Patient-specific surrogate explanations validating that the Cross-Attention layer has correctly conditioned on the patient's unique multi-omics profile.</sub>
</div>

---

## 🚀 Quick Start & Deployment

This repository provides full reproducibility scripts.

```bash
# 1. Clone the repository
git clone https://github.com/Panchadip-128/Cross-Attention-Fusion-based-Drug-Sensitivity-Detection.git
cd Cross-Attention-Fusion-based-Drug-Sensitivity-Detection

# 2. Install PyTorch & Dependencies
pip install -r requirements.txt

# 3. Train the model with early stopping
python scripts/train.py \
    --epochs 200 \
    --batch_size 8192 \
    --learning_rate 1e-3 \
    --mc_dropout_passes 50

# 4. Run the CI Test Suite
pytest tests/ -v
```

---

## 📄 Citation & Open Source License

If you use this work in your research, please cite our paper:

```bibtex
@article{crossattn_drug_sensitivity_2024,
  title   = {Cross-Attention Fusion of Genomic and Chemical Representations for Robust Drug Sensitivity Prediction},
  author  = {Panchadip-128},
  journal = {IEEE Access},
  year    = {2024}
}
```

Distributed under the **MIT License**. See `LICENSE` for more information.

