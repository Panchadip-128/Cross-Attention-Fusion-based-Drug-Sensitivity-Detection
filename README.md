# Cross-Attention Fusion of Genomic and Chemical Representations for Robust Drug Sensitivity Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

Official PyTorch implementation of the paper **"Cross-Attention Fusion of Genomic and Chemical Representations for Robust Drug Sensitivity Prediction"**.

This repository provides a comprehensive, uncertainty-aware Deep Learning framework designed for robust prediction of anticancer drug sensitivity ($IC_{50}$). The model leverages pharmacogenomic data from the Genomics of Drug Sensitivity in Cancer (GDSC) databases, offering a novel architectural approach to handling highly complex, non-linear interactions between tumor genomics and chemical structures.

---

## 📌 Abstract & Architecture

Predicting the clinical efficacy of anti-cancer compounds remains a significant challenge due to the immense heterogeneity of tumor genomics and the vastness of chemical space. We propose the **Cross-Attention Drug-Genomic Fusion Model**, which directly mitigates overfitting to chemical similarities through:
1. **Dynamic Cross-Attention:** Explicitly conditioning genomic mutation and expression profiles on drug identity embeddings to learn patient-specific response mechanisms.
2. **Dual-Stream Processing:** Capturing global context via a multi-head **Transformer Encoder** and localized sequence dynamics via a **Bidirectional LSTM (BiLSTM)**.
3. **Rigorous Generalization:** Enforcing strict out-of-distribution evaluation through **Murcko Scaffold-blind splitting**.

---

## 📊 Experimental Results & Visual Diagnostics

Our framework achieves exceptional predictive accuracy on unseen chemical scaffolds while emphasizing model transparency and clinical reliability.

### 1. Training Convergence (Validation R² = 0.9958)
The dual-stream architecture ensures smooth optimization without gradient explosion. Evaluated on a strictly scaffold-blind validation set, the model reaches a peak **R² of 0.9958 at epoch 49**, demonstrating its powerful capacity to generalize to novel chemical structures rather than memorizing training compounds.

<p align="center">
  <img src="results/plots/training_curves.png" alt="Training Convergence" width="100%">
</p>

### 2. Epistemic Uncertainty Quantification
For clinical application, a model must know when it is unsure. By utilizing Monte Carlo (MC) Dropout (50 passes) during inference, we estimate the epistemic uncertainty (σ) for every prediction. 
- **Center:** We observe a strong positive correlation (slope = 0.47) between the model's uncertainty and its absolute error, proving that high variance accurately flags potentially incorrect predictions on highly novel structures.
- **Right:** The reliability diagram confirms that the empirical RMSE closely tracks the predicted standard deviation across uncertainty bins, validating the model's calibration.

<p align="center">
  <img src="results/plots/uncertainty_plots.png" alt="Uncertainty Quantification" width="100%">
</p>

### 3. Global Feature Importance via SHAP
To elucidate the biological drivers of drug sensitivity, we apply SHapley Additive exPlanations (SHAP). The analysis reveals that historical response metrics (`log_ic50_mean_pos` and `n_feature_pos`) strongly dominate global decision-making, followed closely by `Tissue Type` and `Feature Name` (specific mutations).
The Beeswarm plot shows the directional impact: high values (pink/red dots) for `log_ic50_mean_pos` systematically drive the IC$_{50}$ predictions higher (conferring resistance).

<p align="center">
  <img src="results/plots/shap_bar.png" alt="SHAP Global Importance" width="45%">
  <img src="results/plots/shap_beeswarm.png" alt="SHAP Beeswarm" width="45%">
</p>

### 4. Patient-Level Interpretability via LIME & SHAP
Beyond global trends, clinical utility demands patient-level explanations. 
The **SHAP Waterfall plot (Sample 0)** traces exactly how the prediction shifts from the expected baseline (0.253) to the final prediction (0.263), quantifying exactly how much `Tissue Type` (+0.05) and `Feature Name` (-0.05) contributed to that specific patient's shift.

<p align="center">
  <img src="results/plots/shap_waterfall.png" alt="SHAP Waterfall" width="70%">
</p>

We further validate these non-linear feature interactions at an individual level using Local Interpretable Model-agnostic Explanations (LIME). The comparison across multiple test samples demonstrates that while global features matter, the relative importance of specific tissue types and drug names dynamically shifts for every unique patient.

<p align="center">
  <img src="results/plots/lime_comparison.png" alt="LIME Comparison" width="100%">
</p>

---

## 🚀 Installation & Setup

Clone the repository and install the required dependencies. We recommend using a virtual environment.

```bash
git clone https://github.com/Panchadip-128/Cross-Attention-Fusion-based-Drug-Sensitivity-Detection.git
cd Cross-Attention-Fusion-based-Drug-Sensitivity-Detection
pip install -r requirements.txt
```

---

## 💻 Usage

### 1. Data Preparation
Ensure the `GDSC1.csv` and `GDSC2.csv` data files are placed in the root directory.

### 2. Model Training
To train the model using the Murcko scaffold-blind splitting methodology:

```bash
python scripts/train.py --epochs 200 --batch_size 8192 --lr 1e-3
```

### 3. Evaluation and Uncertainty Estimation
To evaluate the optimal saved model and execute Monte Carlo Dropout for uncertainty bounds on the test set:

```bash
python scripts/evaluate.py --model_path results/best_model.pth
```

---

## 📜 Citation

If you find this code or our methodology useful in your research, please consider citing our work:

```bibtex
@article{panchadip2026crossattention,
  title={Cross-Attention Fusion of Genomic and Chemical Representations for Robust Drug Sensitivity Prediction},
  author={Panchadip},
  journal={IEEE Access},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
