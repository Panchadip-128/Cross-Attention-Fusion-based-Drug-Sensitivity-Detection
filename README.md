# Cross-Attention Fusion of Genomic and Chemical Representations for Robust Drug Sensitivity Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository contains the official implementation of the paper **"Cross-Attention Fusion of Genomic and Chemical Representations for Robust Drug Sensitivity Prediction"**. 

It provides an advanced, uncertainty-aware Deep Learning framework for predicting anticancer drug sensitivity ($IC_{50}$) utilizing pharmacogenomic data from the Genomics of Drug Sensitivity in Cancer (GDSC) databases (GDSC1 and GDSC2).

---

## 📌 Abstract & Motivation

Predicting the clinical efficacy of anti-cancer compounds is notoriously difficult due to the complex, non-linear interactions between tumor genomics and drug chemistry. Traditional models often overfit on chemical similarities, failing to generalize to novel molecular structures (scaffolds). 

This repository introduces the **Cross-Attention Drug-Genomic Fusion Model**, which directly addresses these limitations by:
1. **Dynamic Cross-Attention:** Conditioning genomic mutation and expression profiles directly on drug identity representations, learning patient-specific drug response mechanisms.
2. **Dual-Stream Processing:** Capturing global context via a multi-head **Transformer Encoder** and localized sequence dynamics via a **Bidirectional LSTM** (BiLSTM).
3. **Robust Evaluation:** Evaluating exclusively via **Murcko Scaffold-blind splitting** to enforce true generalization to novel compounds.
4. **Uncertainty Quantification:** Leveraging Monte Carlo (MC) Dropout to estimate predictive confidence, which is critical for clinical decision-making.

---

## 🧠 Model Architecture

The architecture seamlessly integrates heterogeneous data:

1. **Drug Embeddings**: Trainable 64-dimensional representations of drug chemical space.
2. **Positional Encoding**: Sinusoidal encodings added to genomic features.
3. **Cross-Attention Fusion**: The genomic features act as *queries*, and the drug embedding acts as *keys* and *values*.
4. **Transformer Encoder & BiLSTM**: Processes the fused representations in parallel streams.
5. **Attention Pooling**: Replaces naive mean-pooling with a learnable weighted sum over the sequence dimension to highlight informative features.
6. **MC Dropout**: Stochastic forward passes during inference to quantify epistemic uncertainty.

*(See `src/models/architectures.py` for the core implementation).*

---

## 📊 Results & Explainability

Our framework not only predicts sensitivity with high accuracy on unseen chemical scaffolds but provides deep explainability into its decisions.

### 1. Training & Convergence
The dual-stream architecture ensures smooth and rapid convergence without exploding gradients.
![Training Curves](results/plots/training_curves.png)

### 2. Epistemic Uncertainty Quantification
By keeping MC Dropout active during inference, the model produces confidence intervals for every prediction. High variance indicates low model confidence on structurally novel out-of-distribution drugs.
![Uncertainty Plots](results/plots/uncertainty_plots.png)

### 3. SHAP Feature Importance
Using SHapley Additive exPlanations (SHAP), we interpret which genomic features (e.g., specific mutations or tissue types) drove the prediction for a given cell line.
![SHAP Waterfall](results/plots/shap_waterfall.png)
![SHAP Beeswarm](results/plots/shap_beeswarm.png)

### 4. LIME Local Explanations
Local Interpretable Model-agnostic Explanations (LIME) further validates the non-linear feature interactions at a localized patient level.
![LIME Comparison](results/plots/lime_comparison.png)
![LIME Sample](results/plots/lime_sample0.png)

---

## 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Panchadip-128/Cross-Attention-Fusion-based-Drug-Sensitivity-Detection.git
cd Cross-Attention-Fusion-based-Drug-Sensitivity-Detection
pip install -r requirements.txt
```

---

## 💻 Usage

### 1. Data Preparation
Ensure you download the `GDSC1.csv` and `GDSC2.csv` datasets and place them in the root directory (or update the paths in the CLI arguments).

### 2. Training the Model
To train the model using the Murcko scaffold-blind split with early stopping:

```bash
python scripts/train.py --epochs 200 --batch_size 8192 --lr 1e-3
```

*Note: The script automatically detects CUDA and configures PyTorch's SDPA backend for optimal mathematical stability.*

### 3. Evaluating and Quantifying Uncertainty
To evaluate the best saved model and generate MC Dropout uncertainty bounds on the test set:

```bash
python scripts/evaluate.py --model_path results/best_model.pth
```

---

## 📂 Repository Structure

```text
├── docs/                  # LaTeX Template and Mathematical Theorems (PDF/DOCX)
├── notebooks/             # Original interactive Exploratory & Training Jupyter Notebooks
├── results/               
│   └── plots/             # Saved outputs (SHAP, LIME, Training Curves, Uncertainty)
├── scripts/               # CLI Scripts for training and evaluation
│   ├── train.py
│   └── evaluate.py
├── src/                   # Core Python Package
│   ├── data/              # Loader, Preprocessor, Graph Builder, Scaffold Splitter
│   ├── models/            # Cross-Attention, Transformer, and BiLSTM Modules
│   ├── training/          # Early Stopping, AdamW Training Loop, MC Dropout Evaluator
│   └── utils/             # Seed lock and Visualization Utilities
├── requirements.txt       # Dependencies (PyTorch, PyG, RDKit, SHAP, etc.)
└── README.md
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
