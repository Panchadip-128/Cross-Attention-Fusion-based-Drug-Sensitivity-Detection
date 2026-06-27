# Cross-Attention Fusion of Genomic and Chemical Representations for Robust Drug Sensitivity Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

Official PyTorch implementation of the paper **"Cross-Attention Fusion of Genomic and Chemical Representations for Robust Drug Sensitivity Prediction"**.

This repository provides a comprehensive, uncertainty-aware Deep Learning framework designed for robust prediction of anticancer drug sensitivity ($IC_{50}$). The model leverages pharmacogenomic data from the Genomics of Drug Sensitivity in Cancer (GDSC) databases, offering a novel architectural approach to handling highly complex, non-linear interactions between tumor genomics and chemical structures.

---

## 📚 Documentation Hub

To maintain a clean and professional repository, our comprehensive documentation is organized into the following specialized modules:

- 🔬 **[Exploratory Data Analysis (EDA)](docs/EDA.md):** Detailed breakdown of the GDSC datasets, feature spaces, and our Murcko scaffold-blind splitting strategy.
- 🧠 **[Model Architecture](docs/ARCHITECTURE.md):** The rigorous mathematical formulation of the Dual-Stream Cross-Attention, Transformer, BiLSTM, and Attention Pooling layers.
- 📊 **[Experimental Results & Interpretability](docs/RESULTS.md):** Deep dive into the model's performance, including LIME/SHAP local explanations and MC Dropout uncertainty quantification.
- 🖼️ **[Supplementary Figures Gallery](docs/SUPPLEMENTARY_FIGURES.md):** A complete, categorized archive of 114+ intermediate visualizations and diagnostic plots extracted directly from the research notebooks.

---

## 🚀 Quick Start

Clone the repository and install the required dependencies. We recommend using a virtual environment (e.g., `conda` or `venv`).

```bash
git clone https://github.com/Panchadip-128/Cross-Attention-Fusion-based-Drug-Sensitivity-Detection.git
cd Cross-Attention-Fusion-based-Drug-Sensitivity-Detection
pip install -r requirements.txt
```

### Model Training
To train the model using the Murcko scaffold-blind splitting methodology:
```bash
python scripts/train.py --epochs 200 --batch_size 8192 --lr 1e-3
```

### Evaluation and Uncertainty Estimation
To evaluate the optimal saved model and execute Monte Carlo Dropout for uncertainty bounds on the test set:
```bash
python scripts/evaluate.py --model_path results/best_model.pth
```

### Running Tests
To ensure absolute stability, run the automated `pytest` suite:
```bash
pytest tests/ -v
```

---

## 📂 Repository Structure

```text
├── docs/                  # Detailed documentation (EDA, Architecture, Results, Theorems)
├── notebooks/             # Exploratory Data Analysis and Training Jupyter Notebooks
├── results/               
│   └── plots/             # Diagnostic plots, SHAP, LIME, and Uncertainty visualizations
├── scripts/               # Command-Line Interfaces (CLI) for training and evaluation
│   ├── train.py
│   └── evaluate.py
├── src/                   # Core Python Package
│   ├── data/              # Data ingestion, preprocessing, and RDKit Murcko Scaffold splitting
│   ├── models/            # Architectural components (Cross-Attention, BiLSTM, Transformer)
│   ├── training/          # AdamW optimization, Cosine Annealing, and MC Dropout evaluation
│   └── utils/             # Deterministic seed locking and visualization utilities
├── tests/                 # Automated unit and integration testing suite
├── .github/workflows/     # CI/CD pipelines
├── requirements.txt       # Package dependencies
└── README.md
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
