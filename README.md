# Cross-Attention Fusion of Genomic and Chemical Representations for Robust Drug Sensitivity Prediction

This repository contains the official implementation of the paper **"Cross-Attention Fusion of Genomic and Chemical Representations for Robust Drug Sensitivity Prediction"**. 

It provides an uncertainty-aware framework for predicting anticancer drug sensitivity using a massive pharmacogenomic dataset derived from the GDSC database.

## Architecture

The core of the repository is the **Cross-Attention Drug-Genomic Fusion Model**. It integrates:
1. **Drug Embeddings**: Trainable 64-dimensional representations of drug identity.
2. **Cross-Attention Fusion**: Conditions genomic features dynamically based on the drug context.
3. **Dual-Stream Processing**: A parallel Transformer Encoder and BiLSTM network to capture global and bidirectional sequential dependencies.
4. **Attention Pooling**: A learnable weighted sum over the sequence dimension to highlight informative features.
5. **MC Dropout**: For explicit uncertainty quantification.

## Repository Structure

```
├── data/                  # Place raw GDSC1.csv and GDSC2.csv here
├── notebooks/             # Exploratory analysis and tutorials
├── scripts/               # Training and evaluation CLI scripts
│   ├── train.py
│   └── evaluate.py
├── src/                   # Core Python package
│   ├── data/              # Data loading, preprocessing, scaffold splitting, PyG graph creation
│   ├── models/            # Model architectures and components
│   ├── training/          # Training loops and evaluation logic
│   └── utils/             # Reproducibility and diagnostic plotting
├── requirements.txt       # Python dependencies
└── README.md
```

## Installation

```bash
git clone https://github.com/yourusername/CrossAttentionGDSC.git
cd CrossAttentionGDSC
pip install -r requirements.txt
```

## Usage

### Training

To train the model from scratch, ensure you have placed `GDSC1.csv` and `GDSC2.csv` in the parent directory (or the `data/raw/` directory and update paths), then run:

```bash
python scripts/train.py --epochs 200 --batch_size 8192
```

### Evaluation

To evaluate a saved model and compute MC Dropout uncertainty:

```bash
python scripts/evaluate.py --model_path results/best_model.pth
```

## Dataset and Splitting

The model is evaluated using a rigorous **partial scaffold-blind split** using Murcko scaffolds derived from RDKit. This ensures that the test set evaluates structural generalization rather than simple chemical memorization.

## License

This project is licensed under the MIT License.
