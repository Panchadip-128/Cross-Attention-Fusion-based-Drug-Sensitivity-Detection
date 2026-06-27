# Exploratory Data Analysis & Data Pipeline

This document details the datasets, feature spaces, and rigorous validation strategies utilized in this repository.

## 1. Genomics of Drug Sensitivity in Cancer (GDSC)

Our framework trains on pharmacogenomic data aggregated from the GDSC databases (GDSC1 and GDSC2). These databases provide extensive cancer cell line screening data.

### Feature Spaces
To model the non-linear interaction between biology and chemistry, we utilize two distinct feature modalities:

1. **Genomic Features (Cell Lines):**
   - We extract mutation profiles and gene expression signatures for various cancer cell lines.
   - Categorical metadata, such as `Tissue Type`, is also integrated as contextual information for the model.

2. **Chemical Features (Drugs):**
   - Drugs are represented structurally. In our data pipeline (`src/data/graph.py`), we utilize RDKit to parse SMILES strings into graph representations.
   - These raw chemical identifiers are mapped to a continuous, trainable $d$-dimensional embedding space during model training.

## 2. Murcko Scaffold-Blind Splitting

Standard random cross-validation is highly prone to data leakage in drug discovery. If a model trains on one derivative of a drug, it can easily "memorize" the chemical backbone and predict the efficacy of a highly similar derivative during testing.

To simulate real-world clinical utility where the model must predict efficacy for **novel, unseen compounds**, we implement **Murcko Scaffold-blind splitting**.

### The Splitting Logic (`src/data/split.py`)
1. **Scaffold Extraction:** For every drug in the dataset, we extract its Murcko Scaffold (the core ring structure of the molecule) using RDKit.
2. **Stratification:** We split the dataset not by individual drugs, but by these core scaffolds. 
3. **Out-of-Distribution Validation:** The validation and test sets contain chemical scaffolds that are *entirely absent* from the training set.

This rigorous validation strategy guarantees that our model's performance metrics reflect true generalization rather than memorization.
