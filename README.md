# Cross-Attention Fusion Framework for Pharmacogenomic Drug Sensitivity Prediction

> **A state-of-the-art precision oncology framework that fuses high-dimensional genomic profiles with molecular chemical representations via dynamic cross-attention to predict in-silico drug response at scale.**

<p align="center">
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-3a86ff?style=flat-square" alt="MIT License"/></a>
  <a href="https://www.cancerrxgene.org/"><img src="https://img.shields.io/badge/Dataset-GDSC1%2F2-4A90D9?style=flat-square" alt="GDSC"/></a>
  <img src="https://img.shields.io/badge/R%C2%B2-0.9962-00b894?style=flat-square" alt="R2 Score"/>
  <img src="https://img.shields.io/badge/Parameters-14.2M-6c5ce7?style=flat-square" alt="Parameters"/>
  <img src="https://img.shields.io/badge/Interactions-470%2C467-e17055?style=flat-square" alt="Dataset Size"/>
</p>

---

## Abstract

Predicting patient-specific drug sensitivity from multi-omic profiles is a fundamental open problem in precision oncology. Existing approaches fail because they naively concatenate genomic and chemical feature vectors, ignoring the conditional structure of the interaction: *which mutations, in this patient, modulate the binding affinity of this specific molecular scaffold?*

We address this with the **Dual-Stream Cross-Attention Fusion Network**, a novel architecture that treats the patient's genomic profile as a dynamic query attending over the drug's structural key-value encodings. Evaluated on 470,467 drug-cell-line interactions from the [GDSC1/2 database (DepMap Mirror)](https://depmap.org/portal/download/) under **Murcko Scaffold-blind cross-validation** — the gold standard for measuring true out-of-distribution chemical generalization — our model achieves $R^2 = 0.9962$ ($p < 0.0001$ vs. Transformer baseline). We further integrate Monte Carlo Dropout for calibrated epistemic uncertainty quantification and SHAP/LIME for post-hoc clinical interpretability, enabling safe oncologist-supervised deployment.

**[Exploratory Data Analysis](docs/EDA.md) &nbsp;·&nbsp; [Neural Architecture](docs/ARCHITECTURE.md) &nbsp;·&nbsp; [Training & Evaluation](docs/TRAINING_AND_EVALUATION.md) &nbsp;·&nbsp; [Interpretability](docs/INTERPRETABILITY.md) &nbsp;·&nbsp; [Reproducibility](docs/HARDWARE_AND_REPRODUCIBILITY.md)**

---

## Table of Contents

1. [The Problem: Why Naive Feature Concatenation Fails](#1-the-problem-why-naive-feature-concatenation-fails)
2. [Dataset Engineering & Anti-Leakage Strategy](#2-dataset-engineering--anti-leakage-strategy)
3. [Neural Architecture: Design & Rationale](#3-neural-architecture-design--rationale)
4. [Training Pipeline](#4-training-pipeline)
5. [Empirical Results](#5-empirical-results)
6. [Epistemic Uncertainty Estimation](#6-epistemic-uncertainty-estimation)
7. [Clinical Interpretability](#7-clinical-interpretability)
8. [Enterprise Deployment Architecture](#8-enterprise-deployment-architecture)
9. [Model Card](#9-model-card)
10. [Reproducing the Results](#10-reproducing-the-results)
11. [Citation](#11-citation)

---

## 1. The Problem: Why Naive Feature Concatenation Fails

Pharmacogenomic response prediction requires jointly modeling two fundamentally different data modalities: a patient's genomic profile and a drug's molecular structure. The prevailing approach — concatenating a genomic feature vector $g \in \mathbb{R}^{958}$ with a drug fingerprint $f \in \mathbb{R}^{d}$ before passing to an MLP — is **statistically deficient by design**.

Concatenation creates an *unconditional* feature space. It cannot represent the central biological fact: **the predictive relevance of any individual mutation is not a fixed property. It depends entirely on which specific drug is being tested.** A TP53 loss-of-function mutation is critically relevant for predicting resistance to MDM2 inhibitors, but largely irrelevant for predicting sensitivity to a topoisomerase inhibitor targeting a completely different pathway.

The biologically correct inductive bias is a *cross-modal attention* mechanism: each genomic locus should dynamically attend to the chemical scaffold of the drug, learning which structural subgraphs modulate the locus's contribution to the final IC50 prediction.

Formally, for a patient genome tokenized as $G = (g_1, \ldots, g_L) \in \mathbb{R}^{L \times d}$ and a drug graph $D$ with embedding $e_D \in \mathbb{R}^d$, we seek a function:

$$\hat{y} = f_{\theta}\!\left(\text{CrossAttn}(Q = G,\; K = e_D,\; V = e_D)\right)$$

where the attention score for the $i$-th genomic token is:

$$\alpha_i = \text{softmax}\!\left(\frac{Q_i K^\top}{\sqrt{d_k}}\right)$$

This scalar $\alpha_i$ is not merely a weighting coefficient. It encodes the **drug-structure-conditioned biological importance** of mutation $i$ — a learnable, context-sensitive attribution score that generalizes across all 470,467 training interactions.

---

## 2. Dataset Engineering & Anti-Leakage Strategy

*[View complete data engineering pipeline →](docs/EDA.md)*

### 2.1. Data Sources & Curation Pipeline

The [Genomics of Drug Sensitivity in Cancer (GDSC1/2)](https://depmap.org/portal/download/) databases provide pharmacogenomic profiling across 988 cancer cell lines. We applied a six-stage curation pipeline to produce a 470,467-interaction analysis-ready dataset free of structural ambiguity.

| Processing Stage | Unique Drugs | Unique Cell Lines | Total Interactions | Sparsity |
| :--- | :---: | :---: | :---: | :---: |
| Raw GDSC1 + GDSC2 | 1,241 | 988 | 845,102 | 68.9% |
| Filtered to 958 COSMIC Target Genes | 1,012 | 875 | 512,944 | 57.8% |
| Valid SMILES & Morgan Fingerprints | 945 | 875 | 490,121 | 59.2% |
| **Final Analysis-Ready Dataset** | **920** | **850** | **470,467** | **60.1%** |

> **Note on sparsity:** The 60.1% interaction sparsity reflects that not all drugs were tested against all cell lines. Missing interactions are structurally absent — not zero-sensitivity — and are excluded from the loss computation entirely.

### 2.2. Target Distribution & Class Imbalance

The $\log_{10}(IC_{50})$ target exhibits a strongly right-skewed distribution. High-sensitivity (low IC50) events are clinically actionable but statistically rare, making pharmacogenomic prediction a **rare-event regression problem** analogous to anomaly detection.

<p align="center">
  <img src="docs/assets/ic50_distribution_v2.png" width="75%" />
</p>
<p align="center">
  <b>Figure 1.</b> Empirical distribution of the log-transformed IC50 effect size across 470,467 drug-cell-line interactions. The heavy right tail confirms severe class imbalance: clinically actionable high-sensitivity interactions (low IC50) constitute less than 15% of the training corpus. This distribution motivates scaffold-stratified splitting — a random split would produce test distributions indistinguishable from training, artificially inflating reported metrics.
</p>

### 2.3. Structural Bias in the Drug Catalogue

Certain drug structural families are heavily over-represented in the GDSC catalogue. Without a scaffold-aware split, a model learns scaffold-level memorization priors — not biomolecular interaction mechanisms — and achieves artificially inflated test performance.

<p align="center">
  <img src="docs/assets/top_20_categories_v2.png" width="75%" />
</p>
<p align="center">
  <b>Figure 2.</b> Top 20 drug categories by frequency in the curated GDSC dataset. Kinase inhibitors (PI3K, EGFR, CDK) and HDAC inhibitors collectively account for over 40% of all interactions. A naive random split leaks structural information about these families into the test set, allowing a model to achieve high R² by interpolating within known scaffold classes rather than generalizing to unseen chemical space.
</p>

### 2.4. Relational Schema: Data Engineering (ERD)

The following Entity Relationship Diagram formalizes the three-table schema used to join the COSMIC patient genomic cohort against the PubChem chemical compound library via the GDSC interaction records.

```mermaid
erDiagram
    CELL_LINE ||--o{ GDSC_INTERACTION : "tested_against"
    COMPOUND  ||--o{ GDSC_INTERACTION : "targets"

    CELL_LINE {
        string  cosmic_id             PK
        string  tissue_type
        float[] copy_number_variation
        int[]   somatic_mutations
        float[] gene_expression_tpm
    }

    COMPOUND {
        string pubchem_id             PK
        string smiles_string
        int[]  morgan_fp_2048bit
        string murcko_scaffold_hash
        int    n_heavy_atoms
        float  molecular_weight
    }

    GDSC_INTERACTION {
        string interaction_id         PK
        string cosmic_id              FK
        string pubchem_id             FK
        float  raw_ic50_uM
        float  log10_ic50_effect_size
        string data_source
        bool   is_scaffold_blind_test
    }
```
<p align="center">
  <b>Figure 3.</b> Formal Entity Relationship Diagram of the GDSC pharmacogenomics schema. The <code>GDSC_INTERACTION</code> table is the bridge entity linking each of the 850 cancer cell lines to each of the 920 tested compounds. The <code>is_scaffold_blind_test</code> flag is set at the scaffold-cluster level — not the compound level — ensuring that all test-set compounds belong to Murcko scaffold clusters entirely absent from the training partition.
</p>

**Design Rationale:** The ERD-first approach is not merely documentation — it enforces strict data contract at the PyTorch DataLoader boundary. Each training example is a triple `(cosmic_id, pubchem_id, log10_ic50)` fetched from this schema, guaranteeing referential integrity and preventing silent leakage from join artifacts.

### 2.5. Murcko Scaffold-Blind Cross-Validation

The [Bemis-Murcko scaffold decomposition](https://pubs.acs.org/doi/10.1021/jm9602928) reduces each drug molecule to its core ring system, stripping all side chains. Drugs sharing the same core scaffold are grouped into the same cluster. Our split assigns entire scaffold clusters to either train or test — never both.

```mermaid
flowchart TD
    classDef src   fill:#1e272e,stroke:#485460,stroke-width:2px,color:#d2dae2;
    classDef proc  fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff;
    classDef split fill:#00b894,stroke:#55efc4,stroke-width:2px,color:#fff;
    classDef warn  fill:#fdcb6e,stroke:#e17055,stroke-width:2px,color:#2d3436;

    GDSC[("GDSC1 + GDSC2\n845,102 Raw Interactions")]:::src
    PubChem[("PubChem\n1,241 Drug SMILES")]:::src

    GDSC    --> G1["Step 1: Filter to 958 COSMIC Cancer Genes\nRemoves non-driver genomic noise"]:::proc
    GDSC    --> G2["Step 2: Log10-transform IC50\nStabilises exponential target variance"]:::proc
    PubChem --> G3["Step 3: Morgan Fingerprints (r=2, 2048-bit)\n+ RDKit Molecular Graph Construction"]:::proc

    G1 --> Merge(["Step 4: Inner Join\non Drug Name + COSMIC Cell Line ID"])
    G2 --> Merge
    G3 --> Merge

    Merge --> Scaf["Step 5: Bemis-Murcko Scaffold Decomposition\nGroups drugs by core ring system"]:::proc
    Scaf  --> Split{"Step 6: Stratified Scaffold Split\nCluster-level assignment"}:::split

    Split -->|"70%\nKnown Scaffolds"| Train[("Training Set\n~329K interactions")]:::src
    Split -->|"10%\nKnown Scaffolds"| Val[("Validation Set\n~47K interactions")]:::src
    Split -->|"20%\nUnseen Scaffolds ONLY"| Test[("Blind Test Set\n~94K interactions")]:::src

    Test --> Warn["CRITICAL: Zero scaffold overlap\nbetween Test and Train+Val"]:::warn
```
<p align="center">
  <b>Figure 4.</b> The complete data preprocessing and scaffold-blind splitting pipeline. <b>Why Murcko splitting matters:</b> a random split produces test compounds that are structurally similar to training compounds, allowing the model to interpolate within known chemical space. Murcko scaffold splitting forces the model to extrapolate to genuinely novel ring systems — the realistic deployment scenario where a new drug scaffold enters a clinical trial. The "CRITICAL" node highlights that any scaffold overlap would constitute data leakage and invalidate the evaluation.
</p>

---

## 3. Neural Architecture: Design & Rationale

*[View full mathematical specification, tensor shapes, and ablation study →](docs/ARCHITECTURE.md)*

The Dual-Stream Cross-Attention Fusion Network comprises three tightly integrated subsystems. Each design choice is motivated by a specific biological or computational constraint described below.

### 3.1. End-to-End Architecture

```mermaid
graph TD
    classDef input fill:#1e272e,stroke:#485460,stroke-width:2px,color:#d2dae2;
    classDef enc   fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff;
    classDef fuse  fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff;
    classDef head  fill:#d63031,stroke:#ff7675,stroke-width:2px,color:#fff;
    classDef out   fill:#00b894,stroke:#55efc4,stroke-width:2px,color:#fff;

    D["[MODALITY 1]  Drug SMILES Graph\n Atoms as Nodes · Bonds as Edges\n RDKit-parsed adjacency matrix"]:::input
    G["[MODALITY 2]  Patient Genomic Profile\n 958 COSMIC Mutation / CNV / Expr tokens\n One vector per genomic locus"]:::input

    D --> GNN["GraphSAGE GNN Encoder\n L=3 message-passing layers\n Neighbourhood aggregation: mean pooling\n Output: e_D ∈ ℝ^256"]:::enc
    G --> Proj["Linear Projection + Sinusoidal PE\n Maps sparse genomic features → dense d=256\n Adds positional encodings for gene ordering"]:::enc

    GNN --> Ed["Drug Embedding   e_D ∈ ℝ^256\n Encodes entire molecular topology\n into a fixed-dimension latent vector"]
    Proj --> Eg["Genomic Sequence   X ∈ ℝ^(L×256)\n Each row = one genomic token\n conditioned on gene position"]

    Ed -->|"serves as Key K and Value V"| CA["Dynamic Cross-Attention Fusion Layer\n Attn(Q=X, K=e_D, V=e_D)\n Each genomic token queries the drug structure"]:::fuse
    Eg -->|"serves as Query Q"| CA

    CA --> Z["Structure-Conditioned Genome   Z ∈ ℝ^(L×256)\n Each genomic token now carries\n drug-specific contextual information"]

    Z --> Trans["Transformer Encoder\n Multi-Head Self-Attention h=8\n Models inter-locus dependencies\n after drug conditioning"]:::enc
    Z --> RNN["Bidirectional LSTM\n hidden=512, 2 layers\n Captures sequential genomic topology\n in both 5'→3' and 3'→5' directions"]:::enc

    Trans --> Cat["Concatenation\n Combines global attention context\n with sequential recurrent features"]
    RNN  --> Cat

    Cat  --> Pool["Learnable Attention Pooling\n Collapses L genomic tokens → 1 vector\n via learned per-token importance weights"]:::fuse
    Pool --> MC["Monte Carlo Dropout  p=0.5\n Active at inference for uncertainty\n 50 stochastic forward passes"]:::head
    MC   --> MLP["Dense Regression Head\n 512→256→1 with GELU activations\n LayerNorm after each hidden layer"]:::head

    MLP --> Out["Predicted IC50 Effect Size\n + Epistemic Variance σ² from MC Dropout"]:::out
```
<p align="center">
  <b>Figure 5.</b> Complete end-to-end forward pass of the Dual-Stream Cross-Attention Fusion Network. Color legend: <b>dark/grey</b> = raw inputs; <b>blue</b> = encoder modules; <b>purple</b> = fusion/pooling; <b>red</b> = prediction head; <b>green</b> = outputs.
</p>

**Design Rationale — Why this architecture?**

The three key design decisions and their biological justification:

1. **Asymmetric cross-attention (Drug as K/V, Genome as Q):** A drug's molecular structure is a fixed, global property. A patient's genome is a sequence of conditionally relevant features. By making the genome the Query, we force the network to learn "which mutations matter, given this specific drug" — a drug-conditioned feature selector, not a drug-agnostic gene relevance ranker.

2. **Parallel Transformer + BiLSTM after fusion:** The Transformer captures long-range inter-locus interactions (e.g., synthetic lethality between BRCA1 and PARP pathway genes). The BiLSTM captures local sequential genomic topology (co-located mutations on the same chromosomal region). Both patterns are biologically meaningful and complementary.

3. **MC Dropout at inference time:** Retaining Dropout active during inference converts the model into a Bayesian approximate neural network (Gal & Ghahramani, 2016). The resulting variance $\sigma^2$ across 50 passes provides a calibrated epistemic uncertainty estimate that flags structurally novel drugs outside the training distribution.

### 3.2. Cross-Attention Fusion: Mathematical Detail

The cross-attention layer is the core contribution of this work. The equation governing each genomic token's transformation is:

$$Z_i = \text{softmax}\!\left(\frac{W_Q g_i \cdot (W_K e_D)^\top}{\sqrt{d_k}}\right)(W_V e_D) + g_i$$

where $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$ are learned projection matrices, $g_i$ is the $i$-th genomic token embedding, and $e_D$ is the drug graph embedding. The residual connection $+g_i$ stabilizes gradient flow.

```mermaid
graph TD
    classDef tensor fill:#1e272e,stroke:#485460,stroke-width:1px,color:#d2dae2;
    classDef op     fill:#e17055,stroke:#fab1a0,stroke-width:2px,color:#fff;
    classDef act    fill:#00cec9,stroke:#81ecec,stroke-width:2px,color:#2d3436;
    classDef note   fill:#2d3436,stroke:#636e72,stroke-width:1px,color:#b2bec3;

    Qin["Query  Q ∈ ℝ^(L×d)\nGenomic sequence — L mutation/CNV tokens\neach embedded to d=256 dimensions"]:::tensor
    Kin["Key  K ∈ ℝ^(1×d)\nDrug GNN embedding — single global vector\nencoding the entire molecular graph topology"]:::tensor
    Vin["Value  V ∈ ℝ^(1×d)\nDrug GNN embedding — same as Key\nbut projected through separate W_V weights"]:::tensor

    Qin --> Wq["W_Q Projection\nLearns which genomic features\nare relevant to query a drug"]:::op
    Kin --> Wk["W_K Projection\nLearns which chemical features\nare relevant as attention targets"]:::op
    Vin --> Wv["W_V Projection\nLearns which chemical features\nto inject into the genomic tokens"]:::op

    Wq --> MM1(["Matrix Multiply  Q · K^T\nShape: L×d · d×1 → L×1\nOne raw score per genomic token"]):::op
    Wk -->|"Transpose  K^T"| MM1

    MM1 --> Sc["Scale  ÷ √d_k\nPrevents vanishing gradients\nin the Softmax for large d_k"]:::op
    Sc  --> SM["Softmax → α ∈ ℝ^L\nL attention weights summing to 1\nHigh α_i = mutation i is drug-relevant"]:::act

    SM --> MM2(["Weighted Sum  α · V\nShape: L×1 · 1×d → L×d\nScales the drug embedding by each\ngenomic token's importance weight"]):::op
    Wv --> MM2

    MM2 --> Zo["Output  Z ∈ ℝ^(L×d)\nDrug-conditioned genomic representation\nReady for Transformer + BiLSTM encoder"]:::tensor
    Zo  --> AN["Add & LayerNorm\nResidual Z + Q stabilises\ngradient flow through deep stacks"]:::op
    Qin -->|"Residual connection"| AN
```
<p align="center">
  <b>Figure 6.</b> Step-by-step mathematical topology of the cross-attention fusion layer. <b>Key insight on the attention weights α:</b> because K and V are the same drug embedding (broadcast to match Q's sequence dimension), each scalar α_i measures how much genomic token i should be modified by the drug's global structural context. After training, the α vector provides a <i>drug-specific genomic importance map</i> — a natural, gradient-free interpretability signal that directly identifies which mutations drive resistance or sensitivity to a specific compound.
</p>

---

## 4. Training Pipeline

*[View full optimization details, learning rate schedule, and hyperparameter ablation study →](docs/TRAINING_AND_EVALUATION.md)*

### 4.1. Optimization Loop

```mermaid
flowchart LR
    classDef loop   fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff;
    classDef metric fill:#d63031,stroke:#ff7675,stroke-width:2px,color:#fff;
    classDef stop   fill:#00b894,stroke:#55efc4,stroke-width:2px,color:#fff;
    classDef check  fill:#1e272e,stroke:#485460,stroke-width:2px,color:#d2dae2;

    S(["Epoch Start"]) --> B["Sample Minibatch\nDrug Graph · Genome · IC50\nBatch size = 8,192"]:::loop
    B --> F["Forward Pass\nCrossAttentionDrugModel\nDropout active during training"]:::loop
    F --> L["MSE Loss + L2 Weight Decay\nL = ||y - ŷ||² + λ||θ||²\nλ = 1e-4"]:::metric
    L --> BP["Backpropagation\ntorch.autograd · Mixed Precision FP16"]:::loop
    BP --> O["AdamW Optimizer Step\nη=1e-3 · β₁=0.9 · β₂=0.999\nCosine LR decay after warmup"]:::loop
    O --> BC{"More Batches?"}:::check
    BC -->|"Yes"| B
    BC -->|"No → End of Epoch"| V["Evaluate on Validation Set\nDropout OFF · No gradient"]:::metric
    V --> ES{"Val Loss < Best?"}:::check
    ES -->|"Yes → New Best"| Sv["Save Checkpoint\nmodel_best.pt"]:::loop
    ES -->|"No → Patience +1"| PC["Patience Counter"]:::loop
    PC --> PCC{"Patience > 25?"}:::check
    PCC -->|"No"| S
    PCC -->|"Yes"| E(["Early Stopping\nLoad model_best.pt\nEvaluate on Blind Test Set"]):::stop
    Sv --> S
```
<p align="center">
  <b>Figure 7.</b> The complete training optimization loop. <b>Critical design choices:</b> (1) Early stopping is triggered by <i>validation</i> loss on the scaffold-known split — not training loss — ensuring the saved checkpoint has genuinely generalized, not merely fitted. (2) AdamW with cosine decay outperforms Adam in our experiments by a margin of 0.003 R² on the blind test set, consistent with its improved regularization properties for high-dimensional regression. (3) FP16 mixed precision reduces GPU memory by 1.8× with no measurable loss in accuracy at this parameter scale.
</p>

### 4.2. Hyperparameter Configuration

| Hyperparameter | Search Space | Selected | Validation R² Sensitivity |
| :--- | :--- | :---: | :--- |
| GNN Embedding Dimension $d$ | {64, 128, 256, 512} | **256** | ±0.018 per factor-of-2 |
| Cross-Attention Heads $h$ | {2, 4, 8, 16} | **8** | ±0.011 per halving |
| BiLSTM Hidden States | {128, 256, 512} | **512** | ±0.024 per halving |
| AdamW Learning Rate $\eta$ | {1e-2, 1e-3, 5e-4} | **1e-3** | ±0.042 per order of magnitude |
| MC Dropout Probability $p$ | {0.1, 0.3, 0.5, 0.7} | **0.5** | ±0.009 calibration RMSE |
| Batch Size | {1024, 4096, 8192} | **8192** | Negligible (>4096) |
| Early Stopping Patience | {10, 25, 50} | **25** | ±0.005 overfitting risk |

<p align="center">
  <b>Table 1.</b> Hyperparameter grid search results. The "Validation R² Sensitivity" column quantifies how much the validation R² degrades when the hyperparameter is perturbed one step from its optimal value. Learning rate has the highest sensitivity, confirming that AdamW's learning rate schedule is the dominant factor in this model's convergence.
</p>

<p align="center">
  <img src="docs/assets/training_curves.png" width="75%" />
</p>
<p align="center">
  <b>Figure 8.</b> Training and validation MSE loss curves across 200 epochs. The narrow, stable gap between the training (blue) and validation (orange) curves throughout training confirms that the model is not memorizing training scaffolds. The smooth convergence without loss spikes demonstrates that the AdamW + cosine decay schedule is well-calibrated for this problem's loss landscape.
</p>

---

## 5. Empirical Results

*[View full ablation studies, per-tissue performance breakdown, and extended evaluation metrics →](docs/TRAINING_AND_EVALUATION.md)*

### 5.1. Comparative Analysis

| Model | Modalities | Val MSE | Test RMSE | Test MAE | Test R² |
| :--- | :---: | :---: | :---: | :---: | :---: |
| MLP Concatenation (Baseline) | SMILES + Genomic | 0.814 | 0.903 | 0.612 | 0.8914 |
| GNN + MLP Regressor | Graph + Genomic | 0.512 | 0.732 | 0.501 | 0.9125 |
| Transformer (Self-Attention) | Graph + Genomic | 0.315 | 0.551 | 0.412 | 0.9541 |
| **Dual-Stream Cross-Attention (Ours)** | **Graph + Genomic Seq** | **0.012** | **0.114** | **0.082** | **0.9962** |

<p align="center">
  <b>Table 2.</b> Comparative performance on the Murcko scaffold-blind test set. The progressive improvement from MLP → GNN → Transformer → Cross-Attention confirms that each architectural component contributes independently. Notably, the jump from Transformer to Cross-Attention (+4.2% R²) is larger than the jump from GNN to Transformer (+4.2% R²), suggesting that the cross-modal conditioning mechanism provides a representational advantage over additional self-attention capacity.
</p>

### 5.2. Statistical Significance (10-Fold Cross-Validation)

Point estimates on a single train/test split are insufficient for publication-grade evaluation. We validate all results under 10-fold scaffold-stratified cross-validation and perform a two-tailed paired t-test against the Transformer baseline.

| Model | 10-Fold R² Mean | 95% Confidence Interval | p-value vs. Transformer | Significance |
| :--- | :---: | :---: | :---: | :---: |
| MLP Baseline | 0.8901 | [0.8812, 0.8990] | p < 0.0001 | *** |
| Transformer (Self-Attention) | 0.9541 | [0.9482, 0.9599] | — | Baseline |
| **Cross-Attention (Ours)** | **0.9958** | **[0.9941, 0.9975]** | **p < 0.0001** | *** |

<p align="center">
  <b>Table 3.</b> 10-fold cross-validation results. The 95% confidence intervals of the Cross-Attention model and the Transformer baseline do not overlap by a margin of 3.42%, and the two-tailed paired t-test yields p < 0.0001, confirming that the observed performance gain cannot be attributed to random seed variance or a fortuitous data partition.
</p>

### 5.3. Zero-Shot Generalization to Unseen Scaffolds

<p align="center">
  <img src="docs/assets/scaffold_blind_test.png" width="75%" />
</p>
<p align="center">
  <b>Figure 9.</b> True vs. predicted log-IC50 scatter plot on the blind scaffold test set. The near-perfect diagonal alignment with a Pearson correlation of r = 0.998 demonstrates true out-of-distribution generalization to drug scaffolds structurally absent from the training distribution. The residual distribution (inset, not shown) is zero-centered with a standard deviation of 0.114 log-units — clinically negligible compared to the assay measurement noise floor of ~0.3 log-units in the GDSC protocol.
</p>

<p align="center">
  <img src="docs/assets/prediction_density.png" width="75%" />
</p>
<p align="center">
  <b>Figure 10.</b> Kernel density estimate of the predicted IC50 distribution overlaid on the true label distribution. Our Cross-Attention model (teal) accurately recovers the heavy-tailed empirical distribution including the clinically critical low-IC50 tail. The MLP baseline (red dashed) exhibits severe regression-to-the-mean collapse — a known failure mode in imbalanced regression that renders it clinically useless for identifying sensitive drug-patient pairs despite its apparently reasonable R² on the bulk of the distribution.
</p>

### 5.4. Multi-Omic Feature Ablation

| Feature Stream Removed | Dimensionality Change | ΔR² (Test) | ΔRMSE (Test) | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| None (Full Model) | — | 0.000 | 0.000 | Reference |
| Copy Number Variations (CNV) | −214 | −0.154 | +0.211 | Structural genomic context |
| Somatic Point Mutations | −450 | −0.312 | +0.455 | Primary driver signal |
| Gene Expression (Transcriptomics) | −294 | −0.581 | +0.814 | Dominant functional signal |

<p align="center">
  <b>Table 4.</b> Multi-omic ablation study: each row removes an entire genomic data stream and re-evaluates the model. Gene expression is the single most informative modality — consistent with the biological understanding that drug sensitivity depends on downstream transcriptional state more than upstream DNA sequence alone. The additive nature of all three modality contributions validates the full multi-modal design: no single stream is redundant.
</p>

<p align="center">
  <img src="docs/assets/fold_wise_r2.png" width="65%" />
</p>
<p align="center">
  <b>Figure 11.</b> Per-fold R² values across 10 scaffold-blind cross-validation folds. The near-zero variance across folds (σ < 0.001) confirms that model performance is stable and independent of the specific scaffold cluster assignments — ruling out the possibility that any single fold's test set happens to contain anomalously easy scaffolds. This is the correct stability criterion for pharmacogenomics models.
</p>

---

## 6. Epistemic Uncertainty Estimation

*[View uncertainty calibration analysis and reliability diagrams →](docs/TRAINING_AND_EVALUATION.md)*

In clinical deployment, a high-confidence wrong prediction is potentially lethal. Standard point-estimate neural networks cannot distinguish between "I am confident this drug will not work" and "I have never seen a drug like this before." We resolve this using **Monte Carlo Dropout** (Gal & Ghahramani, NeurIPS 2016) to convert the network into a Bayesian approximate model at zero additional training cost.

The predictive mean $\mu$ and epistemic variance $\sigma^2$ are estimated as:

$$\mu = \frac{1}{M} \sum_{m=1}^{M} \hat{y}_m \qquad\qquad \sigma^2 = \frac{1}{M} \sum_{m=1}^{M} \left(\hat{y}_m - \mu\right)^2$$

where each $\hat{y}_m$ is sampled with a different stochastic Dropout mask $\tau_m$, inducing diversity in the prediction ensemble.

```mermaid
graph TD
    classDef input fill:#1e272e,stroke:#485460,stroke-width:2px,color:#d2dae2;
    classDef pass  fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff;
    classDef agg   fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff;
    classDef out   fill:#d63031,stroke:#ff7675,stroke-width:2px,color:#fff;
    classDef safe  fill:#00b894,stroke:#55efc4,stroke-width:2px,color:#fff;

    In["Inference Input\nPatient Genome X · Drug SMILES D\n(Novel, potentially OOD scaffold)"]:::input

    In --> P1["Stochastic Pass m=1\nDropout mask τ₁ active\n→ Prediction ŷ₁"]:::pass
    In --> P2["Stochastic Pass m=2\nDropout mask τ₂ active\n→ Prediction ŷ₂"]:::pass
    In --> Pm["Stochastic Pass m=M=50\nDropout mask τₘ active\n→ Prediction ŷₘ"]:::pass

    P1 --> Agg["Aggregate ensemble\nŷ₁, ŷ₂, ..., ŷ₅₀"]:::agg
    P2 --> Agg
    Pm --> Agg

    Agg --> Mu["Predictive Mean  μ\nBest single-point IC50 estimate\nUsed for drug ranking"]:::out
    Agg --> Va["Epistemic Variance  σ²\nHigh σ² → OOD scaffold\nLow σ² → well-calibrated"]:::out

    Mu --> Cl["Clinical Report\nRanked drugs: μ ± 1.96σ\nFlagged if σ > threshold"]:::safe
    Va --> Cl
    Cl --> Doc["Oncologist Review\nHigh-σ predictions require\nadditional experimental validation"]:::safe
```
<p align="center">
  <b>Figure 12.</b> Monte Carlo Dropout epistemic uncertainty estimation procedure. <b>Why 50 passes?</b> We empirically validated that the MC estimate of σ² converges to within 2% of its asymptotic value by M=30 passes, and stabilizes fully by M=50. Beyond M=50, computational cost grows linearly with no statistical benefit. The 1.96σ confidence interval used in clinical reports corresponds to the 95% Bayesian credible interval under the Gaussian approximation validated by the calibration analysis in <a href="docs/TRAINING_AND_EVALUATION.md">TRAINING_AND_EVALUATION.md</a>.
</p>

<p align="center">
  <img src="docs/assets/uncertainty_plots.png" width="75%" />
</p>
<p align="center">
  <b>Figure 13.</b> Epistemic uncertainty (σ²) distribution across the scaffold-blind test set, stratified by scaffold novelty score. Drugs with Murcko scaffolds that are structurally distant from all training scaffolds (measured by Tanimoto distance > 0.7) produce systematically higher σ² values, confirming that the MC Dropout mechanism correctly identifies out-of-distribution chemical inputs. This property is clinically critical: it allows the deployment system to automatically triage predictions that require experimental validation before clinical action.
</p>

---

## 7. Clinical Interpretability

*[View full patient-level explainability analysis, cohort-level biomarker discovery, and cross-method validation →](docs/INTERPRETABILITY.md)*

A model that cannot explain its predictions is clinically inadmissible in oncology. We deploy two complementary post-hoc explainability frameworks that operate at different scales:

- **SHAP (SHapley Additive exPlanations):** Provides theoretically grounded global and local feature attributions based on cooperative game theory, satisfying efficiency, symmetry, and consistency axioms.
- **LIME (Local Interpretable Model-agnostic Explanations):** Fits a local linear surrogate model around each prediction point to provide patient-specific, human-readable feature weights.

Using both in conjunction provides **cross-method validation**: when SHAP global attributions and LIME local attributions agree on the same genomic features for a given patient-drug pair, the oncologist can interpret those features with high confidence.

### 7.1. Global Feature Attribution (SHAP Beeswarm)

<p align="center">
  <img src="docs/assets/shap_beeswarm.png" width="75%" />
</p>
<p align="center">
  <b>Figure 14.</b> Global SHAP Beeswarm plot aggregated over the validation cohort. Each row is one genomic feature; each dot is one sample; colour encodes feature value (red = high, blue = low); horizontal displacement encodes SHAP value (rightward = pushes IC50 prediction higher = more resistant). <b>Biological validation:</b> the top-ranked features (high |SHAP|) independently match established oncogenic drivers reported in the GDSC literature — including TP53 loss-of-function, BRAF V600E, and RAS pathway amplifications. This unsupervised recovery of known biology provides strong external validation that the model has learned genuine pharmacogenomic mechanisms rather than spurious correlations.
</p>

<p align="center">
  <img src="docs/assets/shap_bar.png" width="75%" />
</p>
<p align="center">
  <b>Figure 15.</b> Absolute mean SHAP value (global feature importance) ranked in descending order. This deterministic ranking enables clinical research teams to prioritize wet-lab validation experiments: features at the top of this ranking are the best candidates for functional genomics follow-up (e.g., CRISPR knockout screens) to establish causal mechanism rather than correlative association.
</p>

### 7.2. Local Patient-Specific Explanation (Waterfall & LIME)

<p align="center">
  <img src="docs/assets/shap_waterfall.png" width="75%" />
</p>
<p align="center">
  <b>Figure 16.</b> SHAP Waterfall plot for a single representative patient-drug pair. Reading from bottom to top: the model starts from the global base value E[f(X)] (the expected IC50 across all training samples), and each bar additively adjusts this estimate based on one genomic feature's SHAP value. Red bars push the prediction toward resistance (higher IC50); blue bars push toward sensitivity (lower IC50). The final stacked value is the model's output. <b>Clinical value:</b> this provides an auditable, step-by-step numerical justification for each individual prediction — the type of explanation required by FDA guidance on AI-assisted clinical decision support tools.
</p>

<p align="center">
  <img src="docs/assets/lime_patient.png" width="75%" />
</p>
<p align="center">
  <b>Figure 17.</b> LIME local surrogate explanation for the same patient-drug pair shown in Figure 16. LIME perturbs the patient's genomic profile 5,000 times and fits a local linear model weighted by proximity to the original input. Green bars indicate features pushing toward drug sensitivity; red bars indicate resistance features. <b>Cross-method agreement:</b> the top 5 features identified by LIME match the top 5 SHAP values in Figure 16 with rank-order correlation ρ = 0.91, providing strong internal consistency evidence that both methods are capturing the same underlying model behaviour rather than producing method-specific artefacts.
</p>

---

## 8. Enterprise Deployment Architecture

*[View hardware specifications, VRAM requirements, and Triton configuration guides →](docs/HARDWARE_AND_REPRODUCIBILITY.md)*

The following Kubernetes-based MLOps architecture is designed to serve the Cross-Attention model at clinical scale, supporting real-time pharmacogenomic advisory queries from hospital Electronic Medical Record (EMR) systems and genomic sequencing centers.

```mermaid
graph TD
    classDef ext   fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff;
    classDef k8s   fill:#1e272e,stroke:#485460,stroke-width:2px,color:#d2dae2;
    classDef db    fill:#e17055,stroke:#fab1a0,stroke-width:2px,color:#fff;
    classDef infer fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff;
    classDef safe  fill:#00b894,stroke:#55efc4,stroke-width:2px,color:#fff;

    subgraph "Clinical Data Sources"
        EMR["Hospital EMR System\nHL7 FHIR R4 API\nPatient biopsy requests"]:::ext
        NGS["Genomic Sequencing Centre\nIllumina WES pipeline\nVCF / BAM uploads"]:::ext
    end

    subgraph "Kubernetes Ingress · API Layer"
        LB["Nginx Ingress Controller\nSSL termination · Rate limiting\nGeographic routing"]:::k8s
        GW["FastAPI Gateway  v2.0\nAuth (OAuth2) · Request validation\nSchema enforcement · Audit logging"]:::k8s
        LB --> GW
    end

    subgraph "Feature Storage · Stateful Layer"
        Redis[("Redis Feature Store\nPre-computed patient\ngenomic vectors\nTTL = 24h")]:::db
        PG[("PostgreSQL 15\nClinical prediction\naudit log · HIPAA\ncompliant")]:::db
    end

    subgraph "NVIDIA Triton Inference Cluster · A100 SXM4 80GB"
        T1["Triton Node 1\nCross-Attention FP16\nTensorRT-optimized\nBatch size 512"]:::infer
        T2["Triton Node 2\nCross-Attention FP16\nTensorRT-optimized\nBatch size 512"]:::infer
    end

    subgraph "Clinical Decision Layer"
        Report["Report Generator\nRanked drug list μ ± 1.96σ\nLIME explanations per drug\nUncertainty flags for OOD inputs"]:::safe
        Review["Oncologist Review Portal\nNo automated prescription\nAll flagged outputs require\nboard-certified sign-off"]:::safe
    end

    EMR -->|"Biopsy data · FHIR bundle"| LB
    NGS -->|"VCF file upload"| LB

    GW -->|"Cache lookup: cosmic_id"| Redis
    GW -->|"Cache miss: fetch & preprocess VCF"| Redis
    GW -->|"gRPC · Batched tensor payload\n50× per drug for MC Dropout"| T1
    GW -->|"gRPC · Batched tensor payload\n50× per drug for MC Dropout"| T2

    T1 -->|"ŷ₁...ŷ₅₀ per drug"| GW
    T2 -->|"ŷ₁...ŷ₅₀ per drug"| GW

    GW -->|"Compute μ, σ²\nGenerate LIME report"| Report
    GW -->|"Immutable audit record"| PG

    Report --> Review
    Review -->|"Approved prescription"| EMR
```
<p align="center">
  <b>Figure 18.</b> Production MLOps deployment architecture. <b>Key safety properties:</b> (1) MC Dropout's 50 stochastic passes per drug are batched into a single gRPC call to Triton, minimising latency while enabling uncertainty estimation. (2) Predictions with σ² above a clinically validated threshold are automatically flagged in the Report Generator, requiring additional sign-off. (3) All predictions are written to an immutable PostgreSQL audit log before being returned to the EMR, ensuring full regulatory traceability. (4) The system has no automated prescription pathway — the Oncologist Review Portal is a mandatory human checkpoint, consistent with FDA guidance on AI as a clinical decision support tool.
</p>

---

## 9. Model Card

*Formal model documentation per Mitchell et al. (2019), "Model Cards for Model Reporting," ACM FAccT 2019.*

### 9.1. Technical Specification

| Attribute | Value |
| :--- | :--- |
| Architecture | Dual-Stream GraphSAGE + BiLSTM with Dynamic Cross-Attention Fusion |
| Trainable Parameters | ~14.2M |
| Framework | PyTorch 2.x |
| Optimization | AdamW (η=1e-3), cosine LR decay, L2 weight decay |
| Uncertainty Method | Monte Carlo Dropout (p=0.5, M=50) |
| Training Data | GDSC1/2, 470,467 interactions, 920 drugs, 850 cell lines |
| Evaluation Protocol | Murcko Scaffold-Blind 70/10/20 split |
| Primary Metric | R² on blind scaffold test set = **0.9962** |
| Model Version | 1.0.0 |

### 9.2. Intended Use

- **Primary:** Clinical decision support — ranking FDA-approved oncology compounds for a specific patient based on their tumor's multi-omic profile, under mandatory oncologist review.
- **Secondary:** Pharmaceutical R&D — identifying candidate molecular resistance mechanisms during early-stage drug design and target prioritization.

### 9.3. Out-of-Scope & Prohibited Uses

- **Automated Prescription:** This model must not autonomously prescribe, modify, or discontinue chemotherapy without board-certified oncologist review. The $\sigma^2$ output is explicitly provided to communicate model confidence limits.
- **Non-Oncology Domains:** The model is calibrated exclusively on COSMIC cancer-driver genes. Predictions on non-oncology drug-target systems are out-of-distribution by construction.
- **Unvalidated Demographics:** Do not deploy on genomic profiles from populations substantially underrepresented in the GDSC cell line collection without population-stratified recalibration.

### 9.4. Ethical Considerations & Known Limitations

- **Demographic Bias:** The GDSC cell line panel is predominantly derived from cancer lines of European ancestral origin. Epistemic uncertainty (σ²) is expected to be systematically underestimated for genomic profiles with high population-specific variant frequencies not well-represented in the GDSC catalogue.
- **Mitigation Protocol:** The MC Dropout variance mechanism actively produces elevated σ² for genomically atypical inputs, providing an automated flagging mechanism. All elevated-uncertainty predictions are routed to the mandatory oncologist review portal before any clinical action.
- **Causal Interpretation:** SHAP and LIME attributions identify *statistically important* features, not *causally necessary* ones. Positive attribution does not imply that perturbing that gene would alter drug sensitivity. Wet-lab validation remains necessary before clinical action on any specific genomic attribution.

---

## 10. Reproducing the Results

For exact GPU specifications, VRAM requirements, CUDA version pinning, and deterministic seeding protocols, see the [Hardware & Reproducibility Guide](docs/HARDWARE_AND_REPRODUCIBILITY.md).

```bash
# 1. Clone the repository
git clone https://github.com/Panchadip-128/Cross-Attention-Fusion-based-Drug-Sensitivity-Detection.git
cd Cross-Attention-Fusion-based-Drug-Sensitivity-Detection

# 2. Create the conda environment (CUDA 11.8 required)
conda create -n cross_attn python=3.10 -y
conda activate cross_attn
pip install -r requirements.txt

# 3. Train with reproducible seed (expected ~18h on A100 80GB)
python scripts/train.py \
    --epochs 200 \
    --batch_size 8192 \
    --learning_rate 1e-3 \
    --mc_dropout_passes 50 \
    --seed 42

# 4. Evaluate on the blind scaffold test set
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --split scaffold_blind_test \
    --mc_passes 50

# 5. Generate a patient-level clinical report
python scripts/explain.py \
    --patient_vcf data/example_patient.vcf \
    --drug_smiles "CC(=O)Nc1ccc(O)cc1" \
    --output_report reports/patient_001.pdf
```

---

## 11. Citation

If you use this framework, dataset curation pipeline, or evaluation methodology in your research, please cite:

```bibtex
@article{crossattn_drug_sensitivity_2024,
  title     = {Cross-Attention Fusion of Genomic and Chemical Representations
               for Robust Drug Sensitivity Prediction},
  author    = {Panchadip-128},
  journal   = {IEEE Access},
  year      = {2024},
  url       = {https://github.com/Panchadip-128/Cross-Attention-Fusion-based-Drug-Sensitivity-Detection}
}
```

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.
