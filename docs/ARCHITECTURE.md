# Model Architecture: Mathematical Formulation

This document details the mathematical architecture of the **CrossAttentionDrugModel**.

## 1. Embedding and Positional Encoding

### Drug Embedding
Let $D \in \mathbb{N}$ represent the categorical index of a drug. We project this index into a continuous embedding space:
$$ e_{drug} = \text{Embedding}(D) \in \mathbb{R}^d $$
where $d$ is the hidden dimension.

### Genomic Sequence & Positional Encoding
Let $X \in \mathbb{R}^{L \times f}$ be the genomic feature sequence of length $L$. We apply a linear projection to match the hidden dimension, followed by standard sinusoidal Positional Encoding (PE) to preserve sequence order:
$$ X_{pos} = \text{Linear}(X) + \text{PE} \in \mathbb{R}^{L \times d} $$

## 2. Dynamic Cross-Attention Fusion

To condition the patient's genomic profile on the specific drug, we utilize Multi-Head Cross-Attention (MHCA).

- **Query ($Q$):** The genomic features ($X_{pos}$)
- **Key ($K$) & Value ($V$):** The drug embedding ($e_{drug}$) expanded to match sequence dimensions.

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

This mechanism yields a fused representation $Z_{fused} \in \mathbb{R}^{L \times d}$, highlighting the specific genomic features that are biologically relevant to the input drug structure.

## 3. Dual-Stream Processing

The fused representation $Z_{fused}$ is processed in parallel by two distinct neural streams to capture both global context and localized sequential dynamics.

### Stream 1: Transformer Encoder
A standard Multi-Head Self-Attention Transformer Encoder captures global, long-range dependencies across the entire genomic sequence.
$$ Z_{global} = \text{TransformerEncoder}(Z_{fused}) $$

### Stream 2: Bidirectional LSTM
A BiLSTM processes the sequence to capture localized, directional dependencies that the Transformer might overlook.
$$ Z_{local} = \text{BiLSTM}(Z_{fused}) $$

The streams are subsequently concatenated along the feature dimension:
$$ Z_{concat} = [Z_{global} ; Z_{local}] \in \mathbb{R}^{L \times 2d} $$

## 4. Attention Pooling

Instead of naive mean-pooling across the sequence dimension $L$, we employ a learnable Attention Pooling mechanism to dynamically weight the most informative sequence steps.

$$ w = \text{softmax}(\text{Linear}(Z_{concat})) $$
$$ Z_{pooled} = \sum_{i=1}^{L} w_i \cdot Z_{concat, i} \in \mathbb{R}^{2d} $$

## 5. Epistemic Uncertainty via MC Dropout

Before the final regression head, we apply Monte Carlo (MC) Dropout. By keeping dropout active during inference, we can sample the network $N$ times to produce an empirical distribution of predictions. The variance of this distribution quantifies the model's **epistemic uncertainty**.
