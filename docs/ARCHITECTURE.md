# Research-Grade Systems Architecture & Methodological Flowcharts

This document serves as the comprehensive visual and mathematical guide to the **Cross-Attention Drug Sensitivity Prediction Framework**. Below are 8 meticulously detailed schematics, separated into the **Architectural Tensor Designs** and the **Operational Data Flowcharts**, providing a complete end-to-end understanding of the system's logic, layers, and clinical deployment strategies.

---

## Part 1: Architectural Designs

These diagrams illustrate the forward-pass mathematics, tensor shape transformations, and structural graph topologies of the neural networks involved.

### 1. Full End-to-End Prediction Architecture
The master schematic showing the integration of molecular graph representations and genomic sequence embeddings via dynamic cross-attention fusion.

```mermaid
graph TD
    classDef input fill:#2d3436,stroke:#636e72,stroke-width:2px,color:#fff;
    classDef encoder fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff;
    classDef fusion fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff;
    classDef head fill:#d63031,stroke:#ff7675,stroke-width:2px,color:#fff;
    classDef output fill:#00b894,stroke:#55efc4,stroke-width:2px,color:#fff;

    D[Drug SMILES Graph<br>Nodes & Adjacency]:::input --> GNN[GNN Molecular Encoder<br>GraphSAGE / GCN]:::encoder
    G[Patient Genomic Profile<br>Mutations & CNV]:::input --> SEQ[Linear Projection & PE<br>Sequence Embedding]:::encoder

    GNN --> D_Emb[Drug Latent Vector<br>e_drug]
    SEQ --> G_Emb[Genomic Embeddings<br>X_pos]

    D_Emb -->|Key & Value| CA[Dynamic Cross-Attention Fusion Layer]:::fusion
    G_Emb -->|Query| CA

    CA --> Fused[Structure-Conditioned Genome<br>Z_fused]
    
    Fused --> Trans[Multi-Head Self-Attention<br>Transformer Encoder]:::encoder
    Fused --> BiLSTM[Bidirectional LSTM<br>Recurrent Dynamics]:::encoder

    Trans --> Concat[Concatenation]
    BiLSTM --> Concat

    Concat --> AttnPool[Learnable Attention Pooling]:::fusion
    AttnPool --> MCDrop[Monte Carlo Dropout Layer<br>p=0.5]:::head
    MCDrop --> MLP[Dense Regression Head<br>GELU & LayerNorm]:::head
    
    MLP --> Out[Predicted IC50 Effect Size<br>+ Predictive Variance]:::output
```

### 2. Dual-Stream Cross-Attention Mechanism
A deep dive into the $Q, K, V$ matrix projections that allow genomic mutations to directly attend to structural chemical features.

```mermaid
graph TD
    classDef tensor fill:#2d3436,stroke:#b2bec3,stroke-width:1px,color:#fff;
    classDef op fill:#e17055,stroke:#fab1a0,stroke-width:2px,color:#fff;
    classDef act fill:#00cec9,stroke:#81ecec,stroke-width:2px,color:#fff;

    Q_in[Genomic Query Input<br>Q ∈ ℝᴸˣᵈ]:::tensor
    K_in[Chemical Key Input<br>K ∈ ℝ¹ˣᵈ]:::tensor
    V_in[Chemical Value Input<br>V ∈ ℝ¹ˣᵈ]:::tensor

    Q_in --> Q_proj[W_q Projection]:::op
    K_in --> K_proj[W_k Projection]:::op
    V_in --> V_proj[W_v Projection]:::op

    Q_proj --> MatMul1((⊗ MatMul)):::op
    K_proj -->|Transpose| MatMul1

    MatMul1 --> Scale[Scale by 1 / √d_k]:::op
    Scale --> Softmax[Softmax Normalization]:::act

    Softmax --> MatMul2((⊗ MatMul)):::op
    V_proj --> MatMul2

    MatMul2 --> Z[Output Context Z ∈ ℝᴸˣᵈ]:::tensor
    Z --> AddNorm[Add & LayerNorm]:::op
    Q_in -->|Residual Connection| AddNorm
```

### 3. Molecular Graph Neural Network (GNN) Encoder
Visualizing the message-passing and readout aggregation across a drug's structural atoms (nodes) and bonds (edges).

```mermaid
graph LR
    classDef graphLayer fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff;
    classDef pool fill:#d63031,stroke:#ff7675,stroke-width:2px,color:#fff;

    subgraph "Graph Generation"
        SMILES[SMILES String] --> RDKit[RDKit Feature Extractor]
        RDKit --> Nodes[Atom Features<br>v_i]
        RDKit --> Edges[Bond Types<br>e_ij]
    end

    subgraph "Message Passing (x L Layers)"
        Nodes --> MP1[Message Aggregation<br>∑ N(v_i)]:::graphLayer
        Edges --> MP1
        MP1 --> Update1[Node Update<br>GRU / ReLU]:::graphLayer
    end

    Update1 --> Readout[Global Mean/Max Readout]:::pool
    Readout --> MLP_D[Dense Layers + BN]:::graphLayer
    MLP_D --> e_drug[Final Drug Embedding]
```

### 4. Genomic BiLSTM Sequence Encoder
Detailed view of the bidirectional sequential processing of genomic tokens to capture localized dependencies.

```mermaid
graph TD
    classDef rnn fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff;
    
    X1[Token 1] --> L1[Forward LSTM Cell]:::rnn
    X2[Token 2] --> L2[Forward LSTM Cell]:::rnn
    X3[Token ... ] --> L3[Forward LSTM Cell]:::rnn
    
    L1 --> L2
    L2 --> L3
    
    X1 --> R1[Backward LSTM Cell]:::rnn
    X2 --> R2[Backward LSTM Cell]:::rnn
    X3 --> R3[Backward LSTM Cell]:::rnn
    
    R3 --> R2
    R2 --> R1
    
    L1 --> C1((Concat))
    R1 --> C1
    
    L2 --> C2((Concat))
    R2 --> C2
    
    L3 --> C3((Concat))
    R3 --> C3
    
    C1 --> Z_seq[Z_local Sequence]
    C2 --> Z_seq
    C3 --> Z_seq
```
