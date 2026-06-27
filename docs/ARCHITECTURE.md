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

---

## Part 2: Operational Data Flowcharts

These flowcharts describe the rigorous methodological workflows governing data engineering, model training, explainability, and clinical deployment.

### 5. Data Preprocessing & Splitting Pipeline (Murcko Scaffolds)
Ensuring strict generalization by preventing structural chemical leakage between train and test distributions.

```mermaid
flowchart TD
    classDef data fill:#b2bec3,stroke:#636e72,stroke-width:2px,color:#2d3436;
    classDef process fill:#e17055,stroke:#fab1a0,stroke-width:2px,color:#fff;
    classDef split fill:#00b894,stroke:#55efc4,stroke-width:2px,color:#fff;

    GDSC[(GDSC Database<br>Raw IC50 & Genomics)]:::data
    PubChem[(PubChem<br>SMILES Strings)]:::data

    GDSC --> FilterGen[Filter 958 COSMIC Cancer Genes]:::process
    GDSC --> FilterIC50[Log-transform IC50 Effect Size]:::process
    PubChem --> Morgan[Generate Morgan Fingerprints & Graphs]:::process

    FilterGen --> Merge((Merge Datasets))
    FilterIC50 --> Merge
    Morgan --> Merge

    Merge --> Scaffold[Murcko Scaffold Clustering]:::process
    Scaffold --> SplitStrat{Stratified Scaffold Split}:::split

    SplitStrat -->|70%| Train[(Train Set<br>Known Scaffolds)]:::data
    SplitStrat -->|10%| Val[(Validation Set<br>Known Scaffolds)]:::data
    SplitStrat -->|20%| Test[(Blind Test Set<br>Unseen Scaffolds)]:::data
```

### 6. Training & Optimization Workflow
The iterative loop of forward propagation, loss calculation, and backpropagation utilizing Early Stopping.

```mermaid
flowchart LR
    classDef loop fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff;
    classDef metric fill:#d63031,stroke:#ff7675,stroke-width:2px,color:#fff;

    Start((Start Epoch)) --> Batch[Load Minibatch<br>Drug, Genome, IC50]:::loop
    Batch --> Forward[Forward Pass<br>CrossAttentionDrugModel]:::loop
    Forward --> Loss[Calculate MSE Loss + L2 Penalty]:::metric
    Loss --> Backprop[Backpropagation<br>Autograd]:::loop
    Backprop --> Step[Optimizer Step<br>AdamW]:::loop
    
    Step --> BatchCheck{More Batches?}
    BatchCheck -->|Yes| Batch
    BatchCheck -->|No| Val[Evaluate on Validation Set]:::metric
    
    Val --> ES{Validation Loss Improved?}
    ES -->|Yes| Save[Save Best Weights]:::loop
    ES -->|No| Wait[Increment Patience Counter]:::loop
    
    Wait --> PatienceCheck{Patience > 25?}
    PatienceCheck -->|Yes| Stop((Early Stop))
    PatienceCheck -->|No| Start
    Save --> Start
```

### 7. SHAP & LIME Interpretability Pipeline
Extracting post-hoc actionable intelligence from the black-box model.

```mermaid
flowchart TD
    classDef explain fill:#fdcb6e,stroke:#ffeaa7,stroke-width:2px,color:#2d3436;

    Model[Trained Cross-Attention Model] --> Data[Scaffold-Blind Test Set]
    Data --> SHAP[SHAP DeepExplainer]:::explain
    Data --> LIME[LIME Tabular Explainer]:::explain

    SHAP --> S_Global[Aggregate Marginal Contributions]:::explain
    S_Global --> S_Plot[Global Beeswarm / Bar Plots<br>Identify Key Genomic Drivers]

    LIME --> L_Perturb[Perturb Patient Genomic Profile]:::explain
    L_Perturb --> L_Local[Fit Local Surrogate Ridge Model]:::explain
    L_Local --> L_Plot[Patient-Specific Waterfall Plots<br>Identify Individual Resistance Factors]
```

### 8. Clinical Deployment & Precision Oncology Workflow
Translating the computational model into a practical, real-time clinical advisory tool.

```mermaid
flowchart TD
    classDef clinic fill:#00cec9,stroke:#81ecec,stroke-width:2px,color:#2d3436;
    classDef sys fill:#2d3436,stroke:#b2bec3,stroke-width:2px,color:#fff;

    Patient[New Patient Biopsy]:::clinic --> Seq[Next-Generation Sequencing<br>WES/WGS]:::clinic
    Seq --> Mut[Extract 958 Target Mutations]:::clinic
    
    FDA[FDA-Approved Drug Library]:::sys --> Lib[SMILES Database]:::sys
    
    Mut --> Inference{Cross-Attention Model Server}
    Lib --> Inference
    
    Inference --> MCD[100 MC Dropout Passes]:::sys
    MCD --> Agg[Calculate Mean IC50 & Variance]:::sys
    
    Agg --> Rank[Rank Drugs by Predicted Sensitivity]:::clinic
    Rank --> Filter[Filter out High Variance/Uncertainty]:::clinic
    Filter --> Report[Generate Clinical Interpretability Report<br>with LIME explanations]:::clinic
    Report --> Oncologist((Oncologist Review))
```
