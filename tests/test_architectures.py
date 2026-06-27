import pytest
import torch
from src.models.architectures import CrossAttentionDrugModel

def test_cross_attention_drug_model_forward(mock_genomic_features, mock_drug_indices):
    # Setup hyperparameters
    num_drugs = 100
    genomic_features_dim = 32
    d_model = 64
    num_heads = 4
    lstm_hidden_size = 32
    dropout = 0.1
    
    model = CrossAttentionDrugModel(
        num_drugs=num_drugs,
        genomic_features_dim=genomic_features_dim,
        d_model=d_model,
        num_heads=num_heads,
        lstm_hidden_size=lstm_hidden_size,
        dropout=dropout
    )
    
    # Pass through the model
    # Note: architecture might expect (drug_idx, genomic_feat) depending on forward implementation
    # We will just verify it runs and outputs expected shape.
    
    # src/models/architectures.py expects:
    # forward(self, data) where data has data.x (genomic), data.drug_idx
    # Since we don't have PyG data here exactly matching, we will construct a dummy PyG batch
    
    from torch_geometric.data import Data, Batch
    
    # 2 graphs in a batch
    data1 = Data(x=torch.randn(10, genomic_features_dim), drug_idx=torch.tensor([5]))
    data2 = Data(x=torch.randn(15, genomic_features_dim), drug_idx=torch.tensor([12]))
    
    batch = Batch.from_data_list([data1, data2])
    
    out = model(batch)
    
    # Should output a single scalar per graph in the batch
    assert out.shape == (2, 1)
