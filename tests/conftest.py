import pytest
import torch
from torch_geometric.data import Data

@pytest.fixture
def mock_genomic_features():
    """Returns mock genomic features (batch_size=2, seq_len=10, features=32)."""
    return torch.randn(2, 10, 32)

@pytest.fixture
def mock_drug_indices():
    """Returns mock categorical drug indices (batch_size=2)."""
    return torch.tensor([5, 12], dtype=torch.long)

@pytest.fixture
def mock_pyg_data():
    """Returns a mock PyTorch Geometric Data object."""
    x = torch.randn(10, 3) # 10 nodes, 3 node features
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)
