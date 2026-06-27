import torch
import numpy as np
from torch_geometric.data import Data
from collections import defaultdict

def build_edge_index(drug_ids: np.ndarray) -> torch.Tensor:
    """
    Build a COO edge_index tensor connecting all samples that share the same drug ID.
    """
    drug_to_indices = defaultdict(list)
    for sample_idx, drug_id in enumerate(drug_ids):
        drug_to_indices[int(drug_id)].append(sample_idx)

    src_nodes, dst_nodes = [], []
    for indices in drug_to_indices.values():
        for i in range(len(indices)):
            for j in range(len(indices)):
                if i != j:
                    src_nodes.append(indices[i])
                    dst_nodes.append(indices[j])

    if not src_nodes:
        return torch.zeros((2, 0), dtype=torch.long)

    return torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

def build_data(X: np.ndarray, y: np.ndarray, drug_ids: np.ndarray) -> Data:
    """
    Assemble a PyTorch Geometric Data object.
    """
    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    drug_id_tensor = torch.tensor(drug_ids, dtype=torch.long)
    edge_index = build_edge_index(drug_ids)

    return Data(
        x=x_tensor,
        y=y_tensor,
        edge_index=edge_index,
        drug_ids=drug_id_tensor,
    )

def make_batch(data: Data, idx: torch.Tensor) -> Data:
    """
    Creates a mini-batch Data object from selected row indices.
    """
    return Data(
        x=data.x[idx],
        y=data.y[idx],
        drug_ids=data.drug_ids[idx],
        edge_index=torch.empty((2, 0), dtype=torch.long, device=data.x.device)
    )
