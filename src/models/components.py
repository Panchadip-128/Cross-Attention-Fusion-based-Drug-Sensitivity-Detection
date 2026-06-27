import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al., 2017).
    Adds position-dependent sine/cosine signals to token embeddings.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500_000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

class CrossAttentionFusion(nn.Module):
    """
    Cross-attention module that conditions the genomic representation on a drug embedding.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, genomic_seq: torch.Tensor, drug_emb: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.cross_attn(query=genomic_seq, key=drug_emb, value=drug_emb)
        return self.norm(genomic_seq + self.dropout(attn_out))

class AttentionPooling(nn.Module):
    """
    Learnable weighted aggregation over the sequence dimension.
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score_proj = nn.Linear(dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.score_proj(x)
        weights = torch.softmax(scores, dim=1)
        return (weights * x).sum(dim=1)
