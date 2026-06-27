import torch
import torch.nn as nn
from torch_geometric.data import Data
from .components import PositionalEncoding, CrossAttentionFusion, AttentionPooling

class CrossAttentionDrugModel(nn.Module):
    """
    Drug sensitivity regression model combining:
      - Drug embedding lookup
      - Cross-attention drug-genomic fusion
      - Parallel TransformerEncoder and BiLSTM streams
      - AttentionPooling for BiLSTM output
      - MC Dropout for uncertainty quantification
    """
    def __init__(
        self, input_dim: int, hidden_dim: int, num_drugs: int,
        n_heads: int, lstm_hidden: int, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.drug_emb = nn.Embedding(num_embeddings=num_drugs + 1, embedding_dim=hidden_dim, padding_idx=0)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(d_model=hidden_dim, dropout=dropout, max_len=500_000)
        self.cross_attn_fusion = CrossAttentionFusion(embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=2)

        self.bilstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=lstm_hidden, num_layers=1,
            bidirectional=True, batch_first=True, dropout=0.0,
        )
        bilstm_out_dim = lstm_hidden * 2
        self.attn_pool = AttentionPooling(dim=bilstm_out_dim)

        self.mc_dropout = nn.Dropout(p=dropout)

        concat_dim = hidden_dim + bilstm_out_dim
        self.regressor = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, drug_ids = data.x, data.drug_ids
        x = self.input_proj(x).unsqueeze(1)
        x = self.pos_enc(x)

        drug_vec = self.drug_emb(drug_ids + 1).unsqueeze(1)
        x = self.cross_attn_fusion(x, drug_vec)

        t_out = self.transformer(x)
        t_pool = t_out.mean(dim=1)

        lstm_out, _ = self.bilstm(x)
        l_pool = self.attn_pool(lstm_out)

        t_pool = self.mc_dropout(t_pool)
        l_pool = self.mc_dropout(l_pool)

        combined = torch.cat([t_pool, l_pool], dim=-1)
        return self.regressor(combined).squeeze(-1)
