import pytest
import torch
from src.models.components import PositionalEncoding, CrossAttentionFusion, AttentionPooling

def test_positional_encoding():
    d_model = 64
    seq_len = 20
    batch_size = 4
    pe = PositionalEncoding(d_model=d_model, max_len=100)
    
    x = torch.zeros(batch_size, seq_len, d_model)
    out = pe(x)
    
    assert out.shape == (batch_size, seq_len, d_model)
    # The first row of PE shouldn't be all zeros (since it adds sinusoidal values)
    assert not torch.allclose(out[0, 0, :], torch.zeros(d_model))

def test_cross_attention_fusion():
    d_model = 64
    num_heads = 4
    batch_size = 2
    seq_len = 10
    
    fusion = CrossAttentionFusion(d_model=d_model, num_heads=num_heads)
    
    # query: (batch, seq, d_model), keys/values: (batch, seq, d_model)
    query = torch.randn(batch_size, seq_len, d_model)
    key_val = torch.randn(batch_size, seq_len, d_model)
    
    out = fusion(query, key_val)
    
    assert out.shape == (batch_size, seq_len, d_model)

def test_attention_pooling():
    d_model = 64
    batch_size = 4
    seq_len = 15
    
    pool = AttentionPooling(d_model=d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    
    out = pool(x)
    
    # Should reduce the seq_len dimension
    assert out.shape == (batch_size, d_model)
