import pytest
import torch
from src.kernels.fused_attention import FusedAttention

def test_fused_attention_cpu_fallback():
    # Input shapes: [batch_size, num_heads, seq_len, head_dim]
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)
    
    attention = FusedAttention()
    try:
        out = attention(q, k, v)
        # Verify output matches expected dimensions
        assert out.shape == q.shape
    except Exception as e:
        pytest.fail(f"Attention forward pass failed on CPU: {e}")
