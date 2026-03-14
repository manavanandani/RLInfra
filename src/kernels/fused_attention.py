import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from rich.console import Console

console = Console()

class FusedAttention(nn.Module):
    """High-performance Attention module falling back gracefully from Triton custom kernel to native PyTorch SDPA."""

    def __init__(self, scale: float = None):
        super().__init__()
        self.scale = scale

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Forward pass executing fused attention."""
        # Query shapes: [batch_size, num_heads, seq_len, head_dim]
        head_dim = q.shape[-1]
        sm_scale = self.scale if self.scale is not None else 1.0 / math.sqrt(head_dim)
        
        try:
            from .triton_attention import triton_fused_attention, HAS_TRITON
            # Check if CUDA and Triton are fully supported in current process
            if HAS_TRITON and q.is_cuda and torch.cuda.is_available():
                return triton_fused_attention(q, k, v, sm_scale)
        except Exception as e:
            # Catch exceptions and log warnings for diagnostic transparency
            console.print(f"[bold yellow][Warning][/bold yellow] Custom Triton attention launch failed: {e}. Falling back to PyTorch SDPA.")

        # CPU/MPS fallback or Triton-unavailable platforms
        # PyTorch scaled_dot_product_attention utilizes FlashAttention / Memory Efficient SDKs internally
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=sm_scale
        )
