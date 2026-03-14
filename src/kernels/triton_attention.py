import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def _fused_attention_fwd_kernel(
        Q, K, V, sm_scale, Out,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vk, stride_vn,
        stride_oz, stride_oh, stride_om, stride_on,
        Z, H, N_CTX,
        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr
    ):
        """Triton kernel for fused forward FlashAttention."""
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        
        # Compute block pointers
        q_offset = off_hz * stride_qh + start_m * BLOCK_M * stride_qm
        k_offset = off_hz * stride_kh
        v_offset = off_hz * stride_vh
        
        # Load block Q
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)
        
        q_ptrs = Q + q_offset + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
        q = tl.load(q_ptrs)
        
        # Initialize helper variables for online softmax
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        
        # Loop over K, V blocks
        for start_n in range(0, N_CTX, BLOCK_N):
            # Load K and V
            k_ptrs = K + k_offset + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk)
            k = tl.load(k_ptrs)
            
            v_ptrs = V + v_offset + (offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn)
            v = tl.load(v_ptrs)
            
            # Compute QK^T
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, tl.trans(k))
            qk *= sm_scale
            
            # Causal masking if needed (simulated as full attention here for brevity)
            # Apply online softmax scaling
            m_ij = tl.max(qk, 1)
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            
            m_next = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_next)
            beta = tl.exp(m_ij - m_next)
            
            # Correct accumulator
            acc = acc * alpha[:, None]
            acc += tl.dot(p.to(tl.float16), v)
            
            l_i = l_i * alpha + l_ij * beta
            m_i = m_next
            
        # Write output back
        acc = acc / l_i[:, None]
        out_offset = off_hz * stride_oh + start_m * BLOCK_M * stride_om
        o_ptrs = Out + out_offset + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_on)
        tl.store(o_ptrs, acc.to(tl.float16))

def triton_fused_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float) -> torch.Tensor:
    """Invokes the custom Triton fused attention kernel."""
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available on this platform.")
        
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    
    # Batch size, number of heads, seq len, head dim
    Z, H, N_CTX, D_HEAD = q.shape
    out = torch.empty_like(q)
    
    # Configure kernel parameters
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)
    
    _fused_attention_fwd_kernel[grid](
        q, k, v, sm_scale, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, N_CTX,
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=D_HEAD, BLOCK_N=BLOCK_N
    )
    return out
