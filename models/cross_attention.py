import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention:
      Q comes from query tokens
      K,V come from context tokens

    Shapes:
      q: [B, Nq, Dq]
      kv: [B, Nk, Dk]
      output: [B, Nq, Dq]
    """
    def __init__(self, dim_q: int, dim_kv: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim_q % num_heads == 0, "dim_q must be divisible by num_heads"
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.num_heads = num_heads
        self.head_dim = dim_q // num_heads
        self.scale = self.head_dim ** -0.5

        # 3 projections (as your professor said)
        self.wq = nn.Linear(dim_q, dim_q, bias=True)
        self.wk = nn.Linear(dim_kv, dim_q, bias=True)  # project K to dim_q
        self.wv = nn.Linear(dim_kv, dim_q, bias=True)  # project V to dim_q

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_q, dim_q, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        B, Nq, Dq = q_tokens.shape
        B2, Nk, Dk = kv_tokens.shape
        assert B == B2, "Batch mismatch"

        # Q: [B, heads, Nq, head_dim]
        q = self.wq(q_tokens).reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)

        # K,V: [B, heads, Nk, head_dim]
        k = self.wk(kv_tokens).reshape(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(kv_tokens).reshape(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention: [B, heads, Nq, Nk]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Out: [B, heads, Nq, head_dim] -> [B, Nq, Dq]
        out = (attn @ v).transpose(1, 2).reshape(B, Nq, Dq)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class CrossAttentionBlock(nn.Module):
    """
    Residual Cross-Attention block (PreNorm):
      y = q + CrossAttn(LN(q), kv)
    """
    def __init__(self, dim_q: int, dim_kv: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim_q)
        self.xattn = CrossAttention(dim_q, dim_kv, num_heads, attn_drop=attn_drop, proj_drop=proj_drop)

    def forward(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        return q_tokens + self.xattn(self.norm(q_tokens), kv_tokens)
