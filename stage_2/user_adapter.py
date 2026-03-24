from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

# User embedding을 cross-attention token space로 projection
class UserProjection(nn.Module):
    # d_cross: cross-attention token dimension
    # in_dim: input embedding dimension
    # eps: LayerNorm epsilon
    def __init__(self, d_cross: int, in_dim: int = 3584, eps: float = 1e-5) -> None:
        super().__init__()
        if d_cross <= 0:
            raise ValueError(f"d_cross must be positive, got {d_cross}")
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")

        self.in_dim = in_dim
        self.d_cross = d_cross

        self.proj = nn.Linear(in_dim, d_cross)
        self.norm = nn.LayerNorm(d_cross, eps=eps)

    def forward(
        self,
        user_emb: Tensor,
        user_emb_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if user_emb.ndim != 3:
            raise ValueError(
                f"user_emb must have shape [B, L, {self.in_dim}], got ndim={user_emb.ndim}"
            )
        bsz, seq_len, emb_dim = user_emb.shape
        if emb_dim != self.in_dim:
            raise ValueError(
                f"Expected user_emb last dim={self.in_dim}, but got {emb_dim}"
            )

        if user_emb_attention_mask is not None:
            if user_emb_attention_mask.ndim != 2:
                raise ValueError(
                    "user_emb_attention_mask must have shape [B, L], "
                    f"got ndim={user_emb_attention_mask.ndim}"
                )
            if user_emb_attention_mask.shape != (bsz, seq_len):
                raise ValueError(
                    "user_emb_attention_mask shape mismatch: "
                    f"expected {(bsz, seq_len)}, got {tuple(user_emb_attention_mask.shape)}"
                )

        x = self.proj(user_emb)
        x = self.norm(x)

        if user_emb_attention_mask is not None:
            mask = user_emb_attention_mask.to(device=x.device)
            if mask.dtype != torch.bool:
                mask = mask > 0
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)

        return x


class UserCrossAttentionAdapter(nn.Module):
    """User-side cross-attention branch independent of text cross-attention.

    This module consumes query states from the main model and projected user tokens
    from ``UserProjection``, then produces user-conditioned attention output.

    Args:
        d_model: Query/output dimension.
        d_cross: User token input dimension.
        num_heads: Number of attention heads.
        dropout: Attention probability dropout.
        eps: LayerNorm epsilon.
        use_query_norm: Whether to apply LayerNorm on query input.
        use_user_norm: Whether to apply LayerNorm on user token input.
    """

    def __init__(
        self,
        d_model: int,
        d_cross: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        eps: float = 1e-5,
        use_query_norm: bool = True,
        use_user_norm: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if d_cross <= 0:
            raise ValueError(f"d_cross must be positive, got {d_cross}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.d_cross = d_cross
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.query_norm = nn.LayerNorm(d_model, eps=eps) if use_query_norm else nn.Identity()
        self.user_norm = nn.LayerNorm(d_cross, eps=eps) if use_user_norm else nn.Identity()

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_cross, d_model)
        self.v_proj = nn.Linear(d_cross, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)

    def _reshape_heads(self, x: Tensor) -> Tensor:
        # [B, N, D] -> [B, H, N, Hd]
        bsz, n_tokens, _ = x.shape
        return x.view(bsz, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _validate_mask(mask: Tensor, batch_size: int, seq_len: int) -> Tensor:
        if mask.ndim != 2:
            raise ValueError(f"user_attention_mask must have shape [B, L], got ndim={mask.ndim}")
        if mask.shape != (batch_size, seq_len):
            raise ValueError(
                "user_attention_mask shape mismatch: "
                f"expected {(batch_size, seq_len)}, got {tuple(mask.shape)}"
            )
        if mask.dtype != torch.bool:
            mask = mask > 0
        return mask

    def _masked_softmax(self, attn_scores: Tensor, attn_mask: Tensor) -> Tensor:
        """Numerically stable masked softmax with all-masked row handling.

        Args:
            attn_scores: [B, H, N_q, L]
            attn_mask: [B, 1, 1, L] (bool)
        """
        fill_value = torch.finfo(attn_scores.dtype).min
        masked_scores = attn_scores.masked_fill(~attn_mask, fill_value)
        probs = torch.softmax(masked_scores, dim=-1)

        # Explicit renormalization keeps rows with all-masked keys at zero.
        probs = probs * attn_mask.to(dtype=probs.dtype)
        denom = probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        probs = probs / denom
        return probs

    def forward(
        self,
        query: Tensor,
        user_tokens: Tensor,
        user_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute user-side cross-attention output.

        Args:
            query: Tensor with shape [B, N_q, d_model].
            user_tokens: Tensor with shape [B, L, d_cross].
            user_attention_mask: Optional mask with shape [B, L].
                Non-zero values indicate valid user token positions.

        Returns:
            Tensor with shape [B, N_q, d_model].
        """
        if query.ndim != 3:
            raise ValueError(
                f"query must have shape [B, N_q, {self.d_model}], got ndim={query.ndim}"
            )
        if user_tokens.ndim != 3:
            raise ValueError(
                f"user_tokens must have shape [B, L, {self.d_cross}], got ndim={user_tokens.ndim}"
            )

        bsz_q, n_q, d_q = query.shape
        bsz_u, l_u, d_u = user_tokens.shape

        if d_q != self.d_model:
            raise ValueError(f"query last dim must be {self.d_model}, got {d_q}")
        if d_u != self.d_cross:
            raise ValueError(f"user_tokens last dim must be {self.d_cross}, got {d_u}")
        if bsz_q != bsz_u:
            raise ValueError(
                f"Batch size mismatch between query ({bsz_q}) and user_tokens ({bsz_u})"
            )

        q_in = self.query_norm(query)
        u_in = self.user_norm(user_tokens)

        q = self._reshape_heads(self.q_proj(q_in))
        k = self._reshape_heads(self.k_proj(u_in))
        v = self._reshape_heads(self.v_proj(u_in))

        # [B, H, N_q, Hd] x [B, H, Hd, L] -> [B, H, N_q, L]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if user_attention_mask is not None:
            mask = self._validate_mask(user_attention_mask.to(device=attn_scores.device), bsz_q, l_u)
            attn_mask = mask[:, None, None, :]  # [B, 1, 1, L]
            attn_probs = self._masked_softmax(attn_scores, attn_mask)
        else:
            attn_probs = torch.softmax(attn_scores, dim=-1)

        attn_probs = self.attn_dropout(attn_probs)

        # [B, H, N_q, L] x [B, H, L, Hd] -> [B, H, N_q, Hd]
        attn_out = torch.matmul(attn_probs, v)

        # [B, H, N_q, Hd] -> [B, N_q, D]
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz_q, n_q, self.d_model)
        return self.out_proj(attn_out)


if __name__ == "__main__":
    torch.manual_seed(42)

    d_model = 1024
    d_cross = 1024

    user_projection = UserProjection(d_cross=d_cross)
    user_adapter = UserCrossAttentionAdapter(
        d_model=d_model,
        d_cross=d_cross,
        num_heads=16,
        dropout=0.0,
    )

    user_emb = torch.randn(2, 29, 3584)
    mask = torch.ones(2, 29, dtype=torch.long)
    mask[1, -4:] = 0
    query = torch.randn(2, 128, d_model)

    projected_user_tokens = user_projection(user_emb, user_emb_attention_mask=mask)
    user_out = user_adapter(query=query, user_tokens=projected_user_tokens, user_attention_mask=mask)

    print("user_emb shape:", tuple(user_emb.shape))
    print("mask shape:", tuple(mask.shape))
    print("query shape:", tuple(query.shape))
    print("projected_user_tokens shape:", tuple(projected_user_tokens.shape))
    print("user_out shape:", tuple(user_out.shape))
