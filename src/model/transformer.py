"""Decoder-only Transformer with RoPE positional encoding and RMSNorm.

Implements the multi-task interface expected by the trainer:

    forward(input_ids, attn_mask, ...) -> dict[str, Tensor]
        {
            "lm_logits":     [B, L, V],
            "validity_logit": [B],         # only when heads enabled
            "rule_id_logits": [B, C],      # only when heads enabled
        }

Designed for ~5M / ~25M / ~100M parameter counts on a single A100.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.validator import NUM_RULE_CLASSES
from src.model.heads import LMHead, RuleIDHead, ValidityHead


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to q and k.

    Shapes:
        q, k: [B, H, L, D_head]
        cos, sin: [L, D_head]   (D_head must be even)
    """

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        # interleave -x2, x1 back into the last dim
        out = torch.stack([-x2, x1], dim=-1)
        return out.flatten(-2)

    # broadcast cos/sin across batch and heads
    cos_b = cos[None, None, :, :]  # [1,1,L,D]
    sin_b = sin[None, None, :, :]
    return q * cos_b + rotate(q) * sin_b, k * cos_b + rotate(k) * sin_b


def make_rope_cache(
    seq_len: int, head_dim: int, device: torch.device, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) caches of shape [seq_len, head_dim]."""
    half = head_dim // 2
    freqs = torch.arange(0, half, device=device, dtype=torch.float32)
    inv = 1.0 / (base ** (freqs / half))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = pos[:, None] * inv[None, :]  # [L, half]
    # interleave each angle so it lines up with the rotate() above
    angles = torch.stack([angles, angles], dim=-1).flatten(-2)  # [L, head_dim]
    return angles.cos(), angles.sin()


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 256,
        use_rope: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope
        self.max_seq_len = max_seq_len

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        # x: [B, L, D]
        B, L, _ = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_rope and rope is not None:
            cos, sin = rope
            q, k = apply_rope(q, k, cos[:L], sin[:L])

        # Recent PyTorch refuses both attn_mask and is_causal=True. Combine
        # the causal triangle with the key-padding mask ourselves when there
        # is padding; otherwise let SDPA do its own causal fast path.
        dropout_p = self.drop.p if self.training else 0.0
        if attn_mask is not None:
            # attn_mask: [B, L] with 1=real, 0=pad
            causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))
            key_pad = attn_mask[:, None, None, :].bool()  # [B, 1, 1, L]
            combined = causal[None, None, :, :] & key_pad  # [B, 1, L, L]
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=combined,
                is_causal=False,
                dropout_p=dropout_p,
            )
        else:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                dropout_p=dropout_p,
            )  # [B, H, L, D_head]
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU MLP — common in modern decoder-only LMs."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc1(x).chunk(2, dim=-1)
        return self.drop(self.fc2(F.silu(a) * b))


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        max_seq_len: int,
        use_rope: bool,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_seq_len, use_rope)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff, dropout)

    def forward(self, x, attn_mask, rope):
        x = x + self.attn(self.norm1(x), attn_mask, rope)
        x = x + self.mlp(self.norm2(x))
        return x


@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 256
    rope: bool = True
    rmsnorm: bool = True


class DecoderTransformer(nn.Module):
    def __init__(self, cfg: TransformerConfig, enable_multitask_heads: bool = False):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [
                Block(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, cfg.max_seq_len, cfg.rope)
                for _ in range(cfg.n_layers)
            ]
        )
        self.norm_out = RMSNorm(cfg.d_model)
        self.lm_head = LMHead(cfg.d_model, cfg.vocab_size, tied_embedding=self.embed)

        self.has_multitask = enable_multitask_heads
        if enable_multitask_heads:
            self.validity_head = ValidityHead(cfg.d_model)
            self.rule_id_head = RuleIDHead(cfg.d_model, num_classes=NUM_RULE_CLASSES)

        self._rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _rope(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._rope_cache is None or self._rope_cache[0].device != device:
            self._rope_cache = make_rope_cache(
                self.cfg.max_seq_len, self.cfg.d_model // self.cfg.n_heads, device
            )
        return self._rope_cache

    def forward(
        self, input_ids: torch.Tensor, attn_mask: torch.Tensor | None = None, **kwargs
    ) -> dict[str, torch.Tensor]:
        x = self.drop(self.embed(input_ids))
        rope = self._rope(input_ids.device) if self.cfg.rope else None
        for block in self.blocks:
            x = block(x, attn_mask, rope)
        x = self.norm_out(x)
        lm_logits = self.lm_head(x)

        out: dict[str, torch.Tensor] = {"lm_logits": lm_logits, "hidden": x}
        if self.has_multitask:
            # Pool at the last *real* position (per attn_mask) — that's the
            # <EOS> token in our schema.
            if attn_mask is not None:
                last_idx = attn_mask.sum(dim=1) - 1  # [B]
                last_idx = last_idx.clamp(min=0)
                pooled = x[torch.arange(x.size(0)), last_idx]  # [B, D]
            else:
                pooled = x[:, -1]
            out["validity_logit"] = self.validity_head(pooled)
            out["rule_id_logits"] = self.rule_id_head(pooled)
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_transformer(
    model_cfg: dict, vocab_size: int, enable_multitask_heads: bool = False
) -> DecoderTransformer:
    cfg = TransformerConfig(
        vocab_size=vocab_size,
        **{
            k: model_cfg[k]
            for k in (
                "d_model",
                "n_layers",
                "n_heads",
                "d_ff",
                "dropout",
                "max_seq_len",
                "rope",
                "rmsnorm",
            )
            if k in model_cfg
        },
    )
    return DecoderTransformer(cfg, enable_multitask_heads=enable_multitask_heads)
