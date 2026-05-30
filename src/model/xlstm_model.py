"""xLSTM (NX-AI) wrapper with the same forward interface as the Transformer.

Requires the `xlstm` package + CUDA + Triton; lazily imported so module load
on CPU-only dev hosts (Mac) doesn't crash. On hosts without the library, the
factory raises a clear error explaining the situation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.data.validator import NUM_RULE_CLASSES
from src.model.heads import LMHead, RuleIDHead, ValidityHead


@dataclass
class XLSTMConfig:
    vocab_size: int
    d_model: int = 256
    num_blocks: int = 4
    block_pattern: tuple[str, ...] = ("mlstm", "slstm", "mlstm", "slstm")
    dropout: float = 0.1
    max_seq_len: int = 768
    # mLSTM
    mlstm_num_heads: int = 4
    mlstm_conv1d_kernel_size: int = 4
    mlstm_qkv_proj_blocksize: int = 4
    # sLSTM
    slstm_num_heads: int = 4
    slstm_conv1d_kernel_size: int = 4
    slstm_backend: str = "cuda"


def _build_xlstm_stack(cfg: XLSTMConfig) -> nn.Module:
    """Construct the xLSTMBlockStack via NX-AI's official API.

    Raises ImportError if the `xlstm` package is missing — Mac-only hosts
    won't have it; that's expected. We only build the stack on CUDA hosts.
    """
    from xlstm import (
        FeedForwardConfig,
        mLSTMBlockConfig,
        mLSTMLayerConfig,
        sLSTMBlockConfig,
        sLSTMLayerConfig,
        xLSTMBlockStack,
        xLSTMBlockStackConfig,
    )

    mlstm_cfg = mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            conv1d_kernel_size=cfg.mlstm_conv1d_kernel_size,
            qkv_proj_blocksize=cfg.mlstm_qkv_proj_blocksize,
            num_heads=cfg.mlstm_num_heads,
        )
    )
    slstm_cfg = sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            backend=cfg.slstm_backend,
            num_heads=cfg.slstm_num_heads,
            conv1d_kernel_size=cfg.slstm_conv1d_kernel_size,
            bias_init="powerlaw_blockdependent",
        ),
        feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
    )

    slstm_at = [i for i, p in enumerate(cfg.block_pattern) if p == "slstm"]
    stack_cfg = xLSTMBlockStackConfig(
        mlstm_block=mlstm_cfg,
        slstm_block=slstm_cfg,
        num_blocks=cfg.num_blocks,
        embedding_dim=cfg.d_model,
        slstm_at=slstm_at,
        context_length=cfg.max_seq_len,
        dropout=cfg.dropout,
    )
    return xLSTMBlockStack(stack_cfg)


class XLSTMModel(nn.Module):
    def __init__(self, cfg: XLSTMConfig, enable_multitask_heads: bool = False):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.stack = _build_xlstm_stack(cfg)
        self.norm_out = nn.LayerNorm(cfg.d_model)
        self.lm_head = LMHead(cfg.d_model, cfg.vocab_size, tied_embedding=self.embed)

        self.has_multitask = enable_multitask_heads
        if enable_multitask_heads:
            self.validity_head = ValidityHead(cfg.d_model)
            self.rule_id_head = RuleIDHead(cfg.d_model, num_classes=NUM_RULE_CLASSES)

        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, attn_mask: torch.Tensor | None = None, **kwargs
    ) -> dict[str, torch.Tensor]:
        x = self.drop(self.embed(input_ids))
        x = self.stack(x)
        x = self.norm_out(x)
        lm_logits = self.lm_head(x)
        out: dict[str, torch.Tensor] = {"lm_logits": lm_logits, "hidden": x}
        if self.has_multitask:
            if attn_mask is not None:
                last_idx = attn_mask.sum(dim=1) - 1
                last_idx = last_idx.clamp(min=0)
                pooled = x[torch.arange(x.size(0)), last_idx]
            else:
                pooled = x[:, -1]
            out["validity_logit"] = self.validity_head(pooled)
            out["rule_id_logits"] = self.rule_id_head(pooled)
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_xlstm(
    model_cfg: dict, vocab_size: int, enable_multitask_heads: bool = False
) -> XLSTMModel:
    mlstm = model_cfg.get("mlstm", {})
    slstm = model_cfg.get("slstm", {})
    cfg = XLSTMConfig(
        vocab_size=vocab_size,
        d_model=model_cfg["d_model"],
        num_blocks=model_cfg["num_blocks"],
        block_pattern=tuple(model_cfg["block_pattern"]),
        dropout=model_cfg.get("dropout", 0.1),
        max_seq_len=model_cfg.get("max_seq_len", 768),
        mlstm_num_heads=mlstm.get("num_heads", 4),
        mlstm_conv1d_kernel_size=mlstm.get("conv1d_kernel_size", 4),
        mlstm_qkv_proj_blocksize=mlstm.get("qkv_proj_blocksize", 4),
        slstm_num_heads=slstm.get("num_heads", 4),
        slstm_conv1d_kernel_size=slstm.get("conv1d_kernel_size", 4),
        slstm_backend=slstm.get("backend", "cuda"),
    )
    return XLSTMModel(cfg, enable_multitask_heads=enable_multitask_heads)
