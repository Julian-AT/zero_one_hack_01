"""Arch-name → builder dispatch.

`build_model("transformer", model_cfg, vocab_size, enable_multitask_heads)`
returns a model with the standard forward interface.
"""

from __future__ import annotations

import torch.nn as nn


def build_model(
    arch: str, model_cfg: dict, vocab_size: int, enable_multitask_heads: bool = False
) -> nn.Module:
    arch = arch.lower()
    if arch == "transformer":
        from src.model.transformer import build_transformer

        return build_transformer(model_cfg, vocab_size, enable_multitask_heads)
    if arch == "xlstm":
        try:
            from src.model.xlstm_model import build_xlstm
        except ImportError as e:
            raise RuntimeError(
                "xlstm package not installed (or CUDA unavailable). "
                "On Leonardo: 'pixi add --pypi xlstm'. On Mac: skip xlstm cells."
            ) from e
        return build_xlstm(model_cfg, vocab_size, enable_multitask_heads)
    raise ValueError(f"Unknown arch: {arch!r}. Choose 'transformer' or 'xlstm'.")
