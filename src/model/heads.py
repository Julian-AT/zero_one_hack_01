"""Multi-task heads applied on top of any sequence backbone.

- LM head: next-token CE over the vocabulary
- Validity head: binary BCE on the <EOS> pooled representation
- Rule-ID head: 11-way (10 rules + "valid") cross-entropy on <EOS> pooled

The validity + rule-ID heads are zeroed at the loss-weight level by default
(see configs/train/default.yaml); they activate once labels are passed in.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.data.validator import NUM_RULE_CLASSES


class LMHead(nn.Module):
    """Tied LM head — weight shared with the input embedding by the caller."""

    def __init__(self, d_model: int, vocab_size: int, tied_embedding: nn.Embedding | None = None):
        super().__init__()
        if tied_embedding is not None:
            self.weight = tied_embedding.weight  # tied
            self.bias: torch.Tensor | None = None
        else:
            proj = nn.Linear(d_model, vocab_size, bias=False)
            self.weight = proj.weight
            self.bias = None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: [B, L, D] → logits: [B, L, V]
        return torch.nn.functional.linear(hidden, self.weight, self.bias)


class ValidityHead(nn.Module):
    """Binary validity classifier on a pooled representation."""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        # pooled: [B, D] → logit: [B]
        return self.net(pooled).squeeze(-1)


class RuleIDHead(nn.Module):
    """11-way rule classifier on a pooled representation (10 rules + valid)."""

    def __init__(self, d_model: int, num_classes: int = NUM_RULE_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        # pooled: [B, D] → logits: [B, num_classes]
        return self.net(pooled)
