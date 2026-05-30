"""Multi-task loss: LM CE + validity BCE + rule-ID CE.

The trainer passes batch labels and the model's output dict; this module
returns the scalar total loss + a per-component dict for logging.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from transformer_xlstm.data.load import IGNORE_INDEX


@dataclass
class LossWeights:
    lm_weight: float = 1.0
    validity_weight: float = 0.0
    rule_id_weight: float = 0.0


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine LM + validity + rule-ID losses.

    The LM loss uses next-token prediction: predict input_ids[1:] from
    input_ids[:-1]. Token positions whose label == IGNORE_INDEX are excluded.
    """
    logs: dict[str, float] = {}

    lm_logits = outputs["lm_logits"]
    labels = batch["labels"]
    shift_logits = lm_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    lm_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )
    logs["lm_loss"] = float(lm_loss.item())

    total = weights.lm_weight * lm_loss

    if weights.validity_weight > 0 and "validity_logit" in outputs:
        validity_logit = outputs["validity_logit"]
        validity_target = batch["validity"].float()
        validity_loss = F.binary_cross_entropy_with_logits(validity_logit, validity_target)
        total = total + weights.validity_weight * validity_loss
        logs["validity_loss"] = float(validity_loss.item())

    if weights.rule_id_weight > 0 and "rule_id_logits" in outputs:
        rule_logits = outputs["rule_id_logits"]
        rule_target = batch["rule_class"]
        rule_loss = F.cross_entropy(rule_logits, rule_target)
        total = total + weights.rule_id_weight * rule_loss
        logs["rule_id_loss"] = float(rule_loss.item())

    logs["total_loss"] = float(total.item())
    return total, logs
