"""Training losses for the constrained neural ranker.

Three components, combined by :func:`total_loss`:

  * ``step_ce``      — next-step cross-entropy (the main LM objective).
  * ``role_ce``      — auxiliary next-role cross-entropy (the transfer signal;
                       roles are family-agnostic so this is what survives OOD).
  * ``semantic_loss``— a differentiable *constraint* loss (Xu et al. 2018,
                       "A Semantic Loss Function for Deep Learning with Symbolic
                       Knowledge"). It penalizes probability mass placed on
                       rule-INVALID continuations, i.e. it pushes the softmax to
                       concentrate inside the symbolically-legal support
                       ``valid_id_mask``. Formally, per non-pad position,

                           L = -log( sum_{c legal} softmax(step_logits)_c )

                       which is 0 iff all mass sits on legal steps and grows as
                       mass leaks onto illegal steps.

This module imports ``torch``. It is only used at *training* time. The
``valid_id_mask`` it consumes is expensive to build (one symbolic
``valid_next_set`` per position), so ``model.train_ranker`` only constructs it
when ``sem_w > 0`` or ``mask_train`` is set — see that module's docstring.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = ["step_ce", "role_ce", "semantic_loss", "total_loss"]

_NEG_INF = float("-inf")


def step_ce(step_logits: torch.Tensor, target: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Next-step cross-entropy.

    Parameters
    ----------
    step_logits : ``[B, T, V]`` float tensor of step logits.
    target      : ``[B, T]`` long tensor of gold next-step ids.
    pad_id      : padding id, ignored in the loss.
    """
    return F.cross_entropy(step_logits.transpose(1, 2), target, ignore_index=pad_id)


def role_ce(role_logits: torch.Tensor, role_target: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Auxiliary next-role cross-entropy.

    Parameters
    ----------
    role_logits : ``[B, T, R]`` float tensor of role logits.
    role_target : ``[B, T]`` long tensor of gold next-role ids.
    pad_id      : padding id (in role-target space), ignored in the loss.
    """
    return F.cross_entropy(role_logits.transpose(1, 2), role_target, ignore_index=pad_id)


def semantic_loss(
    step_logits: torch.Tensor,
    valid_id_mask: torch.Tensor,
    pad_mask: torch.Tensor,
) -> torch.Tensor:
    """Semantic / constraint loss: mean over non-pad positions of
    ``-logsumexp(log_softmax(step_logits) restricted to the legal set)``.

    Parameters
    ----------
    step_logits   : ``[B, T, V]`` float logits.
    valid_id_mask : ``[B, T, V]`` bool tensor, True where the step is legal.
    pad_mask      : ``[B, T]`` float tensor, 1.0 at real (non-pad) positions, 0.0
                    at padding. The loss is averaged over positions where
                    ``pad_mask == 1``.

    Positions whose mask is *all-False* (no legal continuation at all — should not
    happen since the gold next is always legal, but guarded for robustness)
    contribute exactly 0 and are excluded from the denominator, so they neither
    explode the loss (no ``-inf``) nor bias the mean.
    """
    logp = F.log_softmax(step_logits, dim=-1)                       # [B, T, V]
    masked = logp.masked_fill(~valid_id_mask, _NEG_INF)             # illegal -> -inf
    valid_logp = torch.logsumexp(masked, dim=-1)                    # [B, T]

    # Positions with at least one legal step; everything else contributes 0.
    has_legal = valid_id_mask.any(dim=-1)                           # [B, T] bool
    weight = pad_mask * has_legal.to(pad_mask.dtype)                # [B, T]

    # Replace any non-finite entries (all-False positions) with 0 before weighting
    # so the -inf produced by logsumexp over an empty set never propagates.
    per_pos = torch.where(has_legal, -valid_logp, torch.zeros_like(valid_logp))
    num = (per_pos * weight).sum()
    den = weight.sum().clamp(min=1.0)
    return num / den


def total_loss(
    step_logits: torch.Tensor,
    role_logits: torch.Tensor,
    step_tgt: torch.Tensor,
    role_tgt: torch.Tensor,
    pad_id: int,
    valid_id_mask: Optional[torch.Tensor] = None,
    role_w: float = 0.3,
    sem_w: float = 0.0,
    role_pad_id: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combined objective ``step_ce + role_w * role_ce + sem_w * semantic_loss``.

    Parameters
    ----------
    step_logits, role_logits : model outputs, ``[B, T, V]`` and ``[B, T, R]``.
    step_tgt, role_tgt       : gold next-step / next-role ids, ``[B, T]``.
    pad_id                   : padding id for the *step* target stream, ignored
                               in ``step_ce``.
    valid_id_mask            : ``[B, T, V]`` bool legal-step mask. Required iff
                               ``sem_w > 0``; ignored otherwise.
    role_w, sem_w            : component weights.
    role_pad_id              : padding id for the *role* target stream, ignored
                               in ``role_ce``. The role stream pads with a
                               dedicated id (``NUM_ROLES``) that differs from the
                               step ``pad_id`` (0), so this must be passed
                               explicitly — otherwise role-pad slots leak into
                               the loss and real ``LOGISTICS`` (role 0) targets
                               are wrongly ignored. Defaults to ``pad_id`` only
                               for back-compat when the two conventions coincide.

    Returns
    -------
    (loss, components) where ``components`` is a dict of detached python floats:
    ``step_ce``, ``role_ce``, ``semantic`` (0.0 when ``sem_w == 0``) and
    ``total``.
    """
    rpid = pad_id if role_pad_id is None else role_pad_id
    sce = step_ce(step_logits, step_tgt, pad_id)
    rce = role_ce(role_logits, role_tgt, rpid)
    loss = sce + role_w * rce

    sem_val = step_logits.new_zeros(())
    if sem_w > 0.0:
        if valid_id_mask is None:
            raise ValueError("semantic_loss requested (sem_w>0) but valid_id_mask is None")
        # pad_mask: 1.0 at positions whose step target is a real (non-pad) id.
        pad_mask = (step_tgt != pad_id).to(step_logits.dtype)
        sem_val = semantic_loss(step_logits, valid_id_mask, pad_mask)
        loss = loss + sem_w * sem_val

    components = {
        "step_ce": float(sce.detach()),
        "role_ce": float(rce.detach()),
        "semantic": float(sem_val.detach()),
        "total": float(loss.detach()),
    }
    return loss, components


# ---------------------------------------------------------------------------
# Self-test (CPU): tiny tensors, check shapes/values and that semantic_loss is
# zero when all mass is legal, positive when it leaks, and guards all-False rows.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, V, R = 2, 4, 7, 3
    pad_id = 0

    step_logits = torch.randn(B, T, V, requires_grad=True)
    role_logits = torch.randn(B, T, R, requires_grad=True)
    step_tgt = torch.randint(1, V, (B, T))
    role_tgt = torch.randint(0, R, (B, T))
    step_tgt[0, -1] = pad_id  # one padded position

    # Plain CE + role.
    loss, comp = total_loss(step_logits, role_logits, step_tgt, role_tgt, pad_id)
    print("plain total_loss:", round(comp["total"], 4), comp)
    assert comp["semantic"] == 0.0
    loss.backward()
    assert step_logits.grad is not None

    # semantic_loss == 0 when probability mass is forced fully onto a single legal step.
    big = torch.full((B, T, V), -1e4)
    big[..., 1] = 1e4                       # all mass on id 1
    mask = torch.zeros(B, T, V, dtype=torch.bool)
    mask[..., 1] = True                     # id 1 is legal everywhere
    pad_mask = torch.ones(B, T)
    sem0 = semantic_loss(big, mask, pad_mask)
    print("semantic_loss (all mass legal):", float(sem0))
    assert float(sem0) < 1e-3

    # Positive when mass sits on an illegal id.
    mask2 = torch.zeros(B, T, V, dtype=torch.bool)
    mask2[..., 2] = True                    # legal id is 2 but mass is on id 1
    sem1 = semantic_loss(big, mask2, pad_mask)
    print("semantic_loss (mass illegal):", round(float(sem1), 4))
    assert float(sem1) > 1.0

    # All-False mask rows contribute 0 (no inf / nan).
    mask3 = torch.zeros(B, T, V, dtype=torch.bool)   # nothing legal anywhere
    sem2 = semantic_loss(big, mask3, pad_mask)
    print("semantic_loss (all-False mask):", float(sem2))
    assert float(sem2) == 0.0 and torch.isfinite(sem2)

    # total_loss with sem_w>0 wires it in and stays finite.
    valid_mask = torch.zeros(B, T, V, dtype=torch.bool)
    valid_mask.scatter_(-1, step_tgt.unsqueeze(-1).clamp(min=0), True)  # gold is legal
    loss2, comp2 = total_loss(step_logits, role_logits, step_tgt, role_tgt, pad_id,
                              valid_id_mask=valid_mask, role_w=0.3, sem_w=0.5)
    print("total_loss with sem_w=0.5:", round(comp2["total"], 4), comp2)
    assert torch.isfinite(loss2)
    print("SELF-TEST PASSED")
