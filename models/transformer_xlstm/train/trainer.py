"""Training loop — arch-agnostic.

Builds the tokenizer, model (via registry), online-generator dataloader,
optimizer, LR schedule, mixed-precision context, and runs the multi-task
loss for `max_steps`. Logs to TensorBoard + (opportunistic) W&B.

Designed to fit one A100 at the planned sizes; bf16 throughout for speed.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer_xlstm.data.load import load_all_families, make_online_loader, make_static_loader
from transformer_xlstm.data.tokenizer import build_tokenizer
from transformer_xlstm.model.registry import build_model
from transformer_xlstm.train.losses import LossWeights, compute_losses
from transformer_xlstm.train.tracking import Tracker, TrackerConfig
from transformer_xlstm.utils.seed import set_seed

logger = logging.getLogger(__name__)


def cosine_lr(
    step: int, max_steps: int, warmup_steps: int, base_lr: float, min_lr_ratio: float = 0.1
) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def train(cfg: dict[str, Any]) -> dict[str, float]:
    """Train one cell. Returns final-step metrics."""
    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and cfg["train"]["precision"] in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if cfg["train"]["precision"] == "bf16" else torch.float16

    token_mode = cfg["tokenization"]["mode"]
    tokenizer = build_tokenizer(token_mode)
    logger.info(f"tokenizer mode={token_mode}  vocab_size={tokenizer.vocab_size}")

    enable_heads = cfg["loss"]["validity_weight"] > 0 or cfg["loss"]["rule_id_weight"] > 0
    model = build_model(
        cfg["arch"], cfg["model"], tokenizer.vocab_size, enable_multitask_heads=enable_heads
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model={cfg['arch']}  params={n_params:,}  multitask={enable_heads}")

    train_loader = make_online_loader(
        tokenizer,
        families=cfg["data"]["families"],
        batch_size=cfg["train"]["batch_size"],
        max_len=cfg["train"]["max_len"],
        corrupt_fraction=cfg["data"]["corrupt_fraction"],
        canonicalize=cfg["data"].get("canonicalize", False),
        family_dropout=cfg["train"]["family_dropout"],
        num_workers=cfg["data"].get("num_workers", 0),
        seed=cfg["train"]["seed"],
        ood_family_prob=cfg["data"].get("ood_family_prob", 0.0),
        synonym_randomize_prob=cfg["data"].get("synonym_randomize_prob", 0.0),
    )

    val_examples = load_all_families(families=cfg["data"]["families"])
    # Use last 100 per family as a tiny val set (deterministic order).
    val_subset = []
    for fam in cfg["data"]["families"]:
        per_fam = [e for e in val_examples if e.family == fam]
        val_subset.extend(per_fam[-100:])
    val_loader = make_static_loader(
        val_subset,
        tokenizer,
        batch_size=cfg["train"]["batch_size"],
        max_len=cfg["train"]["max_len"],
        shuffle=False,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        betas=(cfg["train"]["beta1"], cfg["train"]["beta2"]),
        weight_decay=cfg["train"]["weight_decay"],
    )

    weights = LossWeights(
        lm_weight=cfg["loss"]["lm_weight"],
        validity_weight=cfg["loss"]["validity_weight"],
        rule_id_weight=cfg["loss"]["rule_id_weight"],
    )

    run_name = cfg.get("run_name", f"{cfg['arch']}-{token_mode}-{int(time.time())}")
    tracker = Tracker(
        TrackerConfig(
            tensorboard_dir=cfg["tracking"]["tensorboard_dir"],
            wandb_project=cfg["tracking"].get("wandb_project"),
            wandb_entity=cfg["tracking"].get("wandb_entity"),
            wandb_mode=cfg["tracking"].get("wandb_mode", "opportunistic"),
            run_name=run_name,
        )
    )
    tracker.log_config(cfg)

    ckpt_dir = Path(cfg["out"]["checkpoint_dir"]) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_iter = iter(train_loader)

    final_metrics: dict[str, float] = {}
    step = 0
    t0 = time.time()
    while step < cfg["train"]["max_steps"]:
        lr = cosine_lr(
            step, cfg["train"]["max_steps"], cfg["train"]["warmup_steps"], cfg["train"]["lr"]
        )
        for g in optimizer.param_groups:
            g["lr"] = lr

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(batch["input_ids"], attn_mask=batch["attn_mask"])
            loss, log_dict = compute_losses(outputs, batch, weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        optimizer.step()

        if step % cfg["train"]["log_every"] == 0:
            elapsed = time.time() - t0
            sps = (step + 1) / elapsed if elapsed > 0 else 0
            metrics = {**log_dict, "lr": lr, "steps_per_sec": sps}
            tracker.log_metrics(metrics, step)
            logger.info(
                f"step={step}  loss={log_dict['total_loss']:.4f}  lr={lr:.2e}  sps={sps:.2f}"
            )

        if step > 0 and step % cfg["train"]["eval_every"] == 0:
            val_metrics = evaluate(model, val_loader, weights, device, amp_dtype, use_amp)
            tracker.log_metrics({f"val/{k}": v for k, v in val_metrics.items()}, step)
            logger.info(
                f"step={step}  val: " + " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            )

        if step > 0 and step % cfg["train"]["save_every"] == 0:
            save_checkpoint(model, optimizer, step, ckpt_dir / f"step_{step:06d}.pt", cfg)

        step += 1

    val_metrics = evaluate(model, val_loader, weights, device, amp_dtype, use_amp)
    tracker.log_metrics({f"val/{k}": v for k, v in val_metrics.items()}, step)
    save_checkpoint(model, optimizer, step, ckpt_dir / "final.pt", cfg)

    # Write a lightweight summary alongside the checkpoint. This is what we
    # commit to git as the official record; the heavy .pt stays on $SCRATCH
    # and gets rsynced down on demand.
    summary = {
        "run_name": run_name,
        "arch": cfg["arch"],
        "tokenization": cfg["tokenization"]["mode"],
        "model_params": n_params,
        "vocab_size": tokenizer.vocab_size,
        "max_steps": cfg["train"]["max_steps"],
        "final_step": step,
        "wall_seconds": time.time() - t0,
        "device": str(device),
        "precision": cfg["train"]["precision"],
        "final_metrics": val_metrics,
        "config": cfg,
    }
    with (ckpt_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"summary written → {ckpt_dir / 'summary.json'}")

    # Belt-and-braces redundancy: copy final.pt + summary.json to $HOME so
    # we still have the checkpoint if $SCRATCH ever has an issue.
    # $HOME has 50GB; 7 cells × ~600MB max well under quota.
    try:
        import os
        import shutil

        home_backup = Path(os.path.expanduser("~")) / "zero_one_hack_01_backup" / run_name
        home_backup.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ckpt_dir / "final.pt", home_backup / "final.pt")
        shutil.copy2(ckpt_dir / "summary.json", home_backup / "summary.json")
        logger.info(f"$HOME backup → {home_backup}")
    except OSError as e:
        # E.g. quota exceeded; not fatal.
        logger.warning(f"$HOME backup skipped: {e}")

    final_metrics = val_metrics

    tracker.close()
    return final_metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    weights: LossWeights,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    n_batches = 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(batch["input_ids"], attn_mask=batch["attn_mask"])
            _, log_dict = compute_losses(outputs, batch, weights)
        for k, v in log_dict.items():
            sums[k] = sums.get(k, 0.0) + v
        n_batches += 1
    model.train()
    return {k: v / max(1, n_batches) for k, v in sums.items()}


def save_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, step: int, path: Path, cfg: dict
) -> None:
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "config": cfg,
        },
        path,
    )
    logger.info(f"checkpoint saved → {path}")
