#!/usr/bin/env python3
"""
Self-supervised pretraining for semiconductor process sequences.

Task:
    Predict the next process step from previous steps.
    This is causal language modeling over symbolic semiconductor process tokens.

Expected input:
    data/generated/valid_long.csv

Expected columns:
    SEQUENCE_ID,FAMILY,STEP

Smoke test:
    python scripts/train_ssl_process_transformer.py \
      --data data/generated/valid_long.csv \
      --out-dir runs/ssl_smoke \
      --max-sequences 3000 \
      --epochs 2 \
      --batch-size 64 \
      --d-model 128 \
      --layers 3 \
      --heads 4

Larger run:
    python scripts/train_ssl_process_transformer.py \
      --data data/generated/valid_long.csv \
      --out-dir runs/ssl_process_transformer \
      --epochs 30 \
      --batch-size 512 \
      --d-model 256 \
      --layers 6 \
      --heads 8 \
      --lr 3e-4 \
      --family-dropout 0.25 \
      --input-step-mask-prob 0.03 \
      --class-weight inverse_sqrt
"""

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<MASK>"]
FAMILY_UNKNOWN = "<FAM_UNKNOWN>"


@dataclass
class Config:
    data: str = "data/generated/valid_long.csv"
    out_dir: str = "runs/ssl_process_transformer"
    seed: int = 42
    max_sequences: int = 0
    train_frac: float = 0.90
    val_frac: float = 0.05
    max_len: int = 192
    batch_size: int = 256
    epochs: int = 20
    d_model: int = 256
    layers: int = 6
    heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.15
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 500
    grad_clip: float = 1.0
    label_smoothing: float = 0.05
    family_dropout: float = 0.25
    input_step_mask_prob: float = 0.03
    class_weight: str = "inverse_sqrt"
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True


def resolve_data_path(path_str: str) -> Path:
    """
    Accept the current structure and the previous overnight/ structure.
    """
    candidates = [
        Path(path_str),
        Path("data/generated/valid_long.csv"),
        Path("data/generated/overnight/valid_long.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find valid_long.csv. Tried:\n"
        + "\n".join(f"  - {p}" for p in candidates)
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_sequences(path: Path, max_sequences: int = 0, seed: int = 42):
    df = pd.read_csv(path)
    required = {"SEQUENCE_ID", "FAMILY", "STEP"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}. Found: {list(df.columns)}")

    records = []
    for sid, g in df.groupby("SEQUENCE_ID", sort=False):
        family = str(g["FAMILY"].iloc[0]).upper()
        steps = [str(x) for x in g["STEP"].tolist()]
        if len(steps) >= 2:
            records.append({"sid": str(sid), "family": family, "steps": steps})

    rng = random.Random(seed)
    rng.shuffle(records)

    if max_sequences and max_sequences > 0:
        records = records[:max_sequences]

    return records


def split_records(records, train_frac: float, val_frac: float, seed: int):
    rng = random.Random(seed)

    by_family = {}
    for r in records:
        by_family.setdefault(r["family"], []).append(r)

    train, val, test = [], [], []
    for family, rs in sorted(by_family.items()):
        rng.shuffle(rs)
        n = len(rs)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train.extend(rs[:n_train])
        val.extend(rs[n_train:n_train + n_val])
        test.extend(rs[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def build_vocab(records):
    step_vocab = sorted({step for r in records for step in r["steps"]})
    token_to_id = {tok: i for i, tok in enumerate(SPECIAL_TOKENS + step_vocab)}
    id_to_token = {str(i): tok for tok, i in token_to_id.items()}

    families = sorted({r["family"] for r in records})
    family_to_id = {FAMILY_UNKNOWN: 0}
    for fam in families:
        family_to_id[fam] = len(family_to_id)

    return token_to_id, id_to_token, family_to_id


class ProcessDataset(Dataset):
    def __init__(self, records, token_to_id, family_to_id, max_len: int):
        self.records = records
        self.token_to_id = token_to_id
        self.family_to_id = family_to_id
        self.max_len = max_len

        self.pad_id = token_to_id["<PAD>"]
        self.bos_id = token_to_id["<BOS>"]
        self.eos_id = token_to_id["<EOS>"]
        self.unk_id = token_to_id["<UNK>"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]

        ids = [self.bos_id]
        ids += [self.token_to_id.get(step, self.unk_id) for step in r["steps"]]
        ids += [self.eos_id]

        if len(ids) > self.max_len:
            ids = ids[:self.max_len]

        x = ids[:-1]
        y = ids[1:]

        family_id = self.family_to_id.get(r["family"], self.family_to_id[FAMILY_UNKNOWN])

        return {
            "input_ids": torch.tensor(x, dtype=torch.long),
            "target_ids": torch.tensor(y, dtype=torch.long),
            "family_id": torch.tensor(family_id, dtype=torch.long),
        }


class Collator:
    def __init__(
        self,
        pad_id: int,
        mask_id: int,
        bos_id: int,
        family_unknown_id: int,
        family_dropout: float,
        input_step_mask_prob: float,
        train: bool,
    ):
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.bos_id = bos_id
        self.family_unknown_id = family_unknown_id
        self.family_dropout = family_dropout
        self.input_step_mask_prob = input_step_mask_prob
        self.train = train

    def __call__(self, batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        bsz = len(batch)

        input_ids = torch.full((bsz, max_len), self.pad_id, dtype=torch.long)
        target_ids = torch.full((bsz, max_len), self.pad_id, dtype=torch.long)
        family_ids = torch.stack([item["family_id"] for item in batch])

        for i, item in enumerate(batch):
            length = len(item["input_ids"])
            input_ids[i, :length] = item["input_ids"]
            target_ids[i, :length] = item["target_ids"]

        # Research-grounded OOD regularizer:
        # sometimes hide the product family so the model cannot overfit to family labels.
        if self.train and self.family_dropout > 0:
            drop = torch.rand_like(family_ids.float()) < self.family_dropout
            family_ids = family_ids.clone()
            family_ids[drop] = self.family_unknown_id

        # Light denoising/noising:
        # randomly mask some input steps while still predicting the true next step.
        if self.train and self.input_step_mask_prob > 0:
            can_mask = (input_ids != self.pad_id) & (input_ids != self.bos_id)
            mask = (torch.rand_like(input_ids.float()) < self.input_step_mask_prob) & can_mask
            input_ids = input_ids.clone()
            input_ids[mask] = self.mask_id

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "family_ids": family_ids,
        }


class ProcessTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_families: int,
        max_len: int,
        d_model: int,
        layers: int,
        heads: int,
        ff_mult: int,
        dropout: float,
        pad_id: int,
    ):
        super().__init__()

        self.pad_id = pad_id
        self.step_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.family_emb = nn.Embedding(num_families, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.step_emb.weight  # weight tying

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, family_ids):
        bsz, seqlen = input_ids.shape
        device = input_ids.device

        positions = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, seqlen)

        x = self.step_emb(input_ids)
        x = x + self.family_emb(family_ids).unsqueeze(1)
        x = x + self.pos_emb(positions)
        x = self.dropout(x)

        causal_mask = torch.triu(
            torch.ones(seqlen, seqlen, device=device, dtype=torch.bool),
            diagonal=1,
        )
        pad_mask = input_ids.eq(self.pad_id)

        h = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=pad_mask,
        )

        h = self.norm(h)
        logits = self.lm_head(h)
        return logits
    
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].fill_(0.0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def make_class_weights(records, token_to_id, max_len: int, mode: str):
    pad_id = token_to_id["<PAD>"]
    bos_id = token_to_id["<BOS>"]
    eos_id = token_to_id["<EOS>"]
    unk_id = token_to_id["<UNK>"]

    vocab_size = len(token_to_id)
    counts = torch.ones(vocab_size, dtype=torch.float32)

    for r in records:
        ids = [bos_id]
        ids += [token_to_id.get(step, unk_id) for step in r["steps"]]
        ids += [eos_id]
        if len(ids) > max_len:
            ids = ids[:max_len]

        targets = ids[1:]
        for t in targets:
            counts[t] += 1.0

    if mode == "none":
        weights = torch.ones(vocab_size, dtype=torch.float32)
    elif mode == "inverse_sqrt":
        weights = counts.pow(-0.5)
        valid = torch.ones(vocab_size, dtype=torch.bool)
        valid[pad_id] = False
        weights = weights / weights[valid].mean()
        weights = torch.clamp(weights, min=0.25, max=4.0)
    else:
        raise ValueError(f"Unknown class_weight mode: {mode}")

    weights[pad_id] = 0.0
    return weights


def lr_schedule(step: int, base_lr: float, warmup_steps: int, total_steps: int):
    if step <= warmup_steps:
        return base_lr * step / max(1, warmup_steps)

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, criterion, device, pad_id):
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    correct1 = 0
    correct3 = 0
    correct5 = 0
    rr_sum = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        target_ids = batch["target_ids"].to(device, non_blocking=True)
        family_ids = batch["family_ids"].to(device, non_blocking=True)

        logits = model(input_ids, family_ids)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
        )

        valid = target_ids.ne(pad_id)
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            continue

        total_loss += float(loss.item()) * n_valid
        total_tokens += n_valid

        k = min(5, logits.size(-1))
        topk = logits.topk(k=k, dim=-1).indices
        target_expanded = target_ids.unsqueeze(-1)

        correct1 += int(((topk[:, :, :1] == target_expanded).any(dim=-1) & valid).sum().item())
        correct3 += int(((topk[:, :, :min(3, k)] == target_expanded).any(dim=-1) & valid).sum().item())
        correct5 += int(((topk == target_expanded).any(dim=-1) & valid).sum().item())

        # Exact rank for MRR. Vocab is small, so full sort is fine.
        flat_logits = logits[valid]
        flat_targets = target_ids[valid]
        order = flat_logits.argsort(dim=-1, descending=True)
        match = order.eq(flat_targets.unsqueeze(1))
        ranks = match.float().argmax(dim=1).float() + 1.0
        rr_sum += float((1.0 / ranks).sum().item())

    loss_avg = total_loss / max(1, total_tokens)

    return {
        "loss": loss_avg,
        "ppl": math.exp(min(20.0, loss_avg)),
        "top1": correct1 / max(1, total_tokens),
        "top3": correct3 / max(1, total_tokens),
        "top5": correct5 / max(1, total_tokens),
        "mrr": rr_sum / max(1, total_tokens),
        "tokens": total_tokens,
    }


def save_checkpoint(path, model, optimizer, cfg, token_to_id, family_to_id, epoch, best_val_loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(cfg),
            "token_to_id": token_to_id,
            "family_to_id": family_to_id,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="data/generated/valid_long.csv")
    parser.add_argument("--out-dir", default="runs/ssl_process_transformer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sequences", type=int, default=0)

    parser.add_argument("--train-frac", type=float, default=0.90)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--max-len", type=int, default=192)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--family-dropout", type=float, default=0.25)
    parser.add_argument("--input-step-mask-prob", type=float, default=0.03)
    parser.add_argument("--class-weight", choices=["none", "inverse_sqrt"], default="inverse_sqrt")

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")

    args = parser.parse_args()

    data_path = resolve_data_path(args.data)

    cfg = Config(
        data=str(data_path),
        out_dir=args.out_dir,
        seed=args.seed,
        max_sequences=args.max_sequences,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        max_len=args.max_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        label_smoothing=args.label_smoothing,
        family_dropout=args.family_dropout,
        input_step_mask_prob=args.input_step_mask_prob,
        class_weight=args.class_weight,
        num_workers=args.num_workers,
        device=args.device,
        amp=not args.no_amp,
    )

    set_seed(cfg.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading sequences...")
    print(f"Using data file: {data_path}")

    records = load_sequences(data_path, max_sequences=cfg.max_sequences, seed=cfg.seed)
    train_records, val_records, test_records = split_records(
        records,
        cfg.train_frac,
        cfg.val_frac,
        cfg.seed,
    )

    print(f"Total sequences: {len(records):,}")
    print(f"Train/val/test: {len(train_records):,}/{len(val_records):,}/{len(test_records):,}")

    family_counts = {}
    for r in records:
        family_counts[r["family"]] = family_counts.get(r["family"], 0) + 1
    print(f"Family counts: {family_counts}")

    print("[2/6] Building vocabulary...")
    token_to_id, id_to_token, family_to_id = build_vocab(records)

    pad_id = token_to_id["<PAD>"]
    mask_id = token_to_id["<MASK>"]
    bos_id = token_to_id["<BOS>"]

    print(f"Vocab size including specials: {len(token_to_id)}")
    print(f"Families: {family_to_id}")

    with (out_dir / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "token_to_id": token_to_id,
                "id_to_token": id_to_token,
                "family_to_id": family_to_id,
                "config": asdict(cfg),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("[3/6] Creating datasets/loaders...")
    train_ds = ProcessDataset(train_records, token_to_id, family_to_id, cfg.max_len)
    val_ds = ProcessDataset(val_records, token_to_id, family_to_id, cfg.max_len)
    test_ds = ProcessDataset(test_records, token_to_id, family_to_id, cfg.max_len)

    train_collator = Collator(
        pad_id=pad_id,
        mask_id=mask_id,
        bos_id=bos_id,
        family_unknown_id=family_to_id[FAMILY_UNKNOWN],
        family_dropout=cfg.family_dropout,
        input_step_mask_prob=cfg.input_step_mask_prob,
        train=True,
    )

    eval_collator = Collator(
        pad_id=pad_id,
        mask_id=mask_id,
        bos_id=bos_id,
        family_unknown_id=family_to_id[FAMILY_UNKNOWN],
        family_dropout=0.0,
        input_step_mask_prob=0.0,
        train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=train_collator,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=eval_collator,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=eval_collator,
        pin_memory=torch.cuda.is_available(),
    )

    print("[4/6] Building model...")
    if cfg.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg.device

    model = ProcessTransformerLM(
    vocab_size=len(token_to_id),
    num_families=len(family_to_id),
    max_len=cfg.max_len,
    d_model=cfg.d_model,
    layers=cfg.layers,
    heads=cfg.heads,
    ff_mult=cfg.ff_mult,
    dropout=cfg.dropout,
    pad_id=pad_id,
)

    model.apply(init_weights)

    model.lm_head.weight = model.step_emb.weight

    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Parameters: {n_params:,}")

    class_weights = make_class_weights(
        train_records,
        token_to_id,
        cfg.max_len,
        cfg.class_weight,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_id,
        weight=class_weights,
        label_smoothing=cfg.label_smoothing,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    total_steps = cfg.epochs * max(1, len(train_loader))
    use_amp = cfg.amp and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    metrics_path = out_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "split",
            "loss",
            "ppl",
            "top1",
            "top3",
            "top5",
            "mrr",
            "tokens",
            "lr",
            "seconds",
        ])

    print("[5/6] Training...")
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        model.train()
        train_loss_sum = 0.0
        train_tokens = 0

        for batch in train_loader:
            global_step += 1

            lr = lr_schedule(global_step, cfg.lr, cfg.warmup_steps, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            target_ids = batch["target_ids"].to(device, non_blocking=True)
            family_ids = batch["family_ids"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(input_ids, family_ids)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1),
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            n_tokens = int(target_ids.ne(pad_id).sum().item())
            train_loss_sum += float(loss.item()) * n_tokens
            train_tokens += n_tokens

        train_loss = train_loss_sum / max(1, train_tokens)
        train_metrics = {
            "loss": train_loss,
            "ppl": math.exp(min(20.0, train_loss)),
            "top1": float("nan"),
            "top3": float("nan"),
            "top5": float("nan"),
            "mrr": float("nan"),
            "tokens": train_tokens,
        }

        val_metrics = evaluate(model, val_loader, criterion, device, pad_id)
        seconds = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | "
            f"val top1 {val_metrics['top1']:.4f} | "
            f"val top3 {val_metrics['top3']:.4f} | "
            f"val top5 {val_metrics['top5']:.4f} | "
            f"val mrr {val_metrics['mrr']:.4f} | "
            f"{seconds:.1f}s"
        )

        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for split, metrics in [("train", train_metrics), ("val", val_metrics)]:
                writer.writerow([
                    epoch,
                    split,
                    metrics["loss"],
                    metrics["ppl"],
                    metrics["top1"],
                    metrics["top3"],
                    metrics["top5"],
                    metrics["mrr"],
                    metrics["tokens"],
                    current_lr,
                    seconds,
                ])

        save_checkpoint(
            out_dir / "checkpoint_last.pt",
            model,
            optimizer,
            cfg,
            token_to_id,
            family_to_id,
            epoch,
            best_val_loss,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                out_dir / "checkpoint_best.pt",
                model,
                optimizer,
                cfg,
                token_to_id,
                family_to_id,
                epoch,
                best_val_loss,
            )
            print(f"  saved new best checkpoint with val loss {best_val_loss:.4f}")

    print("[6/6] Final test evaluation...")
    test_metrics = evaluate(model, test_loader, criterion, device, pad_id)

    print(
        f"TEST | "
        f"loss {test_metrics['loss']:.4f} | "
        f"top1 {test_metrics['top1']:.4f} | "
        f"top3 {test_metrics['top3']:.4f} | "
        f"top5 {test_metrics['top5']:.4f} | "
        f"mrr {test_metrics['mrr']:.4f}"
    )

    with metrics_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            cfg.epochs,
            "test",
            test_metrics["loss"],
            test_metrics["ppl"],
            test_metrics["top1"],
            test_metrics["top3"],
            test_metrics["top5"],
            test_metrics["mrr"],
            test_metrics["tokens"],
            optimizer.param_groups[0]["lr"],
            0.0,
        ])

    print(f"Done. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()