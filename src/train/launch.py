"""CLI: launch one training cell from YAML configs.

Usage:
    python -m src.train.launch \
        --arch-config configs/arch/transformer_small.yaml \
        --train-config configs/train/default.yaml \
        --token-config configs/token/compositional.yaml \
        --run-name my-run-001

Reads, merges via OmegaConf, runs `src.train.trainer.train()`.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

from src.train.trainer import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    stream=sys.stdout,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a training cell.")
    parser.add_argument("--arch-config", required=True,
                        help="YAML with arch + model settings")
    parser.add_argument("--train-config", default="configs/train/default.yaml",
                        help="YAML with train/data/loss/tracking settings")
    parser.add_argument("--token-config", default="configs/token/compositional.yaml",
                        help="YAML with tokenization mode")
    parser.add_argument("--run-name", default=None,
                        help="Run identifier (default: derived from configs + timestamp)")
    parser.add_argument("--override", nargs="*", default=[],
                        help="OmegaConf-style overrides, e.g. train.max_steps=100")
    args = parser.parse_args()

    arch_cfg = OmegaConf.load(args.arch_config)
    train_cfg = OmegaConf.load(args.train_config)
    token_cfg = OmegaConf.load(args.token_config)
    merged = OmegaConf.merge(arch_cfg, train_cfg, token_cfg)
    if args.override:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(args.override))
    if args.run_name:
        merged["run_name"] = args.run_name

    cfg_dict = OmegaConf.to_container(merged, resolve=True)
    final_metrics = train(cfg_dict)
    print("FINAL METRICS:", final_metrics)


if __name__ == "__main__":
    main()
