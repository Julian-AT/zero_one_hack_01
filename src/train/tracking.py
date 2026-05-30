"""Experiment tracking: TensorBoard always, Weights & Biases opportunistically.

W&B is enabled if:
  - tracking.wandb_mode is "online" or "opportunistic"
  - `wandb` is importable
  - either WANDB_API_KEY is set or wandb has been logged in

If anything fails, we fall back silently to TB-only and log a warning.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


@dataclass
class TrackerConfig:
    tensorboard_dir: str = "extras/logs/tb"
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_mode: str = "opportunistic"  # online | offline | opportunistic | disabled
    run_name: str | None = None


class Tracker:
    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        run_name = cfg.run_name or "run"
        tb_path = Path(cfg.tensorboard_dir) / run_name
        tb_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tb_path))
        self.wandb_run = None

        if cfg.wandb_mode != "disabled":
            self._try_init_wandb()

    def _try_init_wandb(self) -> None:
        try:
            import wandb
        except ImportError:
            logger.info("wandb not installed; falling back to TB only.")
            return
        # Opportunistic: only if API key present
        has_key = bool(os.environ.get("WANDB_API_KEY")) or Path("~/.netrc").expanduser().exists()
        if self.cfg.wandb_mode == "opportunistic" and not has_key:
            logger.info("W&B opportunistic mode: no API key found; TB only.")
            return
        try:
            self.wandb_run = wandb.init(
                project=self.cfg.wandb_project,
                entity=self.cfg.wandb_entity,
                name=self.cfg.run_name,
                mode="online" if self.cfg.wandb_mode == "online" else "offline"
                     if self.cfg.wandb_mode == "offline" else "online",
                reinit=True,
            )
        except Exception as e:
            logger.warning(f"W&B init failed: {e}. TB only.")
            self.wandb_run = None

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)
        if self.wandb_run is not None:
            try:
                self.wandb_run.log(metrics, step=step)
            except Exception as e:
                logger.warning(f"W&B log failed: {e}")
                self.wandb_run = None

    def log_text(self, tag: str, text: str, step: int) -> None:
        self.writer.add_text(tag, text, step)
        if self.wandb_run is not None:
            try:
                import wandb
                self.wandb_run.log({tag: wandb.Html(f"<pre>{text}</pre>")}, step=step)
            except Exception:
                pass

    def log_config(self, cfg: dict[str, Any]) -> None:
        import json
        text = json.dumps(cfg, indent=2, default=str)
        self.writer.add_text("config", text, 0)
        if self.wandb_run is not None:
            try:
                self.wandb_run.config.update(cfg, allow_val_change=True)
            except Exception:
                pass

    def close(self) -> None:
        self.writer.close()
        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()
            except Exception:
                pass
