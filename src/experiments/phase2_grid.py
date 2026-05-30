"""Phase-2 LoFO grid — same structure as phase 1, but with:

- max_len=768 fix (configs already updated)
- xLSTM dropped (3-4× slower, no quality benefit per phase-1 finding)
- family_dropout axis dropped (redundant with multitask heads per phase-1)
- cell ids prefixed `v2-` so checkpoints don't collide with phase-1

Resulting grid is 16 cells:
  arch=transformer × size{small, medium} × heads{lm_only, multitask}
    × fold{held_mosfet, held_igbt, held_ic, all3}

CLI mirrors lofo_grid.py.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]

ARCH_CFG = {
    "small":  "configs/arch/transformer_small.yaml",
    "medium": "configs/arch/transformer_medium.yaml",
}
TRAIN_CFG = {
    "lm_only":   "configs/train/default.yaml",
    "multitask": "configs/train/multitask.yaml",
}
TOKEN_CFG = "configs/token/compositional.yaml"
FAMILIES = ("mosfet", "igbt", "ic")


@dataclass
class Cell:
    id: str
    arch_cfg: str
    train_cfg: str
    token_cfg: str
    overrides: List[str] = field(default_factory=list)
    held_out: str | None = None
    train_families: List[str] = field(default_factory=lambda: list(FAMILIES))

    @property
    def launch_cmd(self) -> List[str]:
        out = [
            "python", "-m", "src.train.launch",
            "--arch-config", self.arch_cfg,
            "--train-config", self.train_cfg,
            "--token-config", self.token_cfg,
            "--run-name", self.id,
        ]
        if self.overrides:
            out.append("--override")
            out.extend(self.overrides)
        return out

    @property
    def checkpoint_path(self) -> str:
        return f"extras/checkpoints/{self.id}/final.pt"

    @property
    def eval_dir(self) -> str:
        return f"extras/results/eval/{self.id}"


def _families_override(fams: List[str]) -> str:
    return "data.families=[" + ",".join(fams) + "]"


def all_cells() -> List[Cell]:
    cells: List[Cell] = []
    # LoFO cells (12)
    for size, heads, held in product(("small", "medium"), ("lm_only", "multitask"),
                                       FAMILIES):
        train_fams = [f for f in FAMILIES if f != held]
        cells.append(Cell(
            id=f"v2-transformer-{size}-{heads}-held_{held}",
            arch_cfg=ARCH_CFG[size],
            train_cfg=TRAIN_CFG[heads],
            token_cfg=TOKEN_CFG,
            overrides=[_families_override(train_fams)],
            held_out=held,
            train_families=train_fams,
        ))
    # Final all-3 cells (4)
    for size, heads in product(("small", "medium"), ("lm_only", "multitask")):
        cells.append(Cell(
            id=f"v2-final-transformer-{size}-{heads}-all3",
            arch_cfg=ARCH_CFG[size],
            train_cfg=TRAIN_CFG[heads],
            token_cfg=TOKEN_CFG,
            overrides=[_families_override(list(FAMILIES))],
            held_out=None,
            train_families=list(FAMILIES),
        ))
    return cells


def _eval_args(c: Cell) -> List[str]:
    out = [
        "python", "-m", "src.eval.run_eval",
        "--checkpoint", c.checkpoint_path,
        "--output-dir", c.eval_dir,
        "--max-examples", "60",
        "--max-completion-steps", "60",
    ]
    if c.held_out is not None:
        out.extend(["--held-out-family", c.held_out])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--count", action="store_true")
    g.add_argument("--cell", type=int)
    g.add_argument("--eval-cell", type=int)
    g.add_argument("--cell-id", type=int)
    args = parser.parse_args()

    cells = all_cells()
    if args.count:
        print(len(cells)); return
    if args.list:
        for i, c in enumerate(cells):
            print(f"{i}\t{c.id}\t{' '.join(c.launch_cmd)}")
        return
    if args.cell is not None:
        for tok in cells[args.cell].launch_cmd: print(tok)
        return
    if args.eval_cell is not None:
        for tok in _eval_args(cells[args.eval_cell]): print(tok)
        return
    if args.cell_id is not None:
        print(cells[args.cell_id].id); return


if __name__ == "__main__":
    main()
