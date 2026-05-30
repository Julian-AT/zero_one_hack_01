"""Phase-3 grid: same as Phase-2 multitask, with ood_family_prob=0.25.

Tests whether OOD-family augmentation (DIODE / SCHOTTKY / SIC_MOSFET drawn
into the training stream as <FAMILY_UNK>-tagged) lifts held-out Top-1 over
the Phase-2 baseline (which trains only on MOSFET / IGBT / IC).

Grid: 8 cells = arch=transformer × size{small, medium} × multitask only
                × fold{held_mosfet, held_igbt, held_ic, all3}
                × ood_family_prob=0.25

Compare Phase-3 cells against the matching Phase-2 cells (which had
ood_family_prob=0.0). If Phase-3 wins on held-out Top-1, OOD augmentation
becomes the recipe for the final submission.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from itertools import product
from typing import List

ARCH_CFG = {
    "small":  "configs/arch/transformer_small.yaml",
    "medium": "configs/arch/transformer_medium.yaml",
}
TRAIN_CFG = "configs/train/multitask.yaml"
TOKEN_CFG = "configs/token/compositional.yaml"
FAMILIES = ("mosfet", "igbt", "ic")
OOD_PROB = 0.25


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
            "python", "-m", "transformer_xlstm.train.launch",
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
        return f"shared/extras/checkpoints/{self.id}/final.pt"

    @property
    def eval_dir(self) -> str:
        return f"shared/extras/results/eval/{self.id}"


def _families_override(fams: List[str]) -> str:
    return "data.families=[" + ",".join(fams) + "]"


def all_cells() -> List[Cell]:
    cells: List[Cell] = []
    for size, held in product(("small", "medium"), FAMILIES):
        train_fams = [f for f in FAMILIES if f != held]
        cells.append(Cell(
            id=f"v3-transformer-{size}-multitask-ood25-held_{held}",
            arch_cfg=ARCH_CFG[size],
            train_cfg=TRAIN_CFG,
            token_cfg=TOKEN_CFG,
            overrides=[
                _families_override(train_fams),
                f"data.ood_family_prob={OOD_PROB}",
            ],
            held_out=held,
            train_families=train_fams,
        ))
    for size in ("small", "medium"):
        cells.append(Cell(
            id=f"v3-final-transformer-{size}-multitask-ood25-all3",
            arch_cfg=ARCH_CFG[size],
            train_cfg=TRAIN_CFG,
            token_cfg=TOKEN_CFG,
            overrides=[
                _families_override(list(FAMILIES)),
                f"data.ood_family_prob={OOD_PROB}",
            ],
            held_out=None,
            train_families=list(FAMILIES),
        ))
    return cells


def _eval_args(c: Cell) -> List[str]:
    out = [
        "python", "-m", "transformer_xlstm.eval.run_eval",
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
    if args.count: print(len(cells)); return
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
    if args.cell_id is not None: print(cells[args.cell_id].id); return


if __name__ == "__main__":
    main()
