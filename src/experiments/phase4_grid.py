"""Phase-4 grid: stacks synonym-randomization aug on top of Phase-3 (OOD aug).

8 cells = transformer-multitask × {small, medium} × {3 LoFO + all3},
with synonym_randomize_prob=0.5 AND ood_family_prob=0.25.

Hypothesis: synonym randomization teaches the model that
STRIP RESIST ≡ STRIP PHOTORESIST (and ~25 other synonym pairs from §4
of the grammar). This directly targets Task 2 ExactMatch — the only
metric we haven't moved.
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
        out = ["python", "-m", "src.train.launch",
               "--arch-config", self.arch_cfg,
               "--train-config", self.train_cfg,
               "--token-config", self.token_cfg,
               "--run-name", self.id]
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
    for size, held in product(("small", "medium"), FAMILIES):
        train_fams = [f for f in FAMILIES if f != held]
        cells.append(Cell(
            id=f"v4-transformer-{size}-multitask-syn50-ood25-held_{held}",
            arch_cfg=ARCH_CFG[size],
            train_cfg=TRAIN_CFG,
            token_cfg=TOKEN_CFG,
            overrides=[
                _families_override(train_fams),
                "data.ood_family_prob=0.25",
                "data.synonym_randomize_prob=0.5",
            ],
            held_out=held,
            train_families=train_fams,
        ))
    for size in ("small", "medium"):
        cells.append(Cell(
            id=f"v4-final-transformer-{size}-multitask-syn50-ood25-all3",
            arch_cfg=ARCH_CFG[size],
            train_cfg=TRAIN_CFG,
            token_cfg=TOKEN_CFG,
            overrides=[
                _families_override(list(FAMILIES)),
                "data.ood_family_prob=0.25",
                "data.synonym_randomize_prob=0.5",
            ],
            held_out=None,
            train_families=list(FAMILIES),
        ))
    return cells


def _eval_args(c: Cell) -> List[str]:
    out = ["python", "-m", "src.eval.run_eval",
           "--checkpoint", c.checkpoint_path,
           "--output-dir", c.eval_dir,
           "--max-examples", "60", "--max-completion-steps", "60"]
    if c.held_out is not None:
        out.extend(["--held-out-family", c.held_out])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--count", action="store_true")
    g.add_argument("--cell", type=int)
    g.add_argument("--eval-cell", type=int)
    g.add_argument("--cell-id", type=int)
    args = p.parse_args()
    cells = all_cells()
    if args.count: print(len(cells)); return
    if args.list:
        for i, c in enumerate(cells):
            print(f"{i}\t{c.id}\t{' '.join(c.launch_cmd)}")
        return
    if args.cell is not None:
        for tok in cells[args.cell].launch_cmd:
            print(tok)
        return
    if args.eval_cell is not None:
        for tok in _eval_args(cells[args.eval_cell]):
            print(tok)
        return
    if args.cell_id is not None:
        print(cells[args.cell_id].id)
        return


if __name__ == "__main__":
    main()
