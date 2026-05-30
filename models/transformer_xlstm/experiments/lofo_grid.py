"""LoFO ablation grid — deterministic cell enumeration for the OOD sweep.

The grid is the answer to: "which recipe has the smallest ID->OOD drop?"
We hold out one family at a time, train on the other two, and measure the
held-out family's Top-K / NED / anomaly numbers. The recipe that wins this
sweep is the one we use to train the final all-three-family submission model.

Axes (Phase 1, 48 cells):
  arch       : transformer, xlstm
  size       : small, medium
  heads      : lm_only, multitask
  family_dp  : 0.0, 0.2
  fold       : lofo_mosfet, lofo_igbt, lofo_ic
Tokenization is fixed to compositional — the only mode that has a credible
story for unseen step strings in the hidden 4th family.

A separate 8-cell "final" sweep retrains the cross of {arch, size, heads, dp}
on all three families with no LoFO — used to produce the final submission
checkpoints once we know which recipe survives the LoFO comparison.

CLI:
  python -m transformer_xlstm.experiments.lofo_grid --list        # print id+overrides per cell
  python -m transformer_xlstm.experiments.lofo_grid --count       # number of cells
  python -m transformer_xlstm.experiments.lofo_grid --cell N      # print launch args for cell N
  python -m transformer_xlstm.experiments.lofo_grid --eval-cell N # print eval args for cell N
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[3]

ARCH_CFG = {
    ("transformer", "small"):  "configs/arch/transformer_small.yaml",
    ("transformer", "medium"): "configs/arch/transformer_medium.yaml",
    ("xlstm", "small"):        "configs/arch/xlstm_small.yaml",
    ("xlstm", "medium"):       "configs/arch/xlstm_medium.yaml",
}
TRAIN_CFG = {
    "lm_only":   "configs/train/default.yaml",
    "multitask": "configs/train/multitask.yaml",
}
TOKEN_CFG = "configs/token/compositional.yaml"

FAMILIES = ("mosfet", "igbt", "ic")


@dataclass
class Cell:
    """One training cell — fully parameterised launch."""

    id: str
    arch_cfg: str
    train_cfg: str
    token_cfg: str
    overrides: List[str] = field(default_factory=list)
    held_out: str | None = None       # None => trained on all three (final)
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


def _families_override(train_families: List[str]) -> str:
    return "data.families=[" + ",".join(train_families) + "]"


def lofo_cells() -> List[Cell]:
    """The 48-cell Phase-1 LoFO ablation."""
    cells: List[Cell] = []
    archs = ["transformer", "xlstm"]
    sizes = ["small", "medium"]
    heads = ["lm_only", "multitask"]
    fdps = [0.0, 0.2]
    folds = list(FAMILIES)

    for arch, size, head, fdp, held in product(archs, sizes, heads, fdps, folds):
        train_fams = [f for f in FAMILIES if f != held]
        overrides = [
            _families_override(train_fams),
            f"train.family_dropout={fdp}",
        ]
        cell_id = (
            f"lofo-{arch}-{size}-{head}-fdp{int(fdp*10):02d}-held_{held}"
        )
        cells.append(Cell(
            id=cell_id,
            arch_cfg=ARCH_CFG[(arch, size)],
            train_cfg=TRAIN_CFG[head],
            token_cfg=TOKEN_CFG,
            overrides=overrides,
            held_out=held,
            train_families=train_fams,
        ))
    return cells


def final_cells() -> List[Cell]:
    """8-cell final all-three sweep — same recipes, no held-out, full data."""
    cells: List[Cell] = []
    archs = ["transformer", "xlstm"]
    sizes = ["small", "medium"]
    heads = ["lm_only", "multitask"]
    fdps = [0.0, 0.2]

    for arch, size, head, fdp in product(archs, sizes, heads, fdps):
        overrides = [
            _families_override(list(FAMILIES)),
            f"train.family_dropout={fdp}",
        ]
        cell_id = f"final-{arch}-{size}-{head}-fdp{int(fdp*10):02d}-all3"
        cells.append(Cell(
            id=cell_id,
            arch_cfg=ARCH_CFG[(arch, size)],
            train_cfg=TRAIN_CFG[head],
            token_cfg=TOKEN_CFG,
            overrides=overrides,
            held_out=None,
            train_families=list(FAMILIES),
        ))
    return cells


def all_cells() -> List[Cell]:
    """LoFO + final, in dispatch order. Total 56 cells."""
    return lofo_cells() + final_cells()


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def _eval_args(cell: Cell) -> List[str]:
    """Args for `python -m transformer_xlstm.eval.run_eval`."""
    out = [
        "python", "-m", "transformer_xlstm.eval.run_eval",
        "--checkpoint", cell.checkpoint_path,
        "--output-dir", cell.eval_dir,
    ]
    if cell.held_out is not None:
        out.extend(["--held-out-family", cell.held_out])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true",
                   help="print all cells, one per line: id<TAB>train|eval command")
    g.add_argument("--count", action="store_true",
                   help="print total cell count")
    g.add_argument("--cell", type=int,
                   help="print launch shell command for cell N")
    g.add_argument("--eval-cell", type=int,
                   help="print run_eval shell command for cell N")
    g.add_argument("--cell-id", type=int,
                   help="print just the run id for cell N")
    g.add_argument("--held-out", type=int,
                   help="print held-out family for cell N (empty if final)")
    args = parser.parse_args()

    cells = all_cells()

    if args.count:
        print(len(cells))
        return
    if args.list:
        for i, c in enumerate(cells):
            print(f"{i}\t{c.id}\t{' '.join(c.launch_cmd)}")
        return
    if args.cell is not None:
        c = cells[args.cell]
        # Print one shell-quoted token per line — sbatch reads with `mapfile`.
        for tok in c.launch_cmd:
            print(tok)
        return
    if args.eval_cell is not None:
        c = cells[args.eval_cell]
        for tok in _eval_args(c):
            print(tok)
        return
    if args.cell_id is not None:
        print(cells[args.cell_id].id)
        return
    if args.held_out is not None:
        ho = cells[args.held_out].held_out or ""
        print(ho)
        return


if __name__ == "__main__":
    main()
