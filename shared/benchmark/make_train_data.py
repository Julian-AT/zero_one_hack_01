#!/usr/bin/env python3
"""
make_train_data.py — generate FAMILY-tagged training sequences for the SSL hybrid
(and any model that wants a static training CSV), with a *training* seed that is
DISJOINT from the benchmark eval seeds (90001+) so the common eval set stays held out.

The SSL trainer (`train_ssl_hybrid_process_transformer.py`) requires long-format
`SEQUENCE_ID, FAMILY, STEP`. generate_sequences.py emits `SEQUENCE_ID, STEP`, so we
tag the family and concatenate per regime.

Writes (default --out-dir shared/benchmark/_train_data):
    train_all3.csv                 mosfet + igbt + ic
    train_held_mosfet.csv          igbt + ic   (LoFO: mosfet held out)
    train_held_igbt.csv            mosfet + ic
    train_held_ic.csv              mosfet + igbt
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "competition" / "track-details" / "training_data" / "generate_sequences.py"
FAMILIES = ["mosfet", "igbt", "ic"]
TRAIN_SEEDS = {"mosfet": 42001, "igbt": 42002, "ic": 42003}  # disjoint from eval 90001+


def gen_family(fam: str, count: int, raw_dir: Path, py: str) -> Path:
    out = raw_dir / f"{fam}.csv"
    subprocess.run([py, str(GEN), "--family", fam, "--count", str(count),
                    "--seed", str(TRAIN_SEEDS[fam]), "--output", str(out)],
                   check=True, cwd=ROOT)
    return out


def read_long(path: Path, fam: str) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append({"SEQUENCE_ID": f"{fam}_{r['SEQUENCE_ID']}",
                         "FAMILY": fam.upper(), "STEP": r["STEP"]})
    return rows


def write_long(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["SEQUENCE_ID", "FAMILY", "STEP"])
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} step-rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="shared/benchmark/_train_data")
    ap.add_argument("--count", type=int, default=1500, help="sequences per family")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    out = (ROOT / a.out_dir).resolve()
    raw = out / "_raw"
    raw.mkdir(parents=True, exist_ok=True)

    per_fam: dict[str, list[dict]] = {}
    for fam in FAMILIES:
        p = gen_family(fam, a.count, raw, a.python)
        per_fam[fam] = read_long(p, fam)

    write_long(out / "train_all3.csv", [r for fam in FAMILIES for r in per_fam[fam]])
    for held in FAMILIES:
        keep = [f for f in FAMILIES if f != held]
        write_long(out / f"train_held_{held}.csv", [r for f in keep for r in per_fam[f]])
    print("DONE.")


if __name__ == "__main__":
    main()
