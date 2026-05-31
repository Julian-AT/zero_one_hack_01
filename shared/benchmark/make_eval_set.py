#!/usr/bin/env python3
"""
make_eval_set.py — build the *common, frozen, labeled* benchmark eval set used to
compare all three model approaches (transformer_xlstm, self-supervised SSL hybrid,
neurosymbolic) plus the n-gram / grammar / retrieval baselines on identical data and
identical metrics.

Everything is generated from the committed rule generator
(`competition/track-details/training_data/generate_sequences.py`) with a
**benchmark-only seed disjoint from every training seed**, so the eval set is
genuinely held out for every model and fully reproducible from a clean checkout.

Pipeline
--------
1. generate_sequences.py          -> fresh VALID sequences per family (benchmark seed)
2. generate_invalid_sequences.py  -> EASY invalid (near-miss rule violations)
   generate_hard_invalid_sequences.py -> HARD invalid (late, subtle violations)
3. build_task_datasets.py         -> labeled next-step / completion / anomaly task CSVs
                                     with an 80/10/10 SPLIT column
4. prepare_eval_inputs.py --split test -> organizer-format inputs/ + ground_truth/
5. derive per-family subsets in by_family/<fam>/ for the Leave-One-Family-Out (LoFO)
   OOD regime (model trained on the other two families is scored on the held-out one).

Output layout (default --out-dir shared/benchmark/eval_set_v1)
    inputs/        {nextstep,completion,anomaly}_input.csv   (feed to every model)
    ground_truth/  {nextstep,completion,anomaly}_gt.csv      (for scoring)
    by_family/<fam>/inputs/        per-family inputs   (LoFO held-out family)
    by_family/<fam>/ground_truth/  per-family labels
    MANIFEST.json  exact parameters + row counts (provenance)

Usage
-----
    python shared/benchmark/make_eval_set.py            # default sizes
    python shared/benchmark/make_eval_set.py --valid-count 400 --completion-sample 30
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "competition" / "track-details" / "training_data" / "generate_sequences.py"
GEN_INVALID = ROOT / "competition" / "track-details" / "data" / "generate_invalid_sequences.py"
GEN_HARD = ROOT / "competition" / "track-details" / "data" / "generate_hard_invalid_sequences.py"
BUILD = ROOT / "competition" / "track-details" / "data" / "build_task_datasets.py"
PREPARE = ROOT / "shared" / "benchmark" / "prepare_eval_inputs.py"

FAMILIES = ["mosfet", "igbt", "ic"]
# benchmark-only seeds (disjoint from training seed 42 and any other run seed)
VALID_SEEDS = {"mosfet": 90001, "igbt": 90002, "ic": 90003}
EASY_SEED = 90010
HARD_SEED = 90011
SPLIT_SEED = 90042  # split seed for build_task_datasets (disjoint from training 42)


def run(cmd: list[str]) -> None:
    print("    $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT)


def find_invalid_long(out_dir: Path) -> Path:
    """Locate the long-format invalid CSV (one with STEP + VIOLATED_RULE columns)."""
    for name in ("invalid_sequences.csv", "hard_invalid_sequences.csv"):
        p = out_dir / name
        if p.exists():
            return p
    # fallback: any csv with a STEP column
    for p in sorted(out_dir.glob("*.csv")):
        with p.open(newline="", encoding="utf-8-sig") as f:
            header = next(csv.reader(f), [])
        up = {h.strip().upper() for h in header}
        if "STEP" in up and ("VIOLATED_RULE" in up or "IS_VALID" in up):
            return p
    raise FileNotFoundError(f"no long-format invalid CSV found in {out_dir}")


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, header: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def rebuild_nextstep_from_completion(out_dir: Path) -> int:
    """Replace the generic (all-position) next-step eval with the ORGANIZER protocol:
    next-step is predicted at the 60% / 80% truncation points. We derive it from the
    completion cut points — input = the partial at the cut, truth = the first step of the
    remaining suffix. This matches eval_input_valid.csv (Task 1) and gives every model the
    same long-context prefix instead of penalizing context-using models on 1-step prefixes."""
    comp = read_rows(out_dir / "ground_truth" / "completion_gt.csv")
    comp_in = {r["EXAMPLE_ID"]: r for r in read_rows(out_dir / "inputs" / "completion_input.csv")}
    ns_in, ns_gt = [], []
    for r in comp:
        eid = r["EXAMPLE_ID"]
        partial = [s for s in r["PARTIAL_SEQUENCE"].split("|") if s]
        full = [s for s in r["FULL_SEQUENCE"].split("|") if s]
        if len(full) <= len(partial):
            continue
        nxt = full[len(partial)]
        frac = comp_in.get(eid, {}).get("COMPLETION_FRACTION", "")
        neid = "ns_" + eid
        ns_in.append({"EXAMPLE_ID": neid, "FAMILY": r["FAMILY"], "COMPLETION_FRACTION": frac,
                      "PARTIAL_SEQUENCE": r["PARTIAL_SEQUENCE"]})
        ns_gt.append({"EXAMPLE_ID": neid, "FAMILY": r["FAMILY"], "NEXT_STEP": nxt})
    write_rows(out_dir / "inputs" / "nextstep_input.csv",
               ["EXAMPLE_ID", "FAMILY", "COMPLETION_FRACTION", "PARTIAL_SEQUENCE"], ns_in)
    write_rows(out_dir / "ground_truth" / "nextstep_gt.csv",
               ["EXAMPLE_ID", "FAMILY", "NEXT_STEP"], ns_gt)
    print(f"    next-step rebuilt at 60/80% cut points -> {len(ns_in)} examples (organizer protocol)")
    return len(ns_in)


def subsample_completion(out_dir: Path, per_family: int) -> None:
    """Cap completion inputs/gt to `per_family` examples per family (greedy decoding is
    the slow path; NED/EM/block-acc are stable at ~30/family). next-step + anomaly keep
    all rows. Subsampling is deterministic (first-N per family by EXAMPLE_ID order)."""
    if per_family <= 0:
        return
    inp = out_dir / "inputs" / "completion_input.csv"
    gt = out_dir / "ground_truth" / "completion_gt.csv"
    inp_rows = read_rows(inp)
    gt_rows = read_rows(gt)
    keep: set[str] = set()
    seen: dict[str, int] = {f: 0 for f in [r.get("FAMILY", "") for r in inp_rows]}
    for r in inp_rows:
        fam = r.get("FAMILY", "")
        if seen.get(fam, 0) < per_family:
            keep.add(r["EXAMPLE_ID"])
            seen[fam] = seen.get(fam, 0) + 1
    write_rows(inp, list(inp_rows[0].keys()), [r for r in inp_rows if r["EXAMPLE_ID"] in keep])
    write_rows(gt, list(gt_rows[0].keys()), [r for r in gt_rows if r["EXAMPLE_ID"] in keep])
    print(f"    completion subsampled to <= {per_family}/family -> {len(keep)} rows")


def derive_by_family(out_dir: Path) -> dict[str, dict[str, int]]:
    """Split inputs/ and ground_truth/ by FAMILY into by_family/<fam>/ for LoFO."""
    tasks = {
        "nextstep": ("inputs/nextstep_input.csv", "ground_truth/nextstep_gt.csv"),
        "completion": ("inputs/completion_input.csv", "ground_truth/completion_gt.csv"),
        "anomaly": ("inputs/anomaly_input.csv", "ground_truth/anomaly_gt.csv"),
    }
    counts: dict[str, dict[str, int]] = {f: {} for f in FAMILIES}
    for fam in FAMILIES:
        for task, (in_rel, gt_rel) in tasks.items():
            for rel in (in_rel, gt_rel):
                src = out_dir / rel
                rows = read_rows(src)
                fam_rows = [r for r in rows if (r.get("FAMILY", "") or "").lower() == fam]
                dst = out_dir / "by_family" / fam / rel
                if rows:
                    write_rows(dst, list(rows[0].keys()), fam_rows)
                if rel == gt_rel:
                    counts[fam][task] = len(fam_rows)
    return counts


def count(path: Path) -> int:
    return max(0, len(read_rows(path)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="shared/benchmark/eval_set_v1")
    ap.add_argument("--work-dir", default="shared/benchmark/_evalset_build",
                    help="scratch dir for raw generated data + task datasets")
    ap.add_argument("--valid-count", type=int, default=400, help="valid sequences per family")
    ap.add_argument("--invalid-per-rule-family", type=int, default=30,
                    help="easy invalid samples per (family x rule)")
    ap.add_argument("--hard-per-rule-family", type=int, default=20,
                    help="hard invalid samples per (family x rule)")
    ap.add_argument("--next-step-stride", type=int, default=8,
                    help="one next-step example every N target positions (subsample)")
    ap.add_argument("--context-window", type=int, default=64)
    ap.add_argument("--completion-cut-fracs", nargs="+", type=float, default=[0.6, 0.8],
                    help="organizer protocol: truncate at 60%% and 80%%")
    ap.add_argument("--completion-target-window", type=int, default=80,
                    help="max suffix steps to predict in completion (bounds decode cost)")
    ap.add_argument("--completion-sample", type=int, default=30,
                    help="cap completion examples per family (greedy decode is slow); 0 = keep all")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    out_dir = (ROOT / a.out_dir).resolve()
    work = (ROOT / a.work_dir).resolve()
    raw = work / "_raw"
    raw.mkdir(parents=True, exist_ok=True)
    py = a.python

    print(f"[1/5] generate VALID sequences ({a.valid_count}/family, benchmark seeds)")
    valid_files = []
    for fam in FAMILIES:
        out = raw / f"valid_{fam}.csv"   # family inferred from filename downstream
        run([py, GEN, "--family", fam, "--count", a.valid_count,
             "--seed", VALID_SEEDS[fam], "--output", out])
        valid_files.append(out)

    print("[2/5] generate INVALID sequences (easy + hard)")
    easy_dir = raw / "easy_invalid"
    hard_dir = raw / "hard_invalid"
    run([py, GEN_INVALID, "--valid-input", *valid_files, "--output-dir", easy_dir,
         "--seed", EASY_SEED, "--target-per-rule-family", a.invalid_per_rule_family,
         "--max-validator-violations", 4])
    run([py, GEN_HARD, "--valid-input", *valid_files, "--output-dir", hard_dir,
         "--seed", HARD_SEED, "--target-per-rule-family", a.hard_per_rule_family,
         "--max-validator-violations", 4])
    easy_long = find_invalid_long(easy_dir)
    hard_long = find_invalid_long(hard_dir)

    print(f"[3/5] build labeled task datasets (cut-fracs={a.completion_cut_fracs})")
    task_dir = work / "task_datasets"
    run([py, BUILD, "--valid-input", *valid_files,
         "--easy-invalid-input", easy_long, "--hard-invalid-input", hard_long,
         "--output-dir", task_dir, "--seed", SPLIT_SEED,
         "--context-window", a.context_window, "--next-step-stride", a.next_step_stride,
         "--completion-cut-fracs", *[str(x) for x in a.completion_cut_fracs],
         "--completion-target-window", a.completion_target_window])

    print("[4/5] prepare organizer-format common eval set (test split)")
    run([py, PREPARE, "--task-dir", task_dir, "--split", "test", "--out-dir", out_dir])
    # Next-step at the organizer's 60/80% cut points (derived before completion subsampling).
    rebuild_nextstep_from_completion(out_dir)
    subsample_completion(out_dir, a.completion_sample)

    print("[5/5] derive per-family (LoFO) subsets")
    fam_counts = derive_by_family(out_dir)

    manifest = {
        "out_dir": str(out_dir.relative_to(ROOT)),
        "params": vars(a),
        "valid_seeds": VALID_SEEDS, "easy_seed": EASY_SEED, "hard_seed": HARD_SEED,
        "split_seed": SPLIT_SEED,
        "overall_counts": {
            "nextstep": count(out_dir / "ground_truth/nextstep_gt.csv"),
            "completion": count(out_dir / "ground_truth/completion_gt.csv"),
            "anomaly": count(out_dir / "ground_truth/anomaly_gt.csv"),
        },
        "per_family_counts": fam_counts,
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nDONE. Common eval set:", out_dir)
    print(json.dumps(manifest["overall_counts"], indent=2))
    print("per-family:", json.dumps(fam_counts))


if __name__ == "__main__":
    main()
