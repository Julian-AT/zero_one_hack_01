"""Generate `eval_input_valid.csv` and `eval_input_anomaly.csv` in the
documented organizer format (generation_rules.md §5) from our held-out test
set. Used to wire up + sanity-check the submission pipeline before the real
eval files arrive.

Output schemas:

eval_input_valid.csv:
    EXAMPLE_ID, FAMILY, COMPLETION_FRACTION, PARTIAL_SEQUENCE
    (PARTIAL_SEQUENCE = "STEP1|STEP2|...")

eval_input_anomaly.csv:
    EXAMPLE_ID, FAMILY, SEQUENCE
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from transformer_xlstm.data.corrupt import corrupt_random
from transformer_xlstm.data.load import load_all_families
from transformer_xlstm.utils.paths import REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-family-valid", type=int, default=100)
    parser.add_argument("--per-family-anomaly", type=int, default=100)
    parser.add_argument(
        "--corrupt-frac",
        type=float,
        default=0.4,
        help="Fraction of anomaly examples that are corrupted",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="shared/extras/results/eval_inputs")
    args = parser.parse_args()

    out_dir = Path(REPO_ROOT) / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    examples = load_all_families()
    by_fam: dict[str, list] = {}
    for ex in examples:
        by_fam.setdefault(ex.family, []).append(ex)
    held: dict[str, list] = {fam: lst[-args.per_family_valid :] for fam, lst in by_fam.items()}
    held_anomaly: dict[str, list] = {
        fam: lst[-args.per_family_anomaly :] for fam, lst in by_fam.items()
    }

    valid_path = out_dir / "eval_input_valid.csv"
    with valid_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "FAMILY", "COMPLETION_FRACTION", "PARTIAL_SEQUENCE"])
        for fam, lst in held.items():
            for i, ex in enumerate(lst):
                for frac in (0.6, 0.8):
                    cut = max(2, int(len(ex.steps) * frac))
                    if cut >= len(ex.steps):
                        continue
                    eid = f"valid_{fam}_{i:04d}_f{int(frac * 100)}"
                    w.writerow([eid, fam, frac, "|".join(ex.steps[:cut])])
    print(f"wrote {valid_path}")

    truth_path = out_dir / "eval_input_anomaly_truth.csv"
    test_path = out_dir / "eval_input_anomaly.csv"
    truths: list[dict] = []
    with test_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "FAMILY", "SEQUENCE"])
        for fam, lst in held_anomaly.items():
            for i, ex in enumerate(lst):
                eid = f"anom_{fam}_{i:04d}"
                if rng.random() < args.corrupt_frac:
                    c = corrupt_random(list(ex.steps), rng, verify=True)
                    if c is not None:
                        w.writerow([eid, fam, "|".join(c.corrupted_steps)])
                        truths.append({"EXAMPLE_ID": eid, "IS_VALID": 0, "RULE": c.rule})
                        continue
                w.writerow([eid, fam, "|".join(ex.steps)])
                truths.append({"EXAMPLE_ID": eid, "IS_VALID": 1, "RULE": ""})
    print(f"wrote {test_path}")

    # Truth file (held back from the model)
    with truth_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["EXAMPLE_ID", "IS_VALID", "RULE"])
        w.writeheader()
        for t in truths:
            w.writerow(t)
    print(f"wrote {truth_path}")


if __name__ == "__main__":
    main()
