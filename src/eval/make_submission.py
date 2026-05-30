"""Read the organizers' eval_input_*.csv files and emit the three submission
CSVs in the exact documented format (generation_rules.md §5).

Outputs:
    extras/results/nextstep.csv   (EXAMPLE_ID, RANK_1, RANK_2, RANK_3, RANK_4, RANK_5)
    extras/results/completion.csv (EXAMPLE_ID, PREDICTED_SEQUENCE)
    extras/results/anomaly.csv    (EXAMPLE_ID, IS_VALID, SCORE, PREDICTED_RULE)

Designed to work whether the eval files are the real organizers' input or
our locally-simulated ones (`src.eval.simulate_eval_input`).
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

from src.data.canonicalize import canonicalize_sequence
from src.eval.predict import (
    anomaly_ensemble,
    complete_sequence,
    load_model,
    topk_next_step,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger("submit")


def make_nextstep(lm, valid_input_path: Path, output_path: Path,
                   grammar: bool, canonicalize: bool) -> None:
    with valid_input_path.open(newline="") as f, output_path.open("w", newline="") as out:
        reader = csv.DictReader(f)
        writer = csv.writer(out)
        writer.writerow(["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"])
        n = 0
        t0 = time.time()
        for row in reader:
            eid = row["EXAMPLE_ID"]
            fam = row["FAMILY"].strip().lower()
            partial = row["PARTIAL_SEQUENCE"].split("|")
            ranked = topk_next_step(lm, fam, partial, k=5, grammar=grammar)
            if canonicalize:
                ranked = canonicalize_sequence(ranked)
            while len(ranked) < 5:
                ranked.append("")
            writer.writerow([eid] + ranked[:5])
            n += 1
            if n % 50 == 0:
                logger.info(f"nextstep n={n}  elapsed={time.time()-t0:.1f}s")
    logger.info(f"wrote {output_path}  (n={n}, {time.time()-t0:.1f}s)")


def make_completion(lm, valid_input_path: Path, output_path: Path,
                     grammar: bool, canonicalize: bool) -> None:
    with valid_input_path.open(newline="") as f, output_path.open("w", newline="") as out:
        reader = csv.DictReader(f)
        writer = csv.writer(out)
        writer.writerow(["EXAMPLE_ID", "PREDICTED_SEQUENCE"])
        n = 0
        t0 = time.time()
        for row in reader:
            eid = row["EXAMPLE_ID"]
            fam = row["FAMILY"].strip().lower()
            partial = row["PARTIAL_SEQUENCE"].split("|")
            # Estimate target completion length from frac
            frac = float(row.get("COMPLETION_FRACTION", 0.7))
            expected_full = int(len(partial) / max(frac, 0.05))
            max_completion = expected_full - len(partial) + 30
            pred = complete_sequence(lm, fam, partial, max_len=max_completion,
                                       grammar=grammar)
            if canonicalize:
                pred = canonicalize_sequence(pred)
            writer.writerow([eid, "|".join(pred)])
            n += 1
            if n % 50 == 0:
                logger.info(f"completion n={n}  elapsed={time.time()-t0:.1f}s")
    logger.info(f"wrote {output_path}  (n={n}, {time.time()-t0:.1f}s)")


def make_anomaly(lm, anomaly_input_path: Path, output_path: Path) -> None:
    with anomaly_input_path.open(newline="") as f, output_path.open("w", newline="") as out:
        reader = csv.DictReader(f)
        writer = csv.writer(out)
        writer.writerow(["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"])
        n = 0
        t0 = time.time()
        for row in reader:
            eid = row["EXAMPLE_ID"]
            fam = row["FAMILY"].strip().lower()
            seq = row["SEQUENCE"].split("|")
            result = anomaly_ensemble(lm, fam, seq)
            writer.writerow([
                eid,
                int(result["IS_VALID"]),
                f"{float(result['SCORE']):.4f}",
                result["PREDICTED_RULE"],
            ])
            n += 1
            if n % 50 == 0:
                logger.info(f"anomaly n={n}  elapsed={time.time()-t0:.1f}s")
    logger.info(f"wrote {output_path}  (n={n}, {time.time()-t0:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--valid-input",
                        default="extras/results/eval_inputs/eval_input_valid.csv")
    parser.add_argument("--anomaly-input",
                        default="extras/results/eval_inputs/eval_input_anomaly.csv")
    parser.add_argument("--output-dir", default="extras/results/submission")
    parser.add_argument("--no-grammar", action="store_true")
    parser.add_argument("--canonicalize", action="store_true",
                        help="Canonicalize synonyms on output (helps exact-match)")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Skip any of: nextstep completion anomaly")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {args.checkpoint}…")
    lm = load_model(Path(args.checkpoint))
    logger.info(f"  arch={lm.cfg['arch']}  tok={lm.cfg['tokenization']['mode']}  device={lm.device}")
    grammar = not args.no_grammar

    if "nextstep" not in args.skip:
        make_nextstep(lm, Path(args.valid_input), out_dir / "nextstep.csv",
                      grammar=grammar, canonicalize=args.canonicalize)
    if "completion" not in args.skip:
        make_completion(lm, Path(args.valid_input), out_dir / "completion.csv",
                        grammar=grammar, canonicalize=args.canonicalize)
    if "anomaly" not in args.skip:
        make_anomaly(lm, Path(args.anomaly_input), out_dir / "anomaly.csv")

    logger.info(f"All submission files in {out_dir}")


if __name__ == "__main__":
    main()
