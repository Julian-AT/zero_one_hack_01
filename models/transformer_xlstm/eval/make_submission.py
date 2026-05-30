"""Read the organizers' eval_input_*.csv files and emit the three submission
CSVs in the exact documented format (generation_rules.md §5).

Outputs:
    shared/extras/results/nextstep.csv   (EXAMPLE_ID, RANK_1, RANK_2, RANK_3, RANK_4, RANK_5)
    shared/extras/results/completion.csv (EXAMPLE_ID, PREDICTED_SEQUENCE)
    shared/extras/results/anomaly.csv    (EXAMPLE_ID, IS_VALID, SCORE, PREDICTED_RULE)

Designed to work whether the eval files are the real organizers' input or
our locally-simulated ones (`transformer_xlstm.eval.simulate_eval_input`).
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

from transformer_xlstm.data.canonicalize import canonicalize_sequence
from transformer_xlstm.data.load import load_all_families
from transformer_xlstm.data.validator import validate_sequence
from transformer_xlstm.eval.predict import (
    anomaly_ensemble,
    complete_sequence,
    load_model,
    topk_next_step,
)

# --------------------------------------------------------------------------- #
# Trigram fallback — guarantees we fill all 5 RANK_* slots                    #
# --------------------------------------------------------------------------- #

_TRIGRAM = None


def _load_trigram():
    """Lazy-build the same trigram-with-backoff used in shared/extras/baselines."""
    global _TRIGRAM
    if _TRIGRAM is not None:
        return _TRIGRAM
    from collections import Counter, defaultdict

    class _Trigram:
        def __init__(self):
            self.tri = defaultdict(Counter)
            self.bi = defaultdict(Counter)
            self.uni = Counter()

        def fit(self, sequences):
            for s in sequences:
                for i, w in enumerate(s):
                    self.uni[w] += 1
                    if i >= 1:
                        self.bi[s[i - 1]][w] += 1
                    if i >= 2:
                        self.tri[(s[i - 2], s[i - 1])][w] += 1

        def rank(self, prefix, k=20):
            ctx2 = prefix[-2] if len(prefix) >= 2 else None
            ctx1 = prefix[-1] if len(prefix) >= 1 else None
            out, seen = [], set()

            def add_from(counter):
                for w, _ in counter.most_common():
                    if w in seen:
                        continue
                    out.append(w)
                    seen.add(w)
                    if len(out) >= k:
                        return True
                return False

            if ctx2 is not None and ctx1 is not None and (ctx2, ctx1) in self.tri:
                if add_from(self.tri[(ctx2, ctx1)]):
                    return out
            if ctx1 is not None and ctx1 in self.bi:
                if add_from(self.bi[ctx1]):
                    return out
            add_from(self.uni)
            return out

    t = _Trigram()
    t.fit([ex.steps for ex in load_all_families()])
    _TRIGRAM = t
    return t


def _candidate_violates(prefix: list[str], candidate: str) -> bool:
    new_prefix = prefix + [candidate]
    new_idx = len(prefix)
    return any(v.step_index == new_idx for v in validate_sequence(new_prefix))


def _pad_with_trigram(ranked: list[str], prefix: list[str], k: int = 5) -> list[str]:
    """Fill the rank list to k entries using grammar-trigram fallback.

    Compositional beam search often returns <5 distinct step strings;
    leaving RANK_4 / RANK_5 empty costs Top-5 score on the official scorer.
    Trigram-with-backoff has Top-5 = 0.993 on ID — perfect filler.
    """
    if len(ranked) >= k:
        return ranked[:k]
    trigram = _load_trigram()
    pool = trigram.rank(prefix, k=k * 4)  # broader pool
    # Grammar mask the fallback too
    for cand in pool:
        if cand in ranked:
            continue
        if _candidate_violates(prefix, cand):
            continue
        ranked.append(cand)
        if len(ranked) >= k:
            break
    # If grammar mask was too aggressive, fall back to raw trigram pool
    for cand in pool:
        if len(ranked) >= k:
            break
        if cand not in ranked:
            ranked.append(cand)
    return ranked[:k]


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s", stream=sys.stdout
)
logger = logging.getLogger("submit")


def make_nextstep(
    lm, valid_input_path: Path, output_path: Path, grammar: bool, canonicalize: bool
) -> None:
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
            # Pad with trigram-grammar fallback to guarantee all 5 ranks filled
            # (compositional beam search often returns <5 distinct step strings).
            if len(ranked) < 5:
                ranked = _pad_with_trigram(ranked, partial, k=5)
            if canonicalize:
                ranked = canonicalize_sequence(ranked)
            while len(ranked) < 5:
                ranked.append("")
            writer.writerow([eid] + ranked[:5])
            n += 1
            if n % 50 == 0:
                logger.info(f"nextstep n={n}  elapsed={time.time() - t0:.1f}s")
    logger.info(f"wrote {output_path}  (n={n}, {time.time() - t0:.1f}s)")


def make_completion(
    lm, valid_input_path: Path, output_path: Path, grammar: bool, canonicalize: bool
) -> None:
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
            pred = complete_sequence(lm, fam, partial, max_len=max_completion, grammar=grammar)
            if canonicalize:
                pred = canonicalize_sequence(pred)
            writer.writerow([eid, "|".join(pred)])
            n += 1
            if n % 50 == 0:
                logger.info(f"completion n={n}  elapsed={time.time() - t0:.1f}s")
    logger.info(f"wrote {output_path}  (n={n}, {time.time() - t0:.1f}s)")


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
            writer.writerow(
                [
                    eid,
                    int(result["IS_VALID"]),
                    f"{float(result['SCORE']):.4f}",
                    result["PREDICTED_RULE"],
                ]
            )
            n += 1
            if n % 50 == 0:
                logger.info(f"anomaly n={n}  elapsed={time.time() - t0:.1f}s")
    logger.info(f"wrote {output_path}  (n={n}, {time.time() - t0:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--valid-input", default="shared/extras/results/eval_inputs/eval_input_valid.csv"
    )
    parser.add_argument(
        "--anomaly-input", default="shared/extras/results/eval_inputs/eval_input_anomaly.csv"
    )
    parser.add_argument("--output-dir", default="shared/extras/results/submission")
    parser.add_argument("--no-grammar", action="store_true")
    parser.add_argument(
        "--canonicalize",
        action="store_true",
        help="Canonicalize synonyms on output (helps exact-match)",
    )
    parser.add_argument(
        "--skip", nargs="*", default=[], help="Skip any of: nextstep completion anomaly"
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {args.checkpoint}…")
    lm = load_model(Path(args.checkpoint))
    logger.info(
        f"  arch={lm.cfg['arch']}  tok={lm.cfg['tokenization']['mode']}  device={lm.device}"
    )
    grammar = not args.no_grammar

    if "nextstep" not in args.skip:
        make_nextstep(
            lm,
            Path(args.valid_input),
            out_dir / "nextstep.csv",
            grammar=grammar,
            canonicalize=args.canonicalize,
        )
    if "completion" not in args.skip:
        make_completion(
            lm,
            Path(args.valid_input),
            out_dir / "completion.csv",
            grammar=grammar,
            canonicalize=args.canonicalize,
        )
    if "anomaly" not in args.skip:
        make_anomaly(lm, Path(args.anomaly_input), out_dir / "anomaly.csv")

    logger.info(f"All submission files in {out_dir}")


if __name__ == "__main__":
    main()
