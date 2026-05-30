"""Task-3 anomaly classifier — the pure symbolic oracle (role-augmented).

The default classifier is *fully symbolic*: a sequence is invalid iff the
(role-augmented) official validator reports a violation, and the attributed rule
is the earliest violation. No learning is involved and no torch is imported. The
OOD probe established that this oracle achieves 0 false positives and ~100%
recall on the unseen 4th family when role-induction is enabled, which is why it
is the headline submission.

An *optional* learned residual is supported (OFF by default): given a duck-typed
``ranker`` exposing ``perplexity(seq) -> float`` and a ``ppl_threshold``, a
sequence the symbolic engine deems valid can be downgraded to invalid when its
perplexity exceeds the threshold (catching structurally-odd, novel-vocabulary
anomalies that role-induction alone misses). This path is intended for the
report's OOD story; keep it disabled unless it improves held-out F1.

The SCORE column follows the official scorer convention: higher = more likely
*valid* (used for ROC-AUC).

This module imports only the stdlib + the symbolic core (`nspe.rules`). It does
NOT import torch; the optional ``ranker`` is duck-typed.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Protocol

from nspe import rules

# SCORE values for the symbolic verdict. The scorer treats higher as "more
# likely valid", so valid sequences get a high score and invalid a low one.
_VALID_SCORE = 0.98
_INVALID_SCORE = 0.02
# When the optional residual downgrades a 'valid' verdict, mark it with this rule.
_RESIDUAL_RULE = "RULE_UNKNOWN"
_RESIDUAL_SCORE = 0.10


class _Ranker(Protocol):
    """Duck-typed ranker interface used only by the optional residual path."""

    def perplexity(self, seq: list) -> float: ...


def classify(
    seq,
    ranker: Optional["_Ranker"] = None,
    ppl_threshold: Optional[float] = None,
    use_roles: bool = True,
) -> dict:
    """Classify one sequence for Task 3.

    Args:
        seq: iterable of step strings (one process sequence).
        ranker: optional duck-typed ranker exposing ``perplexity(seq)`` for the
            residual mode. Ignored unless ``ppl_threshold`` is also given.
        ppl_threshold: if given together with ``ranker``, a symbolically-valid
            sequence whose perplexity exceeds this threshold is downgraded to
            invalid (rule ``RULE_UNKNOWN``). OFF by default.
        use_roles: use the role-augmented validator (recommended; needed for the
            OOD/novel-vocabulary generalization).

    Returns:
        dict with keys:
            ``is_valid`` (int: 1 valid, 0 invalid),
            ``score`` (float in [0,1]; higher = more likely valid),
            ``rule`` (str rule-id of the first violation, or None if valid).
    """
    seq = list(seq)
    violations = rules.validate_with_roles(seq) if use_roles else rules.validate(seq)

    if violations:
        return {
            "is_valid": 0,
            "score": _INVALID_SCORE,
            "rule": rules.first_rule(seq, use_roles=use_roles),
        }

    # Symbolic engine says VALID. Optional learned residual for novel anomalies.
    if ranker is not None and ppl_threshold is not None:
        ppl = float(ranker.perplexity(seq))
        if ppl > ppl_threshold:
            return {
                "is_valid": 0,
                "score": _RESIDUAL_SCORE,
                "rule": _RESIDUAL_RULE,
            }

    return {"is_valid": 1, "score": _VALID_SCORE, "rule": None}


def _split_sequence(raw: str) -> list:
    """Split a pipe-joined sequence cell into a list of step strings."""
    if raw is None:
        return []
    return [s for s in (p.strip() for p in raw.split("|")) if s]


def classify_file(
    input_csv,
    out_csv,
    ranker: Optional["_Ranker"] = None,
    ppl_threshold: Optional[float] = None,
    use_roles: bool = True,
) -> int:
    """Classify every row of an anomaly-eval input CSV; write the Task-3 CSV.

    Input schema (official ``eval_input_anomaly.csv``): columns
    ``EXAMPLE_ID,FAMILY,SEQUENCE`` where SEQUENCE is pipe-joined. Extra columns
    are ignored; column names are matched case-insensitively. A ``PARTIAL_SEQUENCE``
    / ``FULL_SEQUENCE`` column is accepted as a fallback sequence source.

    Output schema (official Task 3): ``EXAMPLE_ID,IS_VALID,SCORE,PREDICTED_RULE``.

    Returns the number of rows written.
    """
    input_csv = Path(input_csv)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows_out = []
    with input_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        # Case-insensitive header lookup.
        field_map = {(name or "").strip().upper(): name for name in (reader.fieldnames or [])}
        id_key = field_map.get("EXAMPLE_ID")
        seq_key = (
            field_map.get("SEQUENCE")
            or field_map.get("PARTIAL_SEQUENCE")
            or field_map.get("FULL_SEQUENCE")
        )
        if id_key is None or seq_key is None:
            raise ValueError(
                f"input CSV {input_csv} must have EXAMPLE_ID and a SEQUENCE column; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            example_id = (row.get(id_key) or "").strip()
            seq = _split_sequence(row.get(seq_key))
            res = classify(seq, ranker=ranker, ppl_threshold=ppl_threshold, use_roles=use_roles)
            rows_out.append(
                {
                    "EXAMPLE_ID": example_id,
                    "IS_VALID": res["is_valid"],
                    "SCORE": f"{res['score']:.4f}",
                    "PREDICTED_RULE": res["rule"] if res["rule"] is not None else "",
                }
            )

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"]
        )
        writer.writeheader()
        writer.writerows(rows_out)

    return len(rows_out)


__all__ = ["classify", "classify_file"]


if __name__ == "__main__":
    # ---- Self-test -------------------------------------------------------
    import os

    from nspe import data
    from nspe.official import EVAL_INPUT_ANOMALY

    out_dir = Path(os.environ.get("NSPE_OUT", Path(__file__).resolve().parents[1] / "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("anomaly.py self-test")
    print("=" * 64)

    # 1) Five valid sequences from training data -> is_valid == 1.
    pairs = data.all_sequences()[:5]
    n_valid_ok = 0
    for fam, seq in pairs:
        res = classify(seq)
        ok = res["is_valid"] == 1 and res["rule"] is None
        n_valid_ok += ok
        print(f"  valid[{fam:6s}] -> is_valid={res['is_valid']} score={res['score']} rule={res['rule']} {'OK' if ok else 'FAIL'}")
    assert n_valid_ok == 5, f"expected 5 valid, got {n_valid_ok}"

    # 2) Corrupt five sequences and confirm is_valid==0 with matching rule.
    #    Prefer nspe.corrupt (the dedicated injector). Fall back to a couple of
    #    minimal inline injections if corrupt.py is not yet built, so this
    #    self-test is runnable independent of build order.
    print("-" * 64)
    corrupted = []  # list of (label, corrupted_seq, expected_rule)
    try:
        from nspe import corrupt as _corrupt  # type: ignore

        import random

        # nspe.corrupt exposes CORRUPTORS: {rule_id -> fn(seq, rng, novel=False)}
        # where fn returns the corrupted sequence (or None if the trigger is absent).
        rng = random.Random(0)
        registry = dict(getattr(_corrupt, "CORRUPTORS", {}))
        base_pairs = data.all_sequences()
        for rule_id, fn in registry.items():
            for _fam, seq in base_pairs:
                try:
                    cseq = fn(list(seq), rng)
                except Exception:
                    cseq = None
                if not cseq:
                    continue
                # confirm the intended rule actually fires (some seqs lack trigger)
                if rules.first_rule(cseq, use_roles=True) == rule_id:
                    corrupted.append((rule_id, cseq, rule_id))
                    break
            if len(corrupted) >= 5:
                break
        print(f"  using nspe.corrupt: built {len(corrupted)} corrupted seqs")
    except Exception as exc:  # pragma: no cover - depends on build order
        print(f"  nspe.corrupt unavailable ({exc!r}); using inline fallback injections")

    if len(corrupted) < 5:
        # Inline fallback (runnable without nspe.corrupt): build deterministic
        # single-rule injections. Each candidate is VERIFIED via rules.first_rule
        # so the expected rule is correct by construction; we only keep an
        # injection when its realized first-violation equals the intended rule,
        # which is exactly the property classify() is tested against.
        clean_upper = lambda x: ("CLEAN" in x.upper() or x.upper().endswith(" RINSE")
                                 or x.upper().startswith("DRY ") or x.upper() == "HF DIP")

        def _inj_dep_no_clean(seq):
            """Remove every clean step before the first deposition -> DEP_NO_CLEAN."""
            seq = list(seq)
            di = next((i for i, x in enumerate(seq) if x.upper().startswith("DEPOSIT ")), None)
            if di is None:
                return None
            return [x for j, x in enumerate(seq) if not (j < di and clean_upper(x))]

        def _inj_ship_before_test(seq):
            """Move SHIP LOT to the front -> SHIP_BEFORE_TEST."""
            rest = [x for x in seq if x.upper() != "SHIP LOT"]
            if len(rest) == len(seq):
                return None
            return ["SHIP LOT"] + rest

        def _inj_litho_skip(seq):
            """Bump the first ALIGN MASK LEVEL n -> n+2 -> LITHO_LEVEL_SKIP."""
            import re
            seq = list(seq)
            for i, x in enumerate(seq):
                m = re.search(r"ALIGN MASK LEVEL (\d+)", x.upper())
                if m and int(m.group(1)) == 1:
                    n = int(m.group(1))
                    seq[i] = re.sub(r"(\d+)", str(n + 2), x, count=1)
                    return seq
            return None

        def _inj_backside_before_pass(seq):
            """Move DEPOSIT BACKSIDE METAL before CURE PASSIVATION."""
            seq = list(seq)
            bi = next((i for i, x in enumerate(seq) if "BACKSIDE METAL" in x.upper()), None)
            ci = next((i for i, x in enumerate(seq) if x.upper() == "CURE PASSIVATION"), None)
            if bi is None or ci is None or bi <= ci:
                return None
            step = seq.pop(bi)
            seq.insert(ci, step)
            return seq

        def _inj_test_before_pass(seq):
            """Move a *_TEST step before CURE PASSIVATION -> TEST_BEFORE_PASSIVATION."""
            seq = list(seq)
            ci = next((i for i, x in enumerate(seq) if x.upper() == "CURE PASSIVATION"), None)
            ti = next((i for i, x in enumerate(seq)
                       if x.upper().endswith(" TEST") and i > (ci if ci is not None else -1)), None)
            if ci is None or ti is None:
                return None
            step = seq.pop(ti)
            seq.insert(ci, step)
            return seq

        injectors = [
            ("dep_no_clean", _inj_dep_no_clean, "RULE_DEP_NO_CLEAN"),
            ("ship_before_test", _inj_ship_before_test, "RULE_SHIP_BEFORE_TEST"),
            ("litho_level_skip", _inj_litho_skip, "RULE_LITHO_LEVEL_SKIP"),
            ("backside_before_pass", _inj_backside_before_pass, "RULE_BACKSIDE_BEFORE_PASSIVATION"),
            ("test_before_pass", _inj_test_before_pass, "RULE_TEST_BEFORE_PASSIVATION"),
        ]
        bases = [list(s) for _f, s in data.all_sequences()]
        for label, fn, want_rule in injectors:
            for base in bases:
                cseq = fn(base)
                if cseq is None:
                    continue
                # Verify the intended rule is the FIRST violation (correct-by-construction).
                if rules.first_rule(cseq, use_roles=True) == want_rule:
                    corrupted.append((label, cseq, want_rule))
                    break
            if len(corrupted) >= 5:
                break

    n_inv_ok = 0
    n_checked = 0
    for label, cseq, expected_rule in corrupted[:5]:
        res = classify(cseq)
        # The injected rule must be the *first* violation the classifier reports.
        rule_match = res["rule"] == expected_rule
        ok = res["is_valid"] == 0 and rule_match
        n_inv_ok += ok
        n_checked += 1
        print(f"  invalid[{label:20s}] -> is_valid={res['is_valid']} rule={res['rule']} expect={expected_rule} {'OK' if ok else 'FAIL'}")
    assert n_checked >= 5, f"only built {n_checked} corrupted seqs (need 5)"
    assert n_inv_ok == n_checked, f"expected all {n_checked} invalid w/ matching rule, got {n_inv_ok}"

    # 3) Run classify_file on a 10-row slice of the official anomaly eval input.
    print("-" * 64)
    slice_csv = out_dir / "anomaly_input_slice10.csv"
    with EVAL_INPUT_ANOMALY.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [next(reader) for _ in range(10)]
    with slice_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)

    out_path = out_dir / "anomaly_selftest_task3.csv"
    n = classify_file(slice_csv, out_path)
    print(f"  classify_file wrote {n} rows -> {out_path}")
    with out_path.open(encoding="utf-8") as fh:
        head = fh.read().splitlines()[: n + 1]
    print("  --- Task-3 CSV head ---")
    for line in head:
        print("   ", line)

    print("=" * 64)
    print("SELF-TEST PASSED")
