"""Wrap the official scorer and run the Leave-One-Family-Out (LoFO) harness.

Two responsibilities:

  * ``score(task, gt_csv, pred_csv, valid_supplement=None)`` — invoke the
    organizers' ``eval_metrics.py`` as a subprocess (never imported, never
    modified) and parse its printed metric lines into a clean dict. The raw
    stdout is preserved under ``raw_stdout`` so nothing is lost.

  * ``lofo(make_ranker, ...)`` — the Task-4 measurement. For each held-out
    family in ``nspe.data.LOFO_SPLITS`` we train a ranker on the other two,
    build BOTH an in-distribution (ID) eval set on the train families and an
    out-of-distribution (OOD) eval set on the held-out family via
    ``nspe.simulate_eval``, predict next-step + completion, score with the
    official scorer, and record the per-metric ID, OOD, and ID→OOD drop. The
    flatness of that drop is the headline neurosymbolic result.

Symbolic core — no torch at module top. ``make_ranker`` is a caller-supplied
callable ``train_families -> ranker``; the ranker is duck-typed.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

from nspe import data, predict, simulate_eval
from nspe.official import EVAL_PATH

PathLike = Union[str, Path]

__all__ = ["score", "lofo", "METRIC_KEYS"]

# Metric keys we extract per task (canonical snake_case names).
METRIC_KEYS = {
    "next-step": ["top1", "top3", "top5", "mrr"],
    "completion": ["ned", "exact_match", "token_acc", "block_acc"],
    "anomaly": ["accuracy", "precision", "recall", "f1", "auc", "rule_attr"],
}

# (canonical key, regex capturing the float) for each printed metric line. The
# patterns target the exact labels emitted by eval_metrics.py.
_FLOAT = r"([0-9]*\.?[0-9]+)"
_PATTERNS = {
    "next-step": [
        ("top1", re.compile(r"Top-1 Accuracy\s*:\s*" + _FLOAT)),
        ("top3", re.compile(r"Top-3 Accuracy\s*:\s*" + _FLOAT)),
        ("top5", re.compile(r"Top-5 Accuracy\s*:\s*" + _FLOAT)),
        ("mrr", re.compile(r"MRR\s*:\s*" + _FLOAT)),
    ],
    "completion": [
        ("ned", re.compile(r"Mean Normalized Edit Distance\s*:\s*" + _FLOAT)),
        ("exact_match", re.compile(r"Exact Match Rate\s*:\s*" + _FLOAT)),
        ("token_acc", re.compile(r"Mean Token Accuracy\s*:\s*" + _FLOAT)),
        ("block_acc", re.compile(r"Mean Block-level Accuracy\s*:\s*" + _FLOAT)),
    ],
    "anomaly": [
        ("accuracy", re.compile(r"Binary Accuracy\s*:\s*" + _FLOAT)),
        ("precision", re.compile(r"Precision \(invalid class\)\s*:\s*" + _FLOAT)),
        ("recall", re.compile(r"Recall \(invalid class\)\s*:\s*" + _FLOAT)),
        ("f1", re.compile(r"F1 \(invalid class\)\s*:\s*" + _FLOAT)),
        ("auc", re.compile(r"ROC-AUC\s*:\s*" + _FLOAT)),
        ("rule_attr", re.compile(r"Rule Attribution Accuracy\s*:\s*" + _FLOAT)),
    ],
}


def _parse_metrics(task: str, stdout: str) -> dict:
    """Extract the canonical metric floats from the scorer's stdout.

    Only the *first* match of each pattern is taken (the ALL/overall line is
    printed before per-family/per-fraction breakdowns). Missing metrics (e.g.
    AUC printed as 'n/a') are omitted from the result.
    """
    out: dict = {}
    for key, pat in _PATTERNS.get(task, []):
        m = pat.search(stdout)
        if m:
            try:
                out[key] = float(m.group(1))
            except ValueError:
                pass
    return out


def score(
    task: str,
    gt_csv: PathLike,
    pred_csv: PathLike,
    valid_supplement: Optional[PathLike] = None,
) -> dict:
    """Run the official scorer for ``task`` and return parsed metrics.

    Args:
        task: one of ``next-step`` / ``completion`` / ``anomaly``.
        gt_csv: ground-truth CSV (the official ``--ground-truth``).
        pred_csv: predictions CSV (the official ``--predictions``).
        valid_supplement: optional valid-examples CSV for the anomaly AUC.

    Returns:
        dict of parsed metrics plus:
            ``task`` (echoed), ``returncode`` (subprocess exit code),
            ``raw_stdout`` (full scorer output), ``raw_stderr`` (if any).
    """
    if task not in METRIC_KEYS:
        raise ValueError(f"unknown task {task!r}; expected one of {list(METRIC_KEYS)}")

    cmd: List[str] = [
        sys.executable, str(EVAL_PATH),
        "--task", task,
        "--ground-truth", str(gt_csv),
        "--predictions", str(pred_csv),
    ]
    if task == "anomaly" and valid_supplement is not None:
        cmd += ["--valid-supplement", str(valid_supplement)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    metrics = _parse_metrics(task, proc.stdout)
    metrics.update({
        "task": task,
        "returncode": proc.returncode,
        "raw_stdout": proc.stdout,
    })
    if proc.stderr:
        metrics["raw_stderr"] = proc.stderr
    return metrics


# ---------------------------------------------------------------------------
# LoFO harness (Task 4)
# ---------------------------------------------------------------------------

def _load_seqs(families: Sequence[str], n_per_family: int) -> dict:
    """{family: [seq, ...]} of up to ``n_per_family`` sequences each."""
    out = {}
    for fam in families:
        fam = fam.lower()
        seqs = [list(s) for s in data.load_family(fam)]
        out[fam] = seqs[:n_per_family] if n_per_family else seqs
    return out


def _drop(id_m: dict, ood_m: dict, keys: Sequence[str]) -> dict:
    """Per-metric ID→OOD drop (ID - OOD), only where both are present.

    For NED (lower-is-better) the natural 'drop' is OOD - ID, but to keep one
    convention we report ID - OOD everywhere and label NED explicitly in the
    consumer; the raw ID and OOD values are also returned so either reading works.
    """
    out = {}
    for k in keys:
        if k in id_m and k in ood_m:
            out[k] = id_m[k] - ood_m[k]
    return out


def _score_split(
    held: str,
    train: Sequence[str],
    ranker,
    out_dir: Path,
    n_per_family: int,
    cand,
    use_roles: bool,
    beam: int,
) -> dict:
    """Build ID + OOD eval sets, predict, score; return the per-split record."""
    split_dir = out_dir / f"lofo_{held}"
    split_dir.mkdir(parents=True, exist_ok=True)

    id_seqs = _load_seqs(train, n_per_family)
    ood_seqs = _load_seqs([held], n_per_family)

    id_set = simulate_eval.build_eval_set(id_seqs, split_dir, prefix="id")
    ood_set = simulate_eval.build_eval_set(ood_seqs, split_dir, prefix="ood")

    record: dict = {"held_out": held, "train": list(train)}

    for tag, eset in (("id", id_set), ("ood", ood_set)):
        # ---- next-step ----
        ns_pred = split_dir / f"{tag}_nextstep_pred.csv"
        predict.predict_nextstep(eset["nextstep_input"], ns_pred, ranker, cand,
                                 use_roles=use_roles)
        ns_metrics = score("next-step", eset["nextstep_gt"], ns_pred)
        # ---- completion ----
        cp_pred = split_dir / f"{tag}_completion_pred.csv"
        predict.predict_completion(eset["completion_input"], cp_pred, ranker, cand,
                                   use_roles=use_roles, beam=beam)
        cp_metrics = score("completion", eset["completion_gt"], cp_pred)
        record[tag] = {
            "next-step": {k: ns_metrics[k] for k in METRIC_KEYS["next-step"] if k in ns_metrics},
            "completion": {k: cp_metrics[k] for k in METRIC_KEYS["completion"] if k in cp_metrics},
        }

    record["drop"] = {
        "next-step": _drop(record["id"]["next-step"], record["ood"]["next-step"],
                           METRIC_KEYS["next-step"]),
        "completion": _drop(record["id"]["completion"], record["ood"]["completion"],
                            METRIC_KEYS["completion"]),
    }
    return record


def lofo(
    make_ranker: Callable[[List[str]], object],
    n_per_family: int = 100,
    out_dir: PathLike = None,
    smoke: bool = False,
    use_roles: bool = False,
    beam: int = 1,
) -> dict:
    """Run the LoFO harness across all three held-out families.

    For each ``(held, train)`` split in ``data.LOFO_SPLITS``:
        ranker = make_ranker(train)
        build ID (train) + OOD (held) eval sets, predict next-step + completion,
        score with the official scorer, record ID / OOD / drop per metric.

    Args:
        make_ranker: callable ``train_families -> ranker`` (duck-typed ranker).
        n_per_family: cap on sequences per family for the eval sets (speed).
        out_dir: where to write predictions, GTs, and the summary JSON
            (defaults to ``$NSPE_OUT`` or ``models/neurosymbolic/outputs``).
        smoke: if True, run a single split with very small ``n_per_family`` for a
            fast pipeline check.
        use_roles: pass-through to the constrained decoder.
        beam: completion beam width.

    Returns:
        dict with ``splits`` (list of per-split records) and ``mean_drop``
        (averaged ID→OOD drop per task/metric). Also written to
        ``<out_dir>/lofo.json``.
    """
    import os

    out_dir = Path(out_dir) if out_dir is not None else Path(
        os.environ.get("NSPE_OUT", Path(__file__).resolve().parents[1] / "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = data.LOFO_SPLITS[:1] if smoke else data.LOFO_SPLITS
    if smoke:
        n_per_family = min(n_per_family, 10)

    records: List[dict] = []
    for held, train in splits:
        ranker = make_ranker(list(train))
        # Candidate vocab = the TRAIN families' steps (the OOD setting: the unseen
        # family's novel step strings are intentionally absent).
        cand = list(data.candidate_vocab(train))
        rec = _score_split(held, train, ranker, out_dir, n_per_family, cand,
                            use_roles, beam)
        records.append(rec)

    # Aggregate mean drop per task/metric across the splits.
    mean_drop: dict = {"next-step": {}, "completion": {}}
    for task in ("next-step", "completion"):
        for key in METRIC_KEYS[task]:
            vals = [r["drop"][task][key] for r in records if key in r["drop"][task]]
            if vals:
                mean_drop[task][key] = sum(vals) / len(vals)

    summary = {"splits": records, "mean_drop": mean_drop,
               "n_per_family": n_per_family, "smoke": smoke}
    out_json = out_dir / "lofo.json"
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    summary["json_path"] = str(out_json)
    return summary


# ---------------------------------------------------------------------------
# Self-test (PPM ranker): score next-step + completion via the official scorer
# and run a smoke LoFO.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    from nspe.data import candidate_vocab, load_family
    from nspe.ppm import PPM

    out_dir = Path(os.environ.get(
        "NSPE_OUT", Path(__file__).resolve().parents[1] / "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("eval.py self-test  (PPM ranker, official scorer)")
    print("=" * 64)

    # Fit PPM on mosfet+igbt; build a tiny eval set from 3 held-out ic seqs.
    train = {
        "mosfet": [list(s) for s in load_family("mosfet")[:200]],
        "igbt": [list(s) for s in load_family("igbt")[:200]],
    }
    ranker = PPM().fit(train)
    cand = candidate_vocab(("mosfet", "igbt"))

    ic = {"ic": [list(s) for s in load_family("ic")[:3]]}
    eset = simulate_eval.build_eval_set(ic, out_dir, prefix="eval_selftest")

    # Predict next-step + completion.
    ns_pred = out_dir / "eval_selftest_nextstep_pred.csv"
    cp_pred = out_dir / "eval_selftest_completion_pred.csv"
    predict.predict_nextstep(eset["nextstep_input"], ns_pred, ranker, cand)
    predict.predict_completion(eset["completion_input"], cp_pred, ranker, cand)

    # Score both via the official scorer subprocess.
    ns_score = score("next-step", eset["nextstep_gt"], ns_pred)
    cp_score = score("completion", eset["completion_gt"], cp_pred)

    print("\n[next-step] parsed metrics:")
    for k in METRIC_KEYS["next-step"]:
        print(f"   {k:12s}: {ns_score.get(k)}")
    print(f"   returncode = {ns_score['returncode']}")
    assert ns_score["returncode"] == 0, "scorer failed for next-step"
    assert "top1" in ns_score and "top5" in ns_score, "failed to parse next-step metrics"

    print("\n[completion] parsed metrics:")
    for k in METRIC_KEYS["completion"]:
        print(f"   {k:12s}: {cp_score.get(k)}")
    print(f"   returncode = {cp_score['returncode']}")
    assert cp_score["returncode"] == 0, "scorer failed for completion"
    assert "ned" in cp_score and "token_acc" in cp_score, "failed to parse completion metrics"

    # Smoke LoFO: one split, tiny n, full predict+score pipeline.
    print("\n[lofo smoke] one split, n_per_family<=10 ...")
    def make_ranker(train_families):
        seqs = {f: [list(s) for s in load_family(f)[:200]] for f in train_families}
        return PPM().fit(seqs)

    summary = lofo(make_ranker, n_per_family=8, out_dir=out_dir / "lofo_smoke", smoke=True)
    split = summary["splits"][0]
    print(f"   held_out = {split['held_out']}  train = {split['train']}")
    print(f"   ID  next-step  = {split['id']['next-step']}")
    print(f"   OOD next-step  = {split['ood']['next-step']}")
    print(f"   drop next-step = {split['drop']['next-step']}")
    print(f"   ID  completion = {split['id']['completion']}")
    print(f"   OOD completion = {split['ood']['completion']}")
    print(f"   mean_drop      = {summary['mean_drop']}")
    print(f"   json -> {summary['json_path']}")
    assert Path(summary["json_path"]).exists(), "lofo.json not written"
    assert "top1" in split["id"]["next-step"], "lofo failed to score ID next-step"
    assert "top1" in split["ood"]["next-step"], "lofo failed to score OOD next-step"

    print("\nSELF-TEST PASSED")
