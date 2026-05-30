#!/usr/bin/env python3
"""
score.py — score one model's organizer-format predictions against ground truth.

Reuses the official metric implementations from participant_files/eval_metrics.py so the
numbers are identical to the organizer scorer, while returning structured dicts (the official
script only prints). Used by run_benchmark.py.

Ground-truth CSVs (produced by prepare_eval_inputs.py):
  nextstep_gt.csv    : EXAMPLE_ID, FAMILY, NEXT_STEP
  completion_gt.csv  : EXAMPLE_ID, FAMILY, PARTIAL_SEQUENCE, FULL_SEQUENCE      (pipe '|' joined)
  anomaly_gt.csv     : EXAMPLE_ID, IS_VALID(0/1), VIOLATED_RULE

Prediction CSVs (organizer format):
  nextstep   : EXAMPLE_ID, RANK_1..RANK_5
  completion : EXAMPLE_ID, PREDICTED_SEQUENCE  (pipe '|' joined)
  anomaly    : EXAMPLE_ID, IS_VALID, [SCORE], [PREDICTED_RULE]
"""
from __future__ import annotations

import csv
import importlib.util
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_METRICS = ROOT / "participant_files" / "eval_metrics.py"


def _load_eval_metrics():
    spec = importlib.util.spec_from_file_location("eval_metrics", EVAL_METRICS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EM = _load_eval_metrics()


def _read(path: Path) -> list[dict]:
    """Read a CSV, normalizing header names (strip BOM/quotes/space, upper-case)."""
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({(k or "").strip().lstrip("\ufeff").strip('"').upper():
                         (v or "").strip() for k, v in row.items()})
    return rows


def _split_steps(s: str) -> list[str]:
    # ground truth / predictions use a single '|' separator in organizer format
    return [t.strip() for t in str(s).split("|") if t.strip()]


# --------------------------------------------------------------------------- next-step
def score_nextstep(pred_csv, gt_csv) -> dict:
    gt = {r["EXAMPLE_ID"]: r for r in _read(gt_csv)}
    pred = {}
    for r in _read(pred_csv):
        ranks = [r.get(f"RANK_{i}", "") for i in range(1, 6)]
        pred[r["EXAMPLE_ID"]] = [x for x in ranks if x]

    by_fam_hits = defaultdict(lambda: {"t1": [], "t3": [], "t5": [], "rr": []})
    overall = {"t1": [], "t3": [], "t5": [], "rr": []}
    matched = 0
    for eid, g in gt.items():
        if eid not in pred:
            continue
        matched += 1
        truth, fam, ranks = g["NEXT_STEP"], g.get("FAMILY", "ALL"), pred[eid]
        t1 = bool(ranks) and ranks[0] == truth
        t3 = truth in ranks[:3]
        t5 = truth in ranks[:5]
        rr = (1.0 / (ranks.index(truth) + 1)) if truth in ranks else 0.0
        for bucket in (overall, by_fam_hits[fam]):
            bucket["t1"].append(t1); bucket["t3"].append(t3)
            bucket["t5"].append(t5); bucket["rr"].append(rr)

    def agg(b):
        n = len(b["t1"]) or 1
        return {"top1": sum(b["t1"]) / n, "top3": sum(b["t3"]) / n,
                "top5": sum(b["t5"]) / n, "mrr": sum(b["rr"]) / n}

    out = {"task": "next-step", "n": matched, "overall": agg(overall),
           "per_family": {f: agg(b) for f, b in by_fam_hits.items()}}
    return out


# --------------------------------------------------------------------------- completion
def score_completion(pred_csv, gt_csv, validator=None) -> dict:
    gt = {r["EXAMPLE_ID"]: r for r in _read(gt_csv)}
    pred = {r["EXAMPLE_ID"]: _split_steps(r.get("PREDICTED_SEQUENCE", "")) for r in _read(pred_csv)}

    ned, exact, tok, blk, rule_valid = [], [], [], [], []
    matched = 0
    for eid, g in gt.items():
        if eid not in pred:
            continue
        matched += 1
        full = _split_steps(g["FULL_SEQUENCE"])
        partial = _split_steps(g.get("PARTIAL_SEQUENCE", ""))
        ref = full[len(partial):] if partial else full
        p = pred[eid]
        ned.append(EM.normalized_edit_distance(p, ref))
        exact.append(p == ref)
        tok.append(EM.token_accuracy(p, ref))
        blk.append(EM.block_level_accuracy(p, ref))
        if validator is not None:
            try:
                rule_valid.append(bool(validator(partial + p, g.get("FAMILY", ""))))
            except Exception:
                pass

    def mean(x):
        return sum(x) / len(x) if x else float("nan")

    out = {"task": "completion", "n": matched,
           "overall": {"ned": mean(ned), "exact_match": mean(exact),
                       "token_acc": mean(tok), "block_acc": mean(blk)}}
    if rule_valid:
        out["overall"]["rule_valid_frac"] = mean(rule_valid)
    return out


# --------------------------------------------------------------------------- anomaly
def score_anomaly(pred_csv, gt_csv) -> dict:
    gt = {r["EXAMPLE_ID"]: r for r in _read(gt_csv)}
    pred = {r["EXAMPLE_ID"]: r for r in _read(pred_csv)}

    labels, scores, preds = [], [], []
    rule_pairs = []  # (gt_rule, pred_rule) among correctly-detected invalids
    for eid, g in gt.items():
        if eid not in pred:
            continue
        y = int(float(g["IS_VALID"]))
        p = pred[eid]
        try:
            yhat = int(float(p.get("IS_VALID", "")))
        except (ValueError, TypeError):
            yhat = -1
        try:
            sc = float(p.get("SCORE", ""))
        except (ValueError, TypeError):
            sc = float(yhat) if yhat >= 0 else 0.5
        labels.append(y); preds.append(yhat); scores.append(sc)
        if y == 0 and yhat == 0:
            rule_pairs.append((g.get("VIOLATED_RULE", ""), p.get("PREDICTED_RULE", "")))

    n = len(labels) or 1
    acc = sum(p == y for p, y in zip(preds, labels)) / n
    # invalid class (label 0) is the positive/anomaly class
    tp = sum((y == 0) and (p == 0) for y, p in zip(labels, preds))
    fp = sum((y == 1) and (p == 0) for y, p in zip(labels, preds))
    fn = sum((y == 0) and (p != 0) for y, p in zip(labels, preds))
    precision, recall, f1 = EM._precision_recall_f1(tp, fp, fn)
    auc = EM._roc_auc(labels, scores)
    # balanced accuracy over the two classes
    pos = [y == 0 for y in labels]
    tpr = (sum((y == 0) and (p == 0) for y, p in zip(labels, preds)) / max(sum(pos), 1))
    tnr = (sum((y == 1) and (p == 1) for y, p in zip(labels, preds)) / max(len(labels) - sum(pos), 1))
    bal_acc = (tpr + tnr) / 2
    rule_attr = (sum(a == b for a, b in rule_pairs) / len(rule_pairs)) if rule_pairs else float("nan")

    return {"task": "anomaly", "n": len(labels),
            "overall": {"accuracy": acc, "balanced_accuracy": bal_acc,
                        "precision": precision, "recall": recall, "f1": f1,
                        "auc": auc, "rule_attr": rule_attr}}


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Score one model's predictions vs ground truth.")
    ap.add_argument("--task", required=True, choices=["next-step", "completion", "anomaly"])
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--ground-truth", required=True)
    a = ap.parse_args()
    fn = {"next-step": score_nextstep, "completion": score_completion, "anomaly": score_anomaly}[a.task]
    print(json.dumps(fn(a.predictions, a.ground_truth), indent=2))
