#!/usr/bin/env python3
"""
Plot diagnostic results for official eval prediction files.

This does not compute official scores because the organizer eval inputs are unlabeled.
It creates:
  participant_files/eval_plots/eval_prediction_report.md
  participant_files/eval_plots/*.svg
"""

import html
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.sequence_io import read_csv, split_steps

PF = ROOT / "participant_files"
PRED = PF / "predictions"
OUT = PF / "eval_plots"

EVAL_VALID = PF / "eval_input_valid.csv"
EVAL_ANOMALY = PF / "eval_input_anomaly.csv"

PRED_NEXT = PRED / "predictions_nextstep.csv"
PRED_COMPLETION = PRED / "predictions_completion.csv"
PRED_ANOMALY = PRED / "predictions_anomaly.csv"


def esc(x):
    return html.escape(str(x), quote=True)


def write_svg_bar(path, title, items, max_items=25, width=1100, bar_h=24):
    items = list(items)[:max_items]
    if not items:
        items = [("none", 0)]

    max_val = max(v for _, v in items) or 1
    left = 330
    top = 60
    height = top + 40 + len(items) * bar_h
    plot_w = width - left - 80

    lines = []
    lines.append(
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">' % (width, height)
    )
    lines.append('<rect width="100%" height="100%" fill="white"/>')
    lines.append(
        '<text x="20" y="32" font-family="Arial" font-size="22" font-weight="bold">%s</text>'
        % esc(title)
    )

    for i, (label, value) in enumerate(items):
        y = top + i * bar_h
        bar_w = int(plot_w * value / max_val)
        lines.append(
            '<text x="10" y="%d" font-family="Arial" font-size="12">%s</text>'
            % (y + 16, esc(str(label)[:45]))
        )
        lines.append(
            '<rect x="%d" y="%d" width="%d" height="16" fill="#4C78A8"/>' % (left, y + 3, bar_w)
        )
        lines.append(
            '<text x="%d" y="%d" font-family="Arial" font-size="12">%s</text>'
            % (left + bar_w + 8, y + 16, esc(value))
        )

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_svg_hist(path, title, values, bin_size=5):
    bins = Counter()
    for v in values:
        b = (int(v) // bin_size) * bin_size
        bins["%d-%d" % (b, b + bin_size - 1)] += 1
    write_svg_bar(
        path, title, sorted(bins.items(), key=lambda x: int(x[0].split("-")[0])), max_items=50
    )


def write_md_table(headers, rows):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    valid_rows = read_csv(EVAL_VALID)
    anomaly_input_rows = read_csv(EVAL_ANOMALY)
    next_rows = read_csv(PRED_NEXT)
    completion_rows = read_csv(PRED_COMPLETION)
    anomaly_rows = read_csv(PRED_ANOMALY)

    top1_counts = Counter()
    top5_counts = Counter()

    for row in next_rows:
        if row.get("RANK_1"):
            top1_counts[row["RANK_1"]] += 1

        for col in ["RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"]:
            step = row.get(col, "")
            if step:
                top5_counts[step] += 1

    write_svg_bar(
        OUT / "nextstep_top1_distribution.svg",
        "Next-step Top-1 prediction distribution",
        top1_counts.most_common(25),
    )

    write_svg_bar(
        OUT / "nextstep_top5_distribution.svg",
        "Next-step Top-5 prediction frequency",
        top5_counts.most_common(25),
    )

    completion_lengths = []
    completion_first_steps = Counter()
    completion_last_steps = Counter()

    for row in completion_rows:
        steps = split_steps(row.get("PREDICTED_SEQUENCE", ""), normalize=False)
        completion_lengths.append(len(steps))
        if steps:
            completion_first_steps[steps[0]] += 1
            completion_last_steps[steps[-1]] += 1

    write_svg_hist(
        OUT / "completion_length_distribution.svg",
        "Completion predicted suffix length distribution",
        completion_lengths,
        bin_size=5,
    )

    write_svg_bar(
        OUT / "completion_first_step_distribution.svg",
        "Completion first predicted step distribution",
        completion_first_steps.most_common(25),
    )

    write_svg_bar(
        OUT / "completion_last_step_distribution.svg",
        "Completion last predicted step distribution",
        completion_last_steps.most_common(25),
    )

    validity_counts = Counter()
    rule_counts = Counter()
    score_counts = Counter()

    for row in anomaly_rows:
        is_valid = row.get("IS_VALID", "")
        if is_valid == "1":
            validity_counts["predicted valid"] += 1
        elif is_valid == "0":
            validity_counts["predicted invalid"] += 1
        else:
            validity_counts["invalid label format"] += 1

        rule = row.get("PREDICTED_RULE", "")
        if rule:
            rule_counts[rule] += 1

        score = row.get("SCORE", "")
        if score:
            try:
                s = float(score)
                bucket = int(max(0, min(9, s * 10)))
                score_counts["%.1f-%.1f" % (bucket / 10, (bucket + 1) / 10)] += 1
            except Exception:
                score_counts["invalid score"] += 1

    write_svg_bar(
        OUT / "anomaly_validity_counts.svg",
        "Anomaly predicted valid vs invalid",
        validity_counts.most_common(),
    )

    write_svg_bar(
        OUT / "anomaly_rule_distribution.svg",
        "Anomaly predicted rule distribution",
        rule_counts.most_common(25),
    )

    write_svg_bar(
        OUT / "anomaly_score_distribution.svg",
        "Anomaly score distribution",
        sorted(score_counts.items()),
    )

    report = []
    report.append("# Eval Prediction Diagnostics\n")
    report.append(
        "The organizer eval inputs are unlabeled, so this report does **not** show official accuracy or F1. It only checks and visualizes our generated prediction files.\n"
    )

    report.append("## File Counts\n")
    report.append(
        write_md_table(
            ["File", "Rows including header"],
            [
                ["eval_input_valid.csv", len(valid_rows) + 1],
                ["predictions_nextstep.csv", len(next_rows) + 1],
                ["predictions_completion.csv", len(completion_rows) + 1],
                ["eval_input_anomaly.csv", len(anomaly_input_rows) + 1],
                ["predictions_anomaly.csv", len(anomaly_rows) + 1],
            ],
        )
    )

    report.append("\n## Next-step Predictions\n")
    report.append(
        write_md_table(
            ["Metric", "Value"],
            [
                ["Examples", len(next_rows)],
                ["Unique Top-1 predicted steps", len(top1_counts)],
                ["Unique steps appearing in Top-5", len(top5_counts)],
            ],
        )
    )
    report.append("\n![Top-1 distribution](nextstep_top1_distribution.svg)\n")
    report.append("\n![Top-5 distribution](nextstep_top5_distribution.svg)\n")

    report.append("\n## Sequence Completion\n")
    avg_len = sum(completion_lengths) / len(completion_lengths) if completion_lengths else 0
    report.append(
        write_md_table(
            ["Metric", "Value"],
            [
                ["Examples", len(completion_rows)],
                ["Mean predicted suffix length", "%.2f" % avg_len],
                [
                    "Min predicted suffix length",
                    min(completion_lengths) if completion_lengths else 0,
                ],
                [
                    "Max predicted suffix length",
                    max(completion_lengths) if completion_lengths else 0,
                ],
            ],
        )
    )
    report.append("\n![Completion length distribution](completion_length_distribution.svg)\n")
    report.append(
        "\n![Completion first step distribution](completion_first_step_distribution.svg)\n"
    )
    report.append("\n![Completion last step distribution](completion_last_step_distribution.svg)\n")

    report.append("\n## Anomaly Predictions\n")
    report.append(
        write_md_table(
            ["Metric", "Value"],
            [
                ["Examples", len(anomaly_rows)],
                ["Predicted valid", validity_counts.get("predicted valid", 0)],
                ["Predicted invalid", validity_counts.get("predicted invalid", 0)],
                ["Unique predicted violation rules", len(rule_counts)],
            ],
        )
    )
    report.append("\n![Anomaly validity counts](anomaly_validity_counts.svg)\n")
    report.append("\n![Anomaly rule distribution](anomaly_rule_distribution.svg)\n")
    report.append("\n![Anomaly score distribution](anomaly_score_distribution.svg)\n")

    report.append("\n## Interpretation\n")
    report.append("- Matching row counts mean the prediction files are structurally complete.")
    report.append("- The plots show prediction distributions, not official accuracy.")
    report.append("- Official scores require the hidden organizer ground truth.")
    report.append(
        "- The anomaly predictions are validator-based, so they reflect whether the explicit 10 process rules flag a sequence as invalid."
    )

    report_path = OUT / "eval_prediction_report.md"
    report_path.write_text("\n\n".join(report), encoding="utf-8")

    print("Wrote:", report_path)
    print("Wrote SVG plots to:", OUT)


if __name__ == "__main__":
    main()
