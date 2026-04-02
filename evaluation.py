import argparse
import csv
import json
from pathlib import Path


def load_labels(csv_path, actual_col="actual", predicted_col="predicted"):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Input CSV must include a header row.")
        missing = [col for col in (actual_col, predicted_col) if col not in reader.fieldnames]
        if missing:
            raise ValueError(
                "Missing required column(s): " + ", ".join(missing) +
                ". Available columns: " + ", ".join(reader.fieldnames)
            )
        for row in reader:
            actual = str(row.get(actual_col, "")).strip()
            predicted = str(row.get(predicted_col, "")).strip()
            if not actual or not predicted:
                continue
            rows.append((actual, predicted))
    if not rows:
        raise ValueError("No non-empty label rows were found in the input CSV.")
    return rows


def _safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def infer_labels(rows, labels=None):
    if labels:
        return [label.strip() for label in labels if label.strip()]
    discovered = []
    seen = set()
    for actual, predicted in rows:
        for label in (actual, predicted):
            if label not in seen:
                seen.add(label)
                discovered.append(label)
    return discovered


def build_confusion_matrix(rows, labels):
    index_by_label = {label: idx for idx, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    skipped = []

    for actual, predicted in rows:
        actual_idx = index_by_label.get(actual)
        predicted_idx = index_by_label.get(predicted)
        if actual_idx is None or predicted_idx is None:
            skipped.append((actual, predicted))
            continue
        matrix[actual_idx][predicted_idx] += 1

    return matrix, skipped


def _row_sum(matrix, row_index):
    return sum(matrix[row_index])


def _column_sum(matrix, column_index):
    return sum(row[column_index] for row in matrix)


def build_classification_report(rows, labels=None):
    resolved_labels = infer_labels(rows, labels=labels)
    matrix, skipped = build_confusion_matrix(rows, resolved_labels)

    per_label = []
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[idx][idx] for idx in range(len(resolved_labels)))

    for idx, label in enumerate(resolved_labels):
        tp = matrix[idx][idx]
        fp = _column_sum(matrix, idx) - tp
        fn = _row_sum(matrix, idx) - tp
        support = _row_sum(matrix, idx)
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1_score = _safe_divide(2 * precision * recall, precision + recall)
        per_label.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "support": support,
            }
        )

    accuracy = _safe_divide(correct, total)
    macro_avg = {
        "precision": _safe_divide(sum(item["precision"] for item in per_label), len(per_label)),
        "recall": _safe_divide(sum(item["recall"] for item in per_label), len(per_label)),
        "f1_score": _safe_divide(sum(item["f1_score"] for item in per_label), len(per_label)),
        "support": total,
    }
    weighted_avg = {
        "precision": _safe_divide(sum(item["precision"] * item["support"] for item in per_label), total),
        "recall": _safe_divide(sum(item["recall"] * item["support"] for item in per_label), total),
        "f1_score": _safe_divide(sum(item["f1_score"] * item["support"] for item in per_label), total),
        "support": total,
    }

    report = {
        "labels": resolved_labels,
        "per_label": per_label,
        "accuracy": accuracy,
        "macro_avg": macro_avg,
        "weighted_avg": weighted_avg,
        "confusion_matrix": matrix,
        "total_samples": total,
        "skipped_rows": skipped,
    }
    return report


def format_classification_report(report):
    lines = []
    lines.append(f"{'label':<22}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}")
    lines.append("")
    for item in report["per_label"]:
        lines.append(
            f"{item['label']:<22}"
            f"{item['precision']:>10.2f}"
            f"{item['recall']:>10.2f}"
            f"{item['f1_score']:>10.2f}"
            f"{item['support']:>10d}"
        )
    lines.append("")
    lines.append(
        f"{'accuracy':<22}"
        f"{'':>10}"
        f"{'':>10}"
        f"{report['accuracy']:>10.2f}"
        f"{report['total_samples']:>10d}"
    )
    lines.append(
        f"{'macro avg':<22}"
        f"{report['macro_avg']['precision']:>10.2f}"
        f"{report['macro_avg']['recall']:>10.2f}"
        f"{report['macro_avg']['f1_score']:>10.2f}"
        f"{report['macro_avg']['support']:>10d}"
    )
    lines.append(
        f"{'weighted avg':<22}"
        f"{report['weighted_avg']['precision']:>10.2f}"
        f"{report['weighted_avg']['recall']:>10.2f}"
        f"{report['weighted_avg']['f1_score']:>10.2f}"
        f"{report['weighted_avg']['support']:>10d}"
    )
    if report["skipped_rows"]:
        lines.append("")
        lines.append(f"Skipped rows outside label set: {len(report['skipped_rows'])}")
    return "\n".join(lines)


def write_confusion_matrix_csv(output_path, labels, matrix):
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted"] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + row)


def _escape_xml(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_confusion_matrix_svg(output_path, labels, matrix, title="Confusion Matrix"):
    cell_size = 72
    left_margin = 180
    top_margin = 120
    right_margin = 30
    bottom_margin = 40
    matrix_size = len(labels)
    width = left_margin + matrix_size * cell_size + right_margin
    height = top_margin + matrix_size * cell_size + bottom_margin
    max_count = max((value for row in matrix for value in row), default=0)

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        'text { font-family: Arial, sans-serif; fill: #16324f; }',
        '.title { font-size: 20px; font-weight: bold; }',
        '.axis { font-size: 12px; font-weight: bold; }',
        '.tick { font-size: 12px; }',
        '.cell-value { font-size: 14px; font-weight: bold; text-anchor: middle; dominant-baseline: middle; }',
        '</style>',
        f'<text class="title" x="{left_margin}" y="36">{_escape_xml(title)}</text>',
        f'<text class="axis" x="{left_margin + (matrix_size * cell_size) / 2}" y="70" text-anchor="middle">Predicted label</text>',
        (
            f'<text class="axis" x="28" y="{top_margin + (matrix_size * cell_size) / 2}" '
            'transform="rotate(-90 28 '
            f'{top_margin + (matrix_size * cell_size) / 2})" text-anchor="middle">Actual label</text>'
        ),
    ]

    for col_idx, label in enumerate(labels):
        x = left_margin + (col_idx * cell_size) + (cell_size / 2)
        svg_lines.append(f'<text class="tick" x="{x}" y="{top_margin - 16}" text-anchor="middle">{_escape_xml(label)}</text>')

    for row_idx, label in enumerate(labels):
        y = top_margin + (row_idx * cell_size) + (cell_size / 2) + 4
        svg_lines.append(f'<text class="tick" x="{left_margin - 14}" y="{y}" text-anchor="end">{_escape_xml(label)}</text>')

    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            x = left_margin + col_idx * cell_size
            y = top_margin + row_idx * cell_size
            intensity = _safe_divide(value, max_count) if max_count else 0.0
            shade = int(245 - (intensity * 140))
            fill = f"rgb({shade}, {shade + 5}, 255)"
            svg_lines.append(
                f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" '
                f'fill="{fill}" stroke="#8ea3b5" stroke-width="1" />'
            )
            svg_lines.append(
                f'<text class="cell-value" x="{x + (cell_size / 2)}" y="{y + (cell_size / 2)}">{value}</text>'
            )

    svg_lines.append("</svg>")
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(svg_lines))


def write_report_files(report, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_text = format_classification_report(report)
    report_json = {
        "labels": report["labels"],
        "per_label": report["per_label"],
        "accuracy": report["accuracy"],
        "macro_avg": report["macro_avg"],
        "weighted_avg": report["weighted_avg"],
        "confusion_matrix": report["confusion_matrix"],
        "total_samples": report["total_samples"],
        "skipped_rows": report["skipped_rows"],
    }

    report_text_path = output_path / "classification_report.txt"
    report_json_path = output_path / "classification_report.json"
    matrix_csv_path = output_path / "confusion_matrix.csv"
    matrix_svg_path = output_path / "confusion_matrix.svg"

    report_text_path.write_text(report_text + "\n", encoding="utf-8")
    report_json_path.write_text(json.dumps(report_json, indent=2), encoding="utf-8")
    write_confusion_matrix_csv(matrix_csv_path, report["labels"], report["confusion_matrix"])
    write_confusion_matrix_svg(matrix_svg_path, report["labels"], report["confusion_matrix"])

    return {
        "report_text": report_text_path,
        "report_json": report_json_path,
        "matrix_csv": matrix_csv_path,
        "matrix_svg": matrix_svg_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a classification report and confusion matrix from a CSV of actual and predicted labels."
    )
    parser.add_argument("--input", required=True, help="Path to the CSV file containing labels.")
    parser.add_argument("--actual-col", default="actual", help="CSV column containing ground-truth labels.")
    parser.add_argument("--predicted-col", default="predicted", help="CSV column containing predicted labels.")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional explicit label order. If omitted, labels are inferred from the CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/classification_metrics",
        help="Directory where the report and confusion matrix files will be saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_labels(args.input, actual_col=args.actual_col, predicted_col=args.predicted_col)
    report = build_classification_report(rows, labels=args.labels)
    paths = write_report_files(report, args.output_dir)

    print(format_classification_report(report))
    print("")
    print("Saved files:")
    print(f"- {paths['report_text']}")
    print(f"- {paths['report_json']}")
    print(f"- {paths['matrix_csv']}")
    print(f"- {paths['matrix_svg']}")


if __name__ == "__main__":
    main()
