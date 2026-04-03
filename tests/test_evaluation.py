from pathlib import Path

from evaluation import build_classification_report, format_classification_report, write_report_files

def test_build_classification_report_metrics():
    rows = [
        ("summary", "summary"),
        ("summary", "prescription"),
        ("prescription", "prescription"),
        ("prescription", "prescription"),
    ]

    report = build_classification_report(rows, labels=["summary", "prescription"])

    assert report["confusion_matrix"] == [[1, 1], [0, 2]]
    assert round(report["accuracy"], 2) == 0.75

    summary_metrics = report["per_label"][0]
    assert round(summary_metrics["precision"], 2) == 1.00
    assert round(summary_metrics["recall"], 2) == 0.50
    assert round(summary_metrics["f1_score"], 2) == 0.67

    report_text = format_classification_report(report)
    assert "weighted avg" in report_text
    assert "summary" in report_text


def test_write_report_files_creates_outputs(tmp_path):
    rows = [
        ("normal", "normal"),
        ("normal", "abnormal"),
        ("abnormal", "abnormal"),
    ]

    report = build_classification_report(rows, labels=["normal", "abnormal"])
    output_dir = tmp_path / "metrics"
    paths = write_report_files(report, output_dir)

    for path in paths.values():
        assert Path(path).exists()

    assert "accuracy" in (output_dir / "classification_report.txt").read_text(encoding="utf-8")
    assert "actual/predicted" in (output_dir / "confusion_matrix.csv").read_text(encoding="utf-8")
    assert "<svg" in (output_dir / "confusion_matrix.svg").read_text(encoding="utf-8")
