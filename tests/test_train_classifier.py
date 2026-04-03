from pathlib import Path

from train_classifier import TfidfCentroidClassifier, run_training_pipeline

def test_tfidf_centroid_classifier_learns_simple_labels():
    rows = [
        {"text": "high glucose and elevated hba1c", "label": "diabetes"},
        {"text": "fasting sugar remains high", "label": "diabetes"},
        {"text": "blood pressure remains elevated", "label": "hypertension"},
        {"text": "severe hypertension with headache", "label": "hypertension"},
    ]

    classifier = TfidfCentroidClassifier().fit(rows)

    diabetes_prediction = classifier.predict_one("raised glucose and sugar values")
    hypertension_prediction = classifier.predict_one("patient has elevated blood pressure")

    assert diabetes_prediction["predicted"] == "diabetes"
    assert hypertension_prediction["predicted"] == "hypertension"


def test_run_training_pipeline_writes_model_and_reports(tmp_path):
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()

    train_csv = dataset_dir / "train.csv"
    val_csv = dataset_dir / "val.csv"
    test_csv = dataset_dir / "test.csv"

    train_csv.write_text(
        "text,label\n"
        "\"high glucose and increased thirst\",diabetes\n"
        "\"elevated hba1c and fasting sugar\",diabetes\n"
        "\"high blood pressure and headache\",hypertension\n"
        "\"persistent elevated blood pressure\",hypertension\n",
        encoding="utf-8",
    )
    val_csv.write_text(
        "text,label\n"
        "\"fasting glucose remains high\",diabetes\n"
        "\"blood pressure is elevated today\",hypertension\n",
        encoding="utf-8",
    )
    test_csv.write_text(
        "text,label\n"
        "\"raised sugar levels with thirst\",diabetes\n"
        "\"headache with uncontrolled blood pressure\",hypertension\n",
        encoding="utf-8",
    )

    report_dir = tmp_path / "reports"
    model_out = tmp_path / "models" / "model.json"
    results = run_training_pipeline(train_csv, val_csv, test_csv, model_out, report_dir)

    assert Path(results["model_path"]).exists()
    assert Path(results["summary_path"]).exists()
    assert (report_dir / "validation" / "predictions.csv").exists()
    assert (report_dir / "validation" / "classification_report.txt").exists()
    assert (report_dir / "validation" / "confusion_matrix.svg").exists()
    assert (report_dir / "test" / "classification_report.txt").exists()
