import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from evaluation import build_classification_report, format_classification_report, write_report_files


TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


def read_dataset(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{csv_path} must include a header row.")
        missing = [column for column in ("text", "label") if column not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {', '.join(missing)}. "
                f"Available columns: {', '.join(reader.fieldnames)}"
            )
        for row in reader:
            text = str(row.get("text", "")).strip()
            label = str(row.get("label", "")).strip()
            if not text or not label:
                continue
            rows.append({"text": text, "label": label})
    if not rows:
        raise ValueError(f"{csv_path} did not contain any usable rows.")
    return rows


def write_predictions_csv(rows, predictions, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "actual", "predicted", "score"])
        writer.writeheader()
        for row, prediction in zip(rows, predictions):
            writer.writerow(
                {
                    "text": row["text"],
                    "actual": row["label"],
                    "predicted": prediction["predicted"],
                    "score": f"{prediction['score']:.4f}",
                }
            )


class TfidfCentroidClassifier:
    def __init__(self):
        self.labels = []
        self.idf = {}
        self.centroids = {}
        self.majority_label = None
        self.training_examples = 0

    def tokenize(self, text):
        return TOKEN_PATTERN.findall((text or "").lower())

    def _normalize(self, vector):
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm == 0:
            return {}
        return {term: value / norm for term, value in vector.items()}

    def _vectorize_tokens(self, tokens):
        counts = Counter(token for token in tokens if token in self.idf)
        total = sum(counts.values())
        if total == 0:
            return {}
        vector = {}
        for term, count in counts.items():
            tf = count / total
            vector[term] = tf * self.idf[term]
        return self._normalize(vector)

    def fit(self, rows):
        if not rows:
            raise ValueError("Training data cannot be empty.")

        doc_frequencies = Counter()
        label_counts = Counter()
        vector_rows = []

        for row in rows:
            label = row["label"]
            tokens = self.tokenize(row["text"])
            if not tokens:
                continue
            label_counts[label] += 1
            doc_frequencies.update(set(tokens))
            vector_rows.append((label, tokens))

        if not vector_rows:
            raise ValueError("Training data does not contain any tokenizable examples.")

        self.training_examples = len(vector_rows)
        self.labels = sorted(label_counts.keys())
        self.majority_label = label_counts.most_common(1)[0][0]
        document_count = len(vector_rows)
        self.idf = {
            term: math.log((1 + document_count) / (1 + frequency)) + 1.0
            for term, frequency in doc_frequencies.items()
        }

        centroid_sums = {label: defaultdict(float) for label in self.labels}
        for label, tokens in vector_rows:
            vector = self._vectorize_tokens(tokens)
            for term, value in vector.items():
                centroid_sums[label][term] += value

        self.centroids = {}
        for label in self.labels:
            count = label_counts[label]
            averaged = {
                term: value / count
                for term, value in centroid_sums[label].items()
            }
            self.centroids[label] = self._normalize(averaged)

        return self

    def predict_one(self, text):
        vector = self._vectorize_tokens(self.tokenize(text))
        if not vector:
            return {"predicted": self.majority_label, "score": 0.0}

        best_label = self.majority_label
        best_score = -1.0
        for label in self.labels:
            centroid = self.centroids.get(label, {})
            score = sum(value * centroid.get(term, 0.0) for term, value in vector.items())
            if score > best_score:
                best_score = score
                best_label = label

        return {"predicted": best_label, "score": best_score}

    def predict_rows(self, rows):
        return [self.predict_one(row["text"]) for row in rows]

    def to_dict(self):
        return {
            "model_type": "tfidf_centroid_classifier",
            "labels": self.labels,
            "majority_label": self.majority_label,
            "training_examples": self.training_examples,
            "idf": self.idf,
            "centroids": self.centroids,
        }

    def save(self, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data):
        classifier = cls()
        classifier.labels = list(data.get("labels", []))
        classifier.idf = {str(term): float(value) for term, value in data.get("idf", {}).items()}
        classifier.centroids = {
            str(label): {str(term): float(value) for term, value in centroid.items()}
            for label, centroid in data.get("centroids", {}).items()
        }
        classifier.majority_label = data.get("majority_label")
        classifier.training_examples = int(data.get("training_examples", 0))
        return classifier

    @classmethod
    def load(cls, model_path):
        model_path = Path(model_path)
        data = json.loads(model_path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


def evaluate_split(classifier, rows, split_name, report_root):
    predictions = classifier.predict_rows(rows)
    prediction_rows = [(row["label"], prediction["predicted"]) for row, prediction in zip(rows, predictions)]

    split_dir = Path(report_root) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    write_predictions_csv(rows, predictions, split_dir / "predictions.csv")
    report = build_classification_report(prediction_rows, labels=classifier.labels)
    artifact_paths = write_report_files(report, split_dir)

    return {
        "accuracy": report["accuracy"],
        "report": report,
        "report_text": format_classification_report(report),
        "artifact_paths": artifact_paths,
        "prediction_path": split_dir / "predictions.csv",
    }


def run_training_pipeline(train_path, val_path, test_path, model_out, report_dir):
    train_rows = read_dataset(train_path)
    val_rows = read_dataset(val_path)
    test_rows = read_dataset(test_path)

    classifier = TfidfCentroidClassifier().fit(train_rows)
    classifier.save(model_out)

    results = {
        "model_path": Path(model_out),
        "labels": classifier.labels,
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "test_examples": len(test_rows),
        "validation": evaluate_split(classifier, val_rows, "validation", report_dir),
        "test": evaluate_split(classifier, test_rows, "test", report_dir),
    }

    summary = {
        "model_path": str(results["model_path"]),
        "labels": results["labels"],
        "train_examples": results["train_examples"],
        "val_examples": results["val_examples"],
        "test_examples": results["test_examples"],
        "validation_accuracy": results["validation"]["accuracy"],
        "test_accuracy": results["test"]["accuracy"],
    }
    summary_path = Path(report_dir) / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    results["summary_path"] = summary_path

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a lightweight medical condition classifier and generate evaluation artifacts."
    )
    parser.add_argument(
        "--train",
        default="data/condition_classification/train.csv",
        help="Path to the training CSV with text and label columns.",
    )
    parser.add_argument(
        "--val",
        default="data/condition_classification/val.csv",
        help="Path to the validation CSV with text and label columns.",
    )
    parser.add_argument(
        "--test",
        default="data/condition_classification/test.csv",
        help="Path to the test CSV with text and label columns.",
    )
    parser.add_argument(
        "--model-out",
        default="models/condition_classifier.json",
        help="Path where the trained model artifact will be saved.",
    )
    parser.add_argument(
        "--report-dir",
        default="reports/condition_classifier",
        help="Directory where predictions, reports, and confusion matrices will be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_training_pipeline(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        model_out=args.model_out,
        report_dir=args.report_dir,
    )

    print(f"Saved model: {results['model_path']}")
    print(f"Saved summary: {results['summary_path']}")
    print("")
    print("Validation report:")
    print(results["validation"]["report_text"])
    print("")
    print("Test report:")
    print(results["test"]["report_text"])


if __name__ == "__main__":
    main()
