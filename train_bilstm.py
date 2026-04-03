import argparse
import csv
import json
import random
import re
from pathlib import Path
import torch
import torch.nn as nn
from rouge_score import rouge_scorer

DEFAULT_MODEL_PATH = Path("models/bilstm_extractive.pt")
DEFAULT_REPORT_DIR = Path("reports/bilstm_training")

def _safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _split_sentences(text):
    text = str(text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def _parse_oracle_sentence_ids(value):
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [int(item) for item in parsed]
        except json.JSONDecodeError:
            pass
        items = [item.strip() for item in text.split(",") if item.strip()]
        return [int(item) for item in items]
    return [int(value)]


def load_training_records(input_path):
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Training data file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    records.append(json.loads(text))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            records = payload.get("records", [])
        else:
            records = payload
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError("CSV training data must include a header row.")
            records = list(reader)
    else:
        raise ValueError("Unsupported training data format. Use .jsonl, .json, or .csv.")

    normalized = []
    for index, record in enumerate(records):
        text = str(record.get("text", "")).strip()
        summary = str(record.get("summary", "")).strip()
        oracle_sentence_ids = _parse_oracle_sentence_ids(record.get("oracle_sentence_ids"))
        if not text:
            continue
        if not summary and not oracle_sentence_ids:
            raise ValueError(
                f"Record {index} is missing both `summary` and `oracle_sentence_ids`. "
                "Provide a reference summary or explicit oracle sentence ids."
            )
        normalized.append(
            {
                "text": text,
                "summary": summary,
                "oracle_sentence_ids": oracle_sentence_ids,
            }
        )
    if not normalized:
        raise ValueError("No usable training records were found.")
    return normalized


def build_reference_labels(sentences, reference_summary="", oracle_sentence_ids=None):
    oracle_sentence_ids = oracle_sentence_ids or []
    if oracle_sentence_ids:
        positive_ids = {idx for idx in oracle_sentence_ids if 0 <= idx < len(sentences)}
        if not positive_ids and sentences:
            positive_ids = {0}
        labels = [1.0 if idx in positive_ids else 0.0 for idx in range(len(sentences))]
        return labels, labels[:], max(1, len(positive_ids))

    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    reference_summary = str(reference_summary or "").strip()
    sentence_scores = []
    for sentence in sentences:
        if not reference_summary:
            sentence_scores.append(0.0)
            continue
        rouge_score = scorer.score(reference_summary, sentence)["rouge1"].fmeasure
        sentence_scores.append(float(rouge_score))

    target_count = len(_split_sentences(reference_summary)) or 3
    target_count = max(1, min(target_count, len(sentences)))
    ranked_indices = sorted(range(len(sentences)), key=lambda idx: sentence_scores[idx], reverse=True)
    labels = [0.0 for _ in sentences]
    positives = 0
    for idx in ranked_indices:
        if positives >= target_count:
            break
        if sentence_scores[idx] > 0.0 or positives == 0:
            labels[idx] = 1.0
            positives += 1
    if positives == 0 and labels:
        labels[0] = 1.0
    return labels, sentence_scores, target_count


def prepare_examples(records):
    from summarizer import HybridSummarizer

    summarizer = HybridSummarizer()
    examples = []
    for record in records:
        text = summarizer._preprocess(record["text"])
        sentences = summarizer._sentences_from_text(text)
        if not sentences:
            continue
        textrank_scores = summarizer._textrank_sentence_scores(sentences)
        features, feature_meta = summarizer._sentence_feature_matrix(sentences, textrank_scores)
        labels, alignment_scores, target_count = build_reference_labels(
            sentences,
            reference_summary=record.get("summary", ""),
            oracle_sentence_ids=record.get("oracle_sentence_ids"),
        )
        teacher_targets = []
        for index, meta in enumerate(feature_meta):
            lead_bonus = 1.0 - (index / max(len(sentences) - 1, 1))
            teacher_score = (
                0.45 * meta["textrank_prior"] +
                0.30 * labels[index] +
                0.15 * alignment_scores[index] +
                0.10 * lead_bonus
            )
            teacher_targets.append(float(min(max(teacher_score, 0.0), 1.0)))
        teacher_targets = summarizer._normalize_scores(teacher_targets)
        examples.append(
            {
                "text": text,
                "summary": str(record.get("summary", "")).strip(),
                "sentences": sentences,
                "features": features,
                "labels": labels,
                "teacher_targets": teacher_targets,
                "target_count": target_count,
            }
        )
    if not examples:
        raise ValueError("No training examples produced after sentence extraction.")
    return examples


def split_examples(examples, validation_split=0.2, seed=13):
    items = list(examples)
    random.Random(seed).shuffle(items)
    if len(items) < 2 or validation_split <= 0:
        return items, []
    validation_size = int(round(len(items) * validation_split))
    validation_size = max(1, min(validation_size, len(items) - 1))
    validation_examples = items[:validation_size]
    training_examples = items[validation_size:]
    return training_examples, validation_examples


def _select_summary(sentences, scores, top_k):
    if not sentences:
        return ""
    resolved_top_k = max(1, min(top_k, len(sentences)))
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    selected_indices = sorted(index for index, _ in ranked[:resolved_top_k])
    return " ".join(sentences[index] for index in selected_indices)


def _selection_metrics(predicted_indices, gold_indices):
    predicted = set(predicted_indices)
    gold = set(gold_indices)
    true_positive = len(predicted.intersection(gold))
    precision = _safe_divide(true_positive, len(predicted))
    recall = _safe_divide(true_positive, len(gold))
    f1_score = _safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1_score


def evaluate_model(model, examples):

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
    criterion = nn.BCEWithLogitsLoss()

    losses = []
    rouge1_scores = []
    rouge2_scores = []
    rouge_lsum_scores = []
    precisions = []
    recalls = []
    f1_scores = []

    model.eval()
    with torch.no_grad():
        for example in examples:
            inputs = torch.tensor(example["features"], dtype=torch.float32).unsqueeze(0)
            labels = torch.tensor(example["labels"], dtype=torch.float32).unsqueeze(0)
            logits = model(inputs)
            probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy().tolist()
            losses.append(float(criterion(logits, labels).item()))

            top_k = example["target_count"]
            ranked = sorted(enumerate(probabilities), key=lambda item: item[1], reverse=True)
            predicted_indices = [index for index, _ in ranked[:top_k]]
            gold_indices = [index for index, label in enumerate(example["labels"]) if label >= 0.5]
            precision, recall, f1_score = _selection_metrics(predicted_indices, gold_indices)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)

            if example["summary"]:
                predicted_summary = _select_summary(example["sentences"], probabilities, top_k=top_k)
                rouge_result = scorer.score(example["summary"], predicted_summary)
                rouge1_scores.append(float(rouge_result["rouge1"].fmeasure))
                rouge2_scores.append(float(rouge_result["rouge2"].fmeasure))
                rouge_lsum_scores.append(float(rouge_result["rougeLsum"].fmeasure))

    return {
        "loss": sum(losses) / len(losses) if losses else 0.0,
        "sentence_precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "sentence_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "sentence_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "rouge1_f1": sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
        "rouge2_f1": sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0,
        "rougeLsum_f1": sum(rouge_lsum_scores) / len(rouge_lsum_scores) if rouge_lsum_scores else 0.0,
        "example_count": len(examples),
    }


def _format_metrics_block(title, metrics):
    lines = [title]
    lines.append(f"  loss: {metrics['loss']:.4f}")
    lines.append(f"  sentence_precision: {metrics['sentence_precision']:.4f}")
    lines.append(f"  sentence_recall: {metrics['sentence_recall']:.4f}")
    lines.append(f"  sentence_f1: {metrics['sentence_f1']:.4f}")
    lines.append(f"  rouge1_f1: {metrics['rouge1_f1']:.4f}")
    lines.append(f"  rouge2_f1: {metrics['rouge2_f1']:.4f}")
    lines.append(f"  rougeLsum_f1: {metrics['rougeLsum_f1']:.4f}")
    lines.append(f"  examples: {metrics['example_count']}")
    return "\n".join(lines)


def write_training_report(output_dir, report):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_json_path = output_path / "training_report.json"
    report_text_path = output_path / "training_report.txt"

    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    text_lines = []
    text_lines.append("BiLSTM Extractive Summarizer Training Report")
    text_lines.append("")
    text_lines.append(_format_metrics_block("Best validation metrics", report["best_validation_metrics"]))
    text_lines.append("")
    text_lines.append(_format_metrics_block("Training metrics", report["training_metrics"]))
    if report["validation_metrics"]["example_count"]:
        text_lines.append("")
        text_lines.append(_format_metrics_block("Validation metrics", report["validation_metrics"]))
    text_lines.append("")
    text_lines.append(f"Recommended top_k: {report['recommended_top_k']}")
    text_lines.append(f"Model checkpoint: {report['model_path']}")

    report_text_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    return {
        "report_json": report_json_path,
        "report_text": report_text_path,
    }


def train_model(
    examples,
    output_path,
    report_dir,
    epochs=12,
    learning_rate=1e-3,
    hidden_dim=64,
    num_layers=1,
    dropout=0.1,
    seed=13,
    validation_split=0.2,
):
    from summarizer import BiLSTMSentenceScorer, TORCH_AVAILABLE

    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to train the BiLSTM summarizer.")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    train_examples, validation_examples = split_examples(examples, validation_split=validation_split, seed=seed)
    input_dim = train_examples[0]["features"].shape[1]
    model = BiLSTMSentenceScorer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

    positive_labels = sum(sum(example["labels"]) for example in train_examples)
    total_labels = sum(len(example["labels"]) for example in train_examples)
    negative_labels = max(total_labels - positive_labels, 1.0)
    positive_labels = max(positive_labels, 1.0)
    pos_weight = torch.tensor([negative_labels / positive_labels], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []
    best_model_state = None
    best_validation_score = float("-inf")
    best_validation_metrics = None

    for epoch in range(1, epochs + 1):
        random.shuffle(train_examples)
        model.train()
        total_loss = 0.0

        for example in train_examples:
            inputs = torch.tensor(example["features"], dtype=torch.float32).unsqueeze(0)
            labels = torch.tensor(example["labels"], dtype=torch.float32).unsqueeze(0)
            teacher_targets = torch.tensor(example["teacher_targets"], dtype=torch.float32).unsqueeze(0)

            optimizer.zero_grad()
            logits = model(inputs)
            probabilities = torch.sigmoid(logits)

            classification_loss = criterion(logits, labels)
            teacher_loss = F.mse_loss(probabilities, teacher_targets)
            smoothness = 0.0
            if probabilities.shape[1] > 1:
                smoothness = torch.mean((probabilities[:, 1:] - probabilities[:, :-1]) ** 2)

            loss = classification_loss + (0.15 * teacher_loss) + (0.05 * smoothness)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        training_metrics = evaluate_model(model, train_examples)
        validation_metrics = evaluate_model(model, validation_examples) if validation_examples else training_metrics
        validation_score = validation_metrics["rougeLsum_f1"] + (0.5 * validation_metrics["sentence_f1"])

        history.append(
            {
                "epoch": epoch,
                "average_training_loss": total_loss / len(train_examples),
                "training_metrics": training_metrics,
                "validation_metrics": validation_metrics,
            }
        )

        if validation_score > best_validation_score:
            best_validation_score = validation_score
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_validation_metrics = validation_metrics

    if best_model_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_model_state)
    training_metrics = evaluate_model(model, train_examples)
    validation_metrics = evaluate_model(model, validation_examples) if validation_examples else training_metrics
    recommended_top_k = round(sum(example["target_count"] for example in train_examples) / len(train_examples))
    recommended_top_k = max(1, recommended_top_k)

    output_model_path = Path(output_path)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_name": "bilstm-trained-checkpoint",
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "recommended_top_k": recommended_top_k,
        "state_dict": best_model_state,
        "training_metrics": training_metrics,
        "validation_metrics": validation_metrics,
        "best_validation_metrics": best_validation_metrics or validation_metrics,
    }
    torch.save(checkpoint, output_model_path)

    report = {
        "model_path": str(output_model_path),
        "recommended_top_k": recommended_top_k,
        "training_metrics": training_metrics,
        "validation_metrics": validation_metrics,
        "best_validation_metrics": best_validation_metrics or validation_metrics,
        "history": history,
    }
    report_paths = write_training_report(report_dir, report)
    return {
        "checkpoint_path": output_model_path,
        "report_paths": report_paths,
        "report": report,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a BiLSTM extractive summarizer from labeled document-summary pairs."
    )
    parser.add_argument("--input", required=True, help="Path to .jsonl, .json, or .csv training data.")
    parser.add_argument(
        "--output-model",
        default=str(DEFAULT_MODEL_PATH),
        help="Where to save the trained BiLSTM checkpoint.",
    )
    parser.add_argument(
        "--report-dir",
        default=str(DEFAULT_REPORT_DIR),
        help="Directory where training reports will be written.",
    )
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="BiLSTM hidden size.")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of BiLSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout used inside the sentence scorer head.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for train/validation split and model init.")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of examples to hold out for validation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    records = load_training_records(args.input)
    examples = prepare_examples(records)
    training_result = train_model(
        examples,
        output_path=args.output_model,
        report_dir=args.report_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        seed=args.seed,
        validation_split=args.validation_split,
    )

    print("Training complete.")
    print(f"Checkpoint: {training_result['checkpoint_path']}")
    print(f"Text report: {training_result['report_paths']['report_text']}")
    print(f"JSON report: {training_result['report_paths']['report_json']}")
    print("")
    print(_format_metrics_block("Validation metrics", training_result["report"]["validation_metrics"]))


if __name__ == "__main__":
    main()
