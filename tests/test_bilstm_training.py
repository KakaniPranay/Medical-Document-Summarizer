from pathlib import Path

from train_bilstm import build_reference_labels, load_training_records


def test_load_training_records_from_sample_jsonl():
    sample_path = Path("tests/data/bilstm_training_sample.jsonl")
    records = load_training_records(sample_path)

    assert len(records) == 3
    assert records[0]["text"]
    assert records[0]["summary"]


def test_build_reference_labels_prefers_oracle_sentence_ids():
    sentences = [
        "Sentence one about symptoms.",
        "Sentence two about treatment.",
        "Sentence three about follow-up.",
    ]
    labels, alignment_scores, target_count = build_reference_labels(
        sentences,
        reference_summary="",
        oracle_sentence_ids=[1, 2],
    )

    assert labels == [0.0, 1.0, 1.0]
    assert alignment_scores == labels
    assert target_count == 2


def test_build_reference_labels_generates_at_least_one_positive():
    sentences = [
        "Patient had chest pain and dizziness.",
        "Blood pressure improved with treatment.",
        "Follow-up was advised after discharge.",
    ]
    labels, alignment_scores, target_count = build_reference_labels(
        sentences,
        reference_summary="Blood pressure improved and follow-up was advised.",
        oracle_sentence_ids=[],
    )

    assert sum(labels) >= 1.0
    assert len(alignment_scores) == len(sentences)
    assert target_count >= 1
