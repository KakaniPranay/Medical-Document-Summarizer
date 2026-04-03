# tests/test_textrank.py
from summarizer import HybridSummarizer
class DummyVectorStore:
    def search(self, query, top_k=6):
        return [
            ("The patient received medication and improved. Follow-up was advised after discharge.", {"chunk_id": 0}),
            ("Breathing became easier and blood pressure improved.", {"chunk_id": 1}),
        ]

def test_textrank_basic():
    s = HybridSummarizer()
    text = "This is a test. The patient had fever and cough. The patient improved with treatment. Discharge in good condition."
    summary = s.textrank_extract(text, top_k=2)
    assert isinstance(summary, str) and len(summary) > 0

def test_extract_method_returns_plain_language_summary():
    s = HybridSummarizer()
    text = (
        "The patient has hypertension and dyspnea. "
        "Medication was administered and the patient improved. "
        "Follow-up was advised after discharge."
    )
    result = s.summarize(text, method="extractive", on_premise=True, vector_store=None)
    assert isinstance(result, dict)
    assert result["summary"].startswith("- ")
    assert "high blood pressure" in result["summary"].lower()
    assert "trouble breathing" in result["summary"].lower()

def test_bilstm_method_returns_sequence_summary():
    s = HybridSummarizer()
    text = (
        "The patient has hypertension and dyspnea on admission. "
        "Blood pressure remained elevated overnight despite initial medication. "
        "On day two, breathing improved and oxygen support was reduced. "
        "Repeat examination showed improving chest discomfort and stable vitals. "
        "The patient was discharged with follow-up advice and medication instructions."
    )
    result = s.summarize(text, method="bilstm", on_premise=True, vector_store=None)

    assert isinstance(result, dict)
    assert result["seed"]
    assert result["summary"].startswith("- ")
    assert "bilstm" in result["model"].lower()
    assert result["sources"]

def test_methods_have_distinct_fallback_shapes():
    s = HybridSummarizer()
    text = (
        "The patient has hypertension and dyspnea. "
        "Medication was administered and the patient improved. "
        "Follow-up was advised after discharge."
    )
    extractive = s.summarize(text, method="extractive", on_premise=True, vector_store=None)
    abstractive = s.summarize(text, method="abstractive", on_premise=True, vector_store=None)
    hybrid = s.summarize(text, method="hybrid", on_premise=True, vector_store=DummyVectorStore())

    assert extractive["summary"] != abstractive["summary"]
    assert hybrid["summary"] != abstractive["summary"]
    assert extractive["summary"].startswith("- ")
    assert abstractive["summary"].startswith("- ")
    assert hybrid["summary"].startswith("- ")

def test_hybrid_chunk_budget_scales_with_content_size():
    s = HybridSummarizer()
    short_budget = s._determine_chunk_budget("Short clinical note only.", 10)
    long_budget = s._determine_chunk_budget("word " * 2000, 10)
    assert short_budget < long_budget
