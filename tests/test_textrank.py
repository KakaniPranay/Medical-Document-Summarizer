# tests/test_textrank.py
from summarizer import HybridSummarizer

def test_textrank_basic():
    s = HybridSummarizer()
    text = "This is a test. The patient had fever and cough. The patient improved with treatment. Discharge in good condition."
    summary = s.textrank_extract(text, top_k=2)
    assert isinstance(summary, str) and len(summary) > 0
