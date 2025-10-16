# tests/test_chunker.py
from chunker import chunk_text_by_sentences

def test_chunking_small():
    text = "Sentence one. Sentence two. Sentence three."
    chunks = chunk_text_by_sentences(text, max_words=10, overlap_words=2)
    assert len(chunks) >= 1
    assert "Sentence one" in chunks[0]
