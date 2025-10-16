# chunker.py
from nltk.tokenize import sent_tokenize
from typing import List

def chunk_text_by_sentences(text: str, max_words: int = 600, overlap_words: int = 100) -> List[str]:
    sents = sent_tokenize(text)
    chunks = []
    cur = []
    cur_len = 0
    for sent in sents:
        tok_len = len(sent.split())
        if cur_len + tok_len > max_words and cur:
            chunks.append(" ".join(cur))
            # prepare overlap
            if overlap_words > 0:
                overlap = []
                # keep sentences from end of cur until overlap_words satisfied
                while cur and sum(len(s.split()) for s in overlap) < overlap_words:
                    overlap.insert(0, cur.pop())  # take last into overlap
                cur = overlap
                cur_len = sum(len(s.split()) for s in cur)
            else:
                cur = []
                cur_len = 0
        cur.append(sent)
        cur_len += tok_len
    if cur:
        chunks.append(" ".join(cur))
    return chunks
