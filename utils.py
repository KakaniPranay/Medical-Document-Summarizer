# utils.py
import re

def clean_text(text: str) -> str:
    # basic cleaning to improve tokenization
    if text is None:
        return ""
    text = text.replace("\r", " ").replace("\u200b", "")
    text = re.sub(r'(?m)^\s*page\s*\d+\s*(of\s*\d+)?\s*$', '', text, flags=re.I)
    text = re.sub(r'-\s*\n', '', text)        # fix hyphenated line breaks
    text = re.sub(r'\s+\n\s+', '\n', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def redact_phi(text: str) -> str:
    # simple regex-based PHI redaction (best-effort)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '[REDACTED_EMAIL]', text)
    text = re.sub(r'\b\d{3}[- ]?\d{3}[- ]?\d{4}\b', '[REDACTED_PHONE]', text)
    text = re.sub(r'\b(MRN|mrn|Patient ID|PID)[\s:]*\d+\b', '[REDACTED_ID]', text, flags=re.I)
    return text
