# io_utils.py
import io
import pdfplumber
from docx import Document
from PIL import Image
import pytesseract

def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)

def extract_text_from_docx_bytes(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    paras = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(paras)

def extract_text_from_image_bytes(file_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    text = pytesseract.image_to_string(image)
    return text

def extract_text_from_file(filename: str, file_bytes: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf_bytes(file_bytes)
    if name.endswith(".docx"):
        return extract_text_from_docx_bytes(file_bytes)
    if name.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
        return extract_text_from_image_bytes(file_bytes)
    try:
        return file_bytes.decode("utf-8")
    except Exception:
        return ""
