# Dockerfile aligned with the current Flask + NLP/OCR stack
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FLASK_APP=app.py \
    PORT=5000 \
    NLTK_DATA=/usr/local/share/nltk_data

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libgomp1 \
    poppler-utils \
    tesseract-ocr \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt \
 && python -m nltk.downloader -d "${NLTK_DATA}" punkt punkt_tab

COPY . .

EXPOSE 5000
CMD ["sh", "-c", "flask run --host=0.0.0.0 --port=${PORT:-5000}"]
