FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    FLASK_APP=app.py \
    PORT=5000 \
    NLTK_DATA=/usr/local/share/nltk_data \
    HF_HOME=/opt/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/opt/huggingface \
    TRANSFORMERS_CACHE=/opt/huggingface \
    HOME=/home/app

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libgomp1 \
    poppler-utils \
    tesseract-ocr \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system app \
 && useradd --system --gid app --create-home --home-dir /home/app app \
 && mkdir -p "${NLTK_DATA}" "${HF_HOME}" /app \
 && chown -R app:app "${NLTK_DATA}" "${HF_HOME}" /app /home/app
COPY requirements.runtime.txt ./

RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.runtime.txt \
 && python -m nltk.downloader -d "${NLTK_DATA}" punkt punkt_tab \
 && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY --chown=app:app app.py chunker.py evaluation.py io_utils.py prescription.py summarizer.py train_classifier.py utils.py vector_store.py ./
COPY --chown=app:app templates ./templates
COPY --chown=app:app static ./static
COPY --chown=app:app models ./models

USER app

EXPOSE 5000

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers ${GUNICORN_WORKERS:-2} --threads ${GUNICORN_THREADS:-4} --timeout ${GUNICORN_TIMEOUT:-120} app:app"]
