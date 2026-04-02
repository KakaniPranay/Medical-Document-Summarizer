# Medical Document Summarizer

This app can generate either a report summary or a conservative draft prescription for clinician review based on the uploaded medical report. Select the mode from the web UI before you submit the document.

Available summary modes include extractive TextRank, a BiLSTM-based extractive sentence ranker, hybrid retrieval plus abstraction, and optional LLM-backed abstraction when configured.

Train the BiLSTM summarizer:

1. Prepare training data as `.jsonl`, `.json`, or `.csv` with:
   - `text`: full source document
   - `summary`: reference summary
   - optional `oracle_sentence_ids`: list of extractive sentence indices to supervise directly
2. Example dataset:
   `tests/data/bilstm_training_sample.jsonl`
3. Run training:
   ```bash
   python3 train_bilstm.py \
     --input tests/data/bilstm_training_sample.jsonl \
     --output-model models/bilstm_extractive.pt \
     --report-dir reports/bilstm_training
   ```
4. The web app will automatically use `models/bilstm_extractive.pt` for `bilstm` summaries when the checkpoint exists.

Run locally:

1. Create venv and install:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ````
2. Download NLTK punkt:
   ```
   python -c "import nltk; nltk.download('punkt')"
   ```
3. (Optional) Install spaCy model:
   ```
   python -m spacy download en_core_web_sm
   ```
4. Install system packages (Ubuntu):
   ```
   sudo apt install -y tesseract-ocr poppler-utils
   ```
5. Run:
   ```
   export FLASK_APP=app.py
   flask run
   ```
6. Open http://127.0.0.1:5000
