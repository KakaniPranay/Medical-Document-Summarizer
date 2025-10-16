# Medical Document Summarizer

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
