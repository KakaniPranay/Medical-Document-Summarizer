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

Generate a classification report and confusion matrix:

1. Prepare a CSV with `actual` and `predicted` columns.
   Example: [`tests/data/classification_eval_sample.csv`](/mnt/c/Users/17791/Videos/Major project/Medical-Document-Summarizer/tests/data/classification_eval_sample.csv)
2. Run:
   ```bash
   python3 evaluation.py \
     --input tests/data/classification_eval_sample.csv \
     --output-dir reports/classification_metrics
   ```
3. Review the generated files:
   `reports/classification_metrics/classification_report.txt`
   `reports/classification_metrics/classification_report.json`
   `reports/classification_metrics/confusion_matrix.csv`
   `reports/classification_metrics/confusion_matrix.svg`

If your CSV uses different column names, pass `--actual-col` and `--predicted-col`. You can also set label order explicitly with `--labels`.

Starter training sets for a classifier:

- Synthetic single-label condition-classification data is available in [`data/condition_classification`](/mnt/c/Users/17791/Videos/Major project/Medical-Document-Summarizer/data/condition_classification/README.md).
- Training split: [`data/condition_classification/train.csv`](/mnt/c/Users/17791/Videos/Major project/Medical-Document-Summarizer/data/condition_classification/train.csv)
- Validation split: [`data/condition_classification/val.csv`](/mnt/c/Users/17791/Videos/Major project/Medical-Document-Summarizer/data/condition_classification/val.csv)
- Test split: [`data/condition_classification/test.csv`](/mnt/c/Users/17791/Videos/Major project/Medical-Document-Summarizer/data/condition_classification/test.csv)

These are useful for bootstrapping a text-classification pipeline, but they are not a substitute for real clinician-labeled medical data.

Train and evaluate the starter classifier:

```bash
python3 train_classifier.py \
  --train data/condition_classification/train.csv \
  --val data/condition_classification/val.csv \
  --test data/condition_classification/test.csv \
  --model-out models/condition_classifier.json \
  --report-dir reports/condition_classifier
```

This generates:
- `models/condition_classifier.json`
- `reports/condition_classifier/validation/predictions.csv`
- `reports/condition_classifier/validation/classification_report.txt`
- `reports/condition_classifier/validation/confusion_matrix.svg`
- `reports/condition_classifier/test/predictions.csv`
- `reports/condition_classifier/test/classification_report.txt`
- `reports/condition_classifier/test/confusion_matrix.svg`
