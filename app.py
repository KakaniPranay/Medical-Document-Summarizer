# app.py
import io
import os
import re
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
from dotenv import load_dotenv
from summarizer import HybridSummarizer
from io_utils import extract_text_from_file
from vector_store import FaissStore
from train_classifier import TfidfCentroidClassifier
from prescription import (
    build_diet_recommendations,
    extract_diagnosis_basis,
    extract_medications_from_text,
    build_fallback_payload,
    build_lifestyle_recommendations,
    build_prescription_pdf,
    build_prescription_text,
    build_test_based_monitoring,
    build_precautions,
    extract_hospital_name,
    extract_patient_details,
    extract_report_signals,
    generate_prescription_result,
)

# Load .env (OPENAI_API_KEY, FLASK_SECRET optional)
load_dotenv()

app = Flask(__name__)
# ensure a secret key is set for session
app.secret_key = os.environ.get("FLASK_SECRET", "supersecretkey123")

# Instantiate components (lightweight constructors; models load lazily)
summarizer = HybridSummarizer()
vector_store = FaissStore()  # keep in memory per process; index will be built per doc
classifier_model_path = os.environ.get("CLASSIFIER_MODEL_PATH", "models/condition_classifier.json")
condition_classifier = None
classifier_load_error = None
prescription_download_cache = {}


def get_condition_classifier():
    global condition_classifier, classifier_load_error
    if condition_classifier is not None:
        return condition_classifier
    if classifier_load_error is not None:
        return None
    if not os.path.exists(classifier_model_path):
        classifier_load_error = f"Classifier model not found at {classifier_model_path}"
        return None
    try:
        condition_classifier = TfidfCentroidClassifier.load(classifier_model_path)
        classifier_load_error = None
        return condition_classifier
    except Exception as exc:
        app.logger.warning("Could not load condition classifier: %s", exc)
        classifier_load_error = str(exc)
        return None


def classify_document(text):
    classifier = get_condition_classifier()
    if not classifier:
        return None
    prediction = classifier.predict_one(text)
    return {
        "label": prediction.get("predicted"),
        "score": float(prediction.get("score", 0.0)),
        "model_path": classifier_model_path,
    }


def build_index_context(**overrides):
    stored_extracted = session.get("extracted_text", "")
    stored_filename = session.get("uploaded_filename", "")
    context = {
        "pasted_text": stored_extracted,
        "extracted_text": stored_extracted if stored_extracted else "",
        "uploaded_filename": stored_filename,
        "summary": None,
        "method": "hybrid",
        "extractive_seed": None,
        "sources": None,
        "summary_model": None,
        "classification_result": None,
        "classification_available": get_condition_classifier() is not None,
        "classification_error": classifier_load_error,
    }
    context.update(overrides)
    return context

def _extract_summary_points(summary):
    if not isinstance(summary, str):
        return []
    points = []
    for line in summary.splitlines():
        cleaned = line.strip()
        if cleaned.startswith("- "):
            points.append(cleaned[2:].strip())
    return points

def _safe_download_stem(prescription):
    patient_name = ''
    if isinstance(prescription, dict):
        patient_details = prescription.get('patient_details') or {}
        patient_name = patient_details.get('name') or ''
    if patient_name and patient_name.lower() != 'not provided':
        stem = 'prescription_' + patient_name.lower().replace(' ', '_')
    else:
        stem = 'prescription_draft'
    stem = re.sub(r'[^a-z0-9_]+', '_', stem.lower()).strip('_')
    return stem or 'prescription_draft'

def _store_latest_prescription(prescription, prescription_text, summary_model):
    previous_key = session.get('latest_prescription_key')
    if previous_key:
        prescription_download_cache.pop(previous_key, None)

    cache_key = str(uuid.uuid4())
    prescription_download_cache[cache_key] = {
        'prescription': prescription,
        'prescription_text': prescription_text,
        'summary_model': summary_model or 'local/default',
    }
    session['latest_prescription_key'] = cache_key

def _clear_latest_prescription():
    cache_key = session.pop('latest_prescription_key', None)
    if cache_key:
        prescription_download_cache.pop(cache_key, None)

def _get_latest_prescription():
    cache_key = session.get('latest_prescription_key')
    if not cache_key:
        return None, None, None
    cached = prescription_download_cache.get(cache_key)
    if not cached:
        return None, None, None
    return (
        cached.get('prescription'),
        cached.get('prescription_text'),
        cached.get('summary_model', 'local/default'),
    )

@app.route("/", methods=["GET"])
def index():
    # if a file was uploaded earlier, prefill textarea with stored extracted text
    return render_template("index.html", **build_index_context())

@app.route("/", methods=["POST"])
def summarize_route():
    # read form inputs
    method = request.form.get("method", "hybrid")
    submit_action = request.form.get("submit_action", "")
    if submit_action == "prescription":
        method = "prescription"
    on_prem = bool(request.form.get("on_prem"))
    redact = bool(request.form.get("redact"))

    # file upload takes precedence
    uploaded = request.files.get("file")
    pasted_text = ""
    extracted_text = ""

    if uploaded and uploaded.filename:
        file_bytes = uploaded.read()
        extracted_text = extract_text_from_file(uploaded.filename, file_bytes)
        if not extracted_text.strip():
            flash("Could not extract text from uploaded file.", "danger")
            return redirect(url_for("index"))
        text_to_summarize = extracted_text

        # ---- Persist extracted text and filename into session so user doesn't need to re-upload ----
        session["extracted_text"] = extracted_text
        session["uploaded_filename"] = uploaded.filename
        # ----------------------------------------------------------------------------------------
    else:
        # If no file uploaded, check whether user pasted text in the textarea
        pasted_text = request.form.get("text", "").strip()
        # If pasted_text is blank but we have stored extracted text in session, reuse it
        if not pasted_text and session.get("extracted_text"):
            text_to_summarize = session.get("extracted_text")
            extracted_text = text_to_summarize
            pasted_text = ""  # keep textarea empty indicator
        else:
            text_to_summarize = pasted_text

    if not text_to_summarize:
        flash("No text provided. Paste text or upload a file.", "warning")
        return redirect(url_for("index"))

    # optional redaction
    if redact:
        from utils import redact_phi
        text_to_summarize = redact_phi(text_to_summarize)

    classification_result = classify_document(text_to_summarize)

    try:
        # Build vector store for this document
        chunks = summarizer.chunk_text(text_to_summarize)
        vector_store.reset()
        vector_store.add_texts(chunks, metadatas=[{"chunk_id": i} for i in range(len(chunks))])

        # run summarization
        if method == "prescription": 
            summary_result = generate_prescription_result( 
                summarizer, 
                text_to_summarize, 
                on_premise=on_prem, 
                vector_store=vector_store, 
            ) 
        else: 
            summary_result = summarizer.summarize( 
                text_to_summarize, 
                method=method, 
                on_premise=on_prem, 
                vector_store=vector_store, 
            ) 

    except Exception as e:
        app.logger.exception("Summarization failed")
        if method == "prescription":
            patient_details = extract_patient_details(text_to_summarize)
            hospital_name = extract_hospital_name(text_to_summarize)
            signals = extract_report_signals(text_to_summarize)
            fallback_payload = build_fallback_payload(
                text_to_summarize[:500],
                patient_details=patient_details,
                hospital_name=hospital_name,
            )
            fallback_payload["diagnosis_basis"] = extract_diagnosis_basis(text_to_summarize, signals)
            fallback_payload["tests_reviewed"] = list(signals.get("tests", []))
            fallback_payload["medications"] = extract_medications_from_text(text_to_summarize, signals)
            fallback_payload["precautions"] = build_precautions(signals, fallback_payload["medications"])
            fallback_payload["food_habits"] = build_diet_recommendations(signals)
            fallback_payload["lifestyle_plan"] = build_lifestyle_recommendations(signals)
            fallback_payload["monitoring"] = build_test_based_monitoring(signals)
            prescription_text = build_prescription_text(fallback_payload)
            _store_latest_prescription(
                fallback_payload,
                prescription_text,
                "embedding-guided-prescription-fallback",
            )
            return render_template("prescription.html",
                                   prescription_text=prescription_text,
                                   prescription=fallback_payload,
                                   summary_model="embedding-guided-prescription-fallback",
                                   classification_result=classification_result,
                                   classification_available=get_condition_classifier() is not None,
                                   classification_error=classifier_load_error)
        flash(f"Summarization error: {e}. Returning extractive fallback.", "danger")
        summary_text = summarizer.textrank_extract(text_to_summarize, top_k=6)
        # Ensure the textarea is prefilled with either pasted text or the extracted text
        display_text = pasted_text if pasted_text else extracted_text
        return render_template("index.html",
                               **build_index_context(
                                   pasted_text=display_text,
                                   extracted_text=extracted_text,
                                   uploaded_filename=session.get("uploaded_filename", ""),
                                   summary=summary_text,
                                   summary_points=_extract_summary_points(summary_text),
                                   method="extractive",
                                   extractive_seed=None,
                                   sources=None,
                                   summary_model=None,
                                   classification_result=classification_result,
                               ))

    # Unpack summary_result
    if isinstance(summary_result, dict):
        summary_text = summary_result.get("summary", "")
        extractive_seed = summary_result.get("seed", "")
        sources = summary_result.get("sources", [])
        model_name = summary_result.get("model", None)
        prescription_data = summary_result.get("prescription")
    else:
        summary_text = str(summary_result)
        extractive_seed = summarizer.textrank_extract(text_to_summarize, top_k=6)
        sources = []
        model_name = None
        prescription_data = None

    if method == "prescription":
        if prescription_data and summary_text:
            _store_latest_prescription(prescription_data, summary_text, model_name)
        return render_template("prescription.html",
                               prescription_text=summary_text,
                               prescription=prescription_data,
                               summary_model=model_name,
                               classification_result=classification_result,
                               classification_available=get_condition_classifier() is not None,
                               classification_error=classifier_load_error)

    # Keep the textarea filled with either user-pasted text or the extracted text
    display_text = pasted_text if pasted_text else extracted_text
    return render_template("index.html",
                           **build_index_context(
                               pasted_text=display_text,
                               extracted_text=extracted_text,
                               uploaded_filename=session.get("uploaded_filename", ""),
                               summary=summary_text,
                               summary_points=_extract_summary_points(summary_text),
                               method=method,
                               extractive_seed=extractive_seed,
                               sources=sources,
                               summary_model=model_name,
                               classification_result=classification_result,
                           ))


@app.route("/prescription/download.txt", methods=["GET"])
def download_prescription_text():
    prescription, prescription_text, _ = _get_latest_prescription()
    if not prescription or not prescription_text:
        flash("Generate a prescription draft before downloading it.", "warning")
        return redirect(url_for("index"))
    download_name = _safe_download_stem(prescription) + ".txt"
    return send_file(
        io.BytesIO(prescription_text.encode("utf-8")),
        mimetype="text/plain; charset=utf-8",
        as_attachment=True,
        download_name=download_name,
    )


@app.route("/prescription/download.pdf", methods=["GET"])
def download_prescription_pdf():
    prescription, _, _ = _get_latest_prescription()
    if not prescription:
        flash("Generate a prescription draft before downloading it.", "warning")
        return redirect(url_for("index"))
    try:
        pdf_bytes = build_prescription_pdf(prescription)
    except Exception as exc:
        app.logger.warning("Prescription PDF generation failed: %s", exc)
        flash(f"Could not generate PDF: {exc}", "danger")
        return redirect(url_for("index"))
    download_name = _safe_download_stem(prescription) + ".pdf"
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=download_name,
    )


@app.route("/clear", methods=["POST"])
def clear_stored():
    # Clears stored extracted text and filename from session
    session.pop("extracted_text", None)
    session.pop("uploaded_filename", None)
    _clear_latest_prescription()
    flash("Stored file/text cleared.", "info")
    return redirect(url_for("index"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)
