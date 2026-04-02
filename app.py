# app.py
import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
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
    build_prescription_text,
    build_test_based_monitoring,
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
            fallback_payload["food_habits"] = build_diet_recommendations(signals)
            fallback_payload["lifestyle_plan"] = build_lifestyle_recommendations(signals)
            fallback_payload["monitoring"] = build_test_based_monitoring(signals)
            return render_template("prescription.html",
                                   prescription_text=build_prescription_text(fallback_payload),
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
                               method=method,
                               extractive_seed=extractive_seed,
                               sources=sources,
                               summary_model=model_name,
                               classification_result=classification_result,
                           ))


@app.route("/clear", methods=["POST"])
def clear_stored():
    # Clears stored extracted text and filename from session
    session.pop("extracted_text", None)
    session.pop("uploaded_filename", None)
    flash("Stored file/text cleared.", "info")
    return redirect(url_for("index"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)
