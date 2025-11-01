# app.py
import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
from dotenv import load_dotenv
from summarizer import HybridSummarizer
from io_utils import extract_text_from_file
from vector_store import FaissStore

# Load .env (OPENAI_API_KEY, FLASK_SECRET optional)
load_dotenv()

app = Flask(__name__)
# ensure a secret key is set for session
app.secret_key = os.environ.get("FLASK_SECRET", "supersecretkey123")

# Instantiate components (lightweight constructors; models load lazily)
summarizer = HybridSummarizer()
vector_store = FaissStore()  # keep in memory per process; index will be built per doc

@app.route("/", methods=["GET"])
def index():
    # if a file was uploaded earlier, prefill textarea with stored extracted text
    stored_extracted = session.get("extracted_text", "")
    stored_filename = session.get("uploaded_filename", "")
    return render_template("index.html",
                           pasted_text=stored_extracted,
                           extracted_text=stored_extracted if stored_extracted else "",
                           uploaded_filename=stored_filename,
                           summary=None,
                           method="hybrid",
                           extractive_seed=None,
                           sources=None,
                           summary_model=None)

@app.route("/", methods=["POST"])
def summarize_route():
    # read form inputs
    method = request.form.get("method", "hybrid")
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

    # Build vector store for this document
    chunks = summarizer.chunk_text(text_to_summarize)
    vector_store.reset()
    vector_store.add_texts(chunks, metadatas=[{"chunk_id": i} for i in range(len(chunks))])

    # run summarization
    try:
        summary_result = summarizer.summarize(text_to_summarize, method=method, on_premise=on_prem, vector_store=vector_store)
    except Exception as e:
        app.logger.exception("Summarization failed")
        flash(f"Summarization error: {e}. Returning extractive fallback.", "danger")
        summary_text = summarizer.textrank_extract(text_to_summarize, top_k=6)
        # Ensure the textarea is prefilled with either pasted text or the extracted text
        display_text = pasted_text if pasted_text else extracted_text
        return render_template("index.html",
                               pasted_text=display_text,
                               extracted_text=extracted_text,
                               uploaded_filename=session.get("uploaded_filename", ""),
                               summary=summary_text,
                               method="extractive",
                               extractive_seed=None,
                               sources=None,
                               summary_model=None)

    # Unpack summary_result
    if isinstance(summary_result, dict):
        summary_text = summary_result.get("summary", "")
        extractive_seed = summary_result.get("seed", "")
        sources = summary_result.get("sources", [])
        model_name = summary_result.get("model", None)
    else:
        summary_text = str(summary_result)
        extractive_seed = summarizer.textrank_extract(text_to_summarize, top_k=6)
        sources = []
        model_name = None

    # Keep the textarea filled with either user-pasted text or the extracted text
    display_text = pasted_text if pasted_text else extracted_text
    return render_template("index.html",
                           pasted_text=display_text,
                           extracted_text=extracted_text,
                           uploaded_filename=session.get("uploaded_filename", ""),
                           summary=summary_text,
                           method=method,
                           extractive_seed=extractive_seed,
                           sources=sources,
                           summary_model=model_name)


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
