# summarizer.py
import os
import logging
import re
from typing import List, Dict, Any
import nltk
from nltk.tokenize import sent_tokenize
import networkx as nx
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

# optional heavy imports handled lazily
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# OpenAI handled lazily inside class if API key is present
try:
    import openai  # noqa: F401
    OPENAI_PRESENT = True
except Exception:
    OPENAI_PRESENT = False

# local helpers / modules
from chunker import chunk_text_by_sentences
from utils import clean_text

# sentence tokenizer
nltk.download("punkt", quiet=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "he", "in", "is", "it", "its", "of", "on", "or", "she", "that", "the", "their",
    "this", "to", "was", "were", "with",
}

PLAIN_LANGUAGE_REPLACEMENTS = {
    r"\bhypertension\b": "high blood pressure",
    r"\bhypotension\b": "low blood pressure",
    r"\bdiabetes mellitus\b": "diabetes",
    r"\bmyocardial infarction\b": "heart attack",
    r"\bdyspnea\b": "trouble breathing",
    r"\bshortness of breath\b": "trouble breathing",
    r"\bchest pain\b": "pain in the chest",
    r"\bhemorrhage\b": "bleeding",
    r"\bedema\b": "swelling",
    r"\blesion\b": "abnormal area",
    r"\bmalignancy\b": "cancer",
    r"\bbenign\b": "not cancer",
    r"\bneoplasm\b": "tumor",
    r"\bfracture\b": "broken bone",
    r"\brenal\b": "kidney",
    r"\bhepatic\b": "liver",
    r"\bpulmonary\b": "lung",
    r"\bcardiac\b": "heart",
    r"\bneurological\b": "brain and nerve",
    r"\bmetastasis\b": "spread of cancer",
    r"\bprognosis\b": "expected outlook",
    r"\binfection\b": "infection",
    r"\binflammatory\b": "caused by inflammation",
    r"\bacute\b": "sudden",
    r"\bchronic\b": "long-term",
    r"\bnegative for\b": "did not show",
    r"\bpositive for\b": "showed",
    r"\badministered\b": "given",
    r"\bdischarged\b": "sent home from care",
    r"\badvised\b": "told",
    r"\bfollow-up\b": "follow-up visit",
}


if TORCH_AVAILABLE:
    class BiLSTMSentenceScorer(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 1, dropout: float = 0.1):
            super().__init__()
            recurrent_dropout = dropout if num_layers > 1 else 0.0
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=recurrent_dropout,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x):
            encoded, _ = self.encoder(x)
            return self.head(encoded).squeeze(-1)


class HybridSummarizer:
    def __init__(self,
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 abstractive_model_name: str = "sshleifer/distilbart-cnn-12-6"):
        # Try to load SentenceTransformer embedder (optional)
        self.embedder = None
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(embedding_model_name)
            logger.info(f"Loaded embedder: {embedding_model_name}")
        except Exception as e:
            logger.warning(f"SentenceTransformer load failed or not installed: {e}")
            self.embedder = None

        self.abstractive_model_name = abstractive_model_name
        self.abstractive_pipeline = None  # will be created lazily if requested
        self.bilstm_hidden_dim = 32
        self.bilstm_epochs = 35
        self.bilstm_learning_rate = 0.03
        self.bilstm_teacher_weight = 0.35
        self.bilstm_model_path = os.environ.get("BILSTM_MODEL_PATH", os.path.join("models", "bilstm_extractive.pt"))
        self.bilstm_model = None
        self.bilstm_model_input_dim = None
        self.bilstm_model_meta = {}

        # Setup OpenAI client lazily if API key present
        self.openai = None
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            try:
                import openai as _openai
                _openai.api_key = openai_key
                self.openai = _openai
                logger.info("OpenAI configured (key found in env).")
            except Exception as e:
                logger.warning(f"OpenAI import/config failed: {e}")
                self.openai = None

    # -------------------------
    # Basic preprocessing + chunking
    # -------------------------
    def _preprocess(self, text: str) -> str:
        return clean_text(text)

    def _sentence_case(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        return text[0].upper() + text[1:]

    def _simplify_medical_language(self, text: str) -> str:
        simplified = " " + (text or "").strip() + " "
        for pattern, replacement in PLAIN_LANGUAGE_REPLACEMENTS.items():
            simplified = re.sub(pattern, replacement, simplified, flags=re.IGNORECASE)
        simplified = re.sub(r"\s+", " ", simplified).strip()
        return self._sentence_case(simplified)

    def _plain_language_prompt(self, text: str) -> str:
        return (
            "Rewrite the following medical report in plain language for a patient or family member. "
            "Use short, clear sentences. Explain the main problem, important findings, treatment given, "
            "and what follow-up may be needed. Do not add facts that are not in the source. "
            "Avoid medical jargon when possible, and if a medical term must be used, explain it simply. "
            "Write the answer as short bullet points, not paragraphs.\n\n"
            f"Source text:\n{text}"
        )

    def _format_bullet_summary(self, items: List[str], heading: str = "") -> str:
        cleaned_items = []
        for item in items:
            text = self._simplify_medical_language(item).strip()
            normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
            if not text or not normalized:
                continue

            duplicate = False
            for entry in cleaned_items:
                existing = re.sub(r"[^a-z0-9]+", " ", entry.lower()).strip()
                if normalized == existing or normalized in existing or existing in normalized:
                    duplicate = True
                    break
            if not duplicate:
                cleaned_items.append(text)

        if not cleaned_items:
            cleaned_items.append("The report does not contain enough clear information to create a simple summary.")

        lines = [heading] if heading else []
        for item in cleaned_items:
            lines.append("- " + item)
        return "\n".join(lines)

    def _bulletize_text(self, text: str, limit: int = 4, heading: str = "") -> str:
        return self._format_bullet_summary(self._select_plain_sentences(text, limit=limit), heading)

    def _select_plain_sentences(self, text: str, limit: int = 4) -> List[str]:
        sentences = self._sentences_from_text(text)
        if not sentences:
            return []

        selected = []
        for sentence in sentences:
            simplified = self._simplify_medical_language(sentence)
            lowered = simplified.lower()
            if lowered not in [item.lower() for item in selected]:
                selected.append(simplified)
            if len(selected) == limit:
                break
        return selected

    def _plain_language_fallback(self, text: str, seed: str = "") -> str:
        source = seed or text
        return self._format_bullet_summary(self._select_plain_sentences(source, limit=4))

    def _extractive_patient_summary(self, text: str, seed: str = "") -> str:
        return self._format_bullet_summary(self._select_plain_sentences(seed or text, limit=4))

    def _abstractive_patient_summary(self, text: str, seed: str = "") -> str:
        return self._plain_language_fallback(text, seed=seed)

    def _hybrid_patient_summary(self, seed: str, sources_with_points: List[Dict[str, Any]]) -> str:
        evidence_sentences = []
        for source in sources_with_points:
            for point in source.get("points", []):
                simplified = self._simplify_medical_language(point)
                if simplified.lower() not in [item.lower() for item in evidence_sentences]:
                    evidence_sentences.append(simplified)
                if len(evidence_sentences) == 4:
                    break
            if len(evidence_sentences) == 4:
                break

        main_issue = self._select_plain_sentences(seed, limit=1)
        details = evidence_sentences[:3] or self._select_plain_sentences(seed, limit=3)
        follow_up = evidence_sentences[3:4]

        points = []
        if main_issue:
            points.extend(main_issue)
        if details:
            points.extend(details)
        if follow_up:
            points.extend(follow_up)
        return self._format_bullet_summary(points)

    def chunk_text(self, text: str, max_words: int = 600, overlap_words: int = 100) -> List[str]:
        text = self._preprocess(text)
        return chunk_text_by_sentences(text, max_words=max_words, overlap_words=overlap_words)

    def _determine_chunk_budget(self, text: str, available_chunks: int) -> int:
        word_count = len((text or "").split())
        if word_count < 300:
            budget = 2
        elif word_count < 800:
            budget = 3
        elif word_count < 1500:
            budget = 4
        elif word_count < 2500:
            budget = 5
        else:
            budget = 6
        if available_chunks <= 0:
            return budget
        return max(1, min(budget, available_chunks))

    def _tokenize_sentence(self, sentence: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9']+", (sentence or "").lower())

    def _normalize_scores(self, values: List[float]) -> List[float]:
        if not values:
            return []
        arr = np.array(values, dtype=float)
        max_val = float(arr.max())
        min_val = float(arr.min())
        if abs(max_val - min_val) < 1e-9:
            return [0.5 for _ in values]
        scaled = (arr - min_val) / (max_val - min_val)
        return scaled.tolist()

    def _sentence_overlap_ratio(self, first: str, second: str) -> float:
        first_tokens = set(self._tokenize_sentence(first))
        second_tokens = set(self._tokenize_sentence(second))
        if not first_tokens or not second_tokens:
            return 0.0
        union = first_tokens.union(second_tokens)
        if not union:
            return 0.0
        return len(first_tokens.intersection(second_tokens)) / len(union)

    # -------------------------
    # TextRank extractive (for seeds)
    # -------------------------
    def _build_similarity_graph(self, sentences: List[str]) -> nx.Graph:
        G = nx.Graph()
        n = len(sentences)
        for i in range(n):
            G.add_node(i)

        # Try using embedder; otherwise fallback to token overlap
        embs = None
        if self.embedder:
            try:
                embs = self.embedder.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
            except Exception as e:
                logger.warning(f"Embedder encode failed: {e}")
                embs = None

        for i in range(n):
            for j in range(i + 1, n):
                if embs is not None:
                    w = float(np.dot(embs[i], embs[j]))
                    w = max(0.0, w)
                else:
                    si = set(sentences[i].lower().split())
                    sj = set(sentences[j].lower().split())
                    denom = (len(si) + len(sj)) if (len(si) + len(sj)) > 0 else 1
                    w = len(si.intersection(sj)) / denom
                if w > 0:
                    G.add_edge(i, j, weight=w)
        return G

    def _textrank_sentence_scores(self, sentences: List[str]) -> List[float]:
        if not sentences:
            return []
        G = self._build_similarity_graph(sentences)
        try:
            scores = nx.pagerank(G, weight="weight")
        except Exception:
            scores = {n: G.degree(n) for n in G.nodes()}
        ordered = [float(scores.get(i, 0.0)) for i in range(len(sentences))]
        return self._normalize_scores(ordered)

    def textrank_extract(self, text: str, top_k: int = 5) -> str:
        text_clean = self._preprocess(text)
        sents = sent_tokenize(text_clean)
        sents = [s.strip() for s in sents if len(s.strip()) > 10]
        if not sents:
            return ""
        G = self._build_similarity_graph(sents)
        try:
            scores = nx.pagerank(G, weight="weight")
        except Exception:
            scores = {n: G.degree(n) for n in G.nodes()}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_idxs = sorted([idx for idx, _ in ranked[:min(top_k, len(ranked))]])
        summary = " ".join([sents[i] for i in top_idxs])
        return summary

    def _sentence_feature_matrix(self, sentences: List[str], textrank_scores: List[float]) -> (np.ndarray, List[Dict[str, float]]):
        tokens_per_sentence = [self._tokenize_sentence(sentence) for sentence in sentences]
        all_tokens = [token for tokens in tokens_per_sentence for token in tokens if token not in STOPWORDS and len(token) > 2]
        keyword_counts = {}
        for token in all_tokens:
            keyword_counts[token] = keyword_counts.get(token, 0) + 1
        top_keywords = {token for token, _ in sorted(keyword_counts.items(), key=lambda item: item[1], reverse=True)[:8]}

        embeddings = None
        centroid = None
        if self.embedder and sentences:
            try:
                embeddings = self.embedder.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
                centroid = embeddings.mean(axis=0)
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm > 0:
                    centroid = centroid / centroid_norm
            except Exception as e:
                logger.warning(f"Sentence embedding features failed: {e}")
                embeddings = None
                centroid = None

        feature_rows = []
        feature_meta = []
        denom = max(len(sentences) - 1, 1)
        for idx, sentence in enumerate(sentences):
            tokens = tokens_per_sentence[idx]
            token_count = len(tokens)
            unique_ratio = (len(set(tokens)) / token_count) if token_count else 0.0
            keyword_ratio = (sum(1 for token in tokens if token in top_keywords) / token_count) if token_count else 0.0
            digit_ratio = (sum(1 for token in tokens if any(ch.isdigit() for ch in token)) / token_count) if token_count else 0.0
            avg_token_length = (sum(len(token) for token in tokens) / token_count) if token_count else 0.0
            prev_overlap = self._sentence_overlap_ratio(sentence, sentences[idx - 1]) if idx > 0 else 0.0
            next_overlap = self._sentence_overlap_ratio(sentence, sentences[idx + 1]) if idx < len(sentences) - 1 else 0.0
            leading_signal = 1.0 if idx < 2 else 0.0
            trailing_signal = 1.0 if idx >= max(len(sentences) - 2, 0) else 0.0

            centroid_similarity = 0.0
            local_context_similarity = max(prev_overlap, next_overlap)
            if embeddings is not None and centroid is not None:
                centroid_similarity = float(np.dot(embeddings[idx], centroid))
                neighbor_sims = []
                if idx > 0:
                    neighbor_sims.append(float(np.dot(embeddings[idx], embeddings[idx - 1])))
                if idx < len(sentences) - 1:
                    neighbor_sims.append(float(np.dot(embeddings[idx], embeddings[idx + 1])))
                if neighbor_sims:
                    local_context_similarity = max(neighbor_sims)

            feature_rows.append([
                idx / denom,
                (len(sentences) - idx - 1) / denom,
                min(token_count / 40.0, 1.0),
                min(avg_token_length / 10.0, 1.0),
                unique_ratio,
                keyword_ratio,
                digit_ratio,
                prev_overlap,
                next_overlap,
                textrank_scores[idx] if idx < len(textrank_scores) else 0.5,
                centroid_similarity,
                local_context_similarity,
                leading_signal,
                trailing_signal,
            ])
            feature_meta.append({
                "keyword_ratio": keyword_ratio,
                "textrank_prior": textrank_scores[idx] if idx < len(textrank_scores) else 0.5,
                "centroid_similarity": centroid_similarity,
            })

        return np.array(feature_rows, dtype=np.float32), feature_meta

    def bilstm_extract(self, text: str, top_k: int = 5) -> Dict[str, Any]:
        text_clean = self._preprocess(text)
        sentences = [s.strip() for s in sent_tokenize(text_clean) if len(s.strip()) > 10]
        if not sentences:
            return {"summary": "", "sources": [], "model": "bilstm-empty"}

        textrank_scores = self._textrank_sentence_scores(sentences)
        features, feature_meta = self._sentence_feature_matrix(sentences, textrank_scores)
        top_k = max(1, min(top_k, len(sentences)))

        teacher_targets = []
        for idx, meta in enumerate(feature_meta):
            lead_bonus = 1.0 - (idx / max(len(sentences) - 1, 1))
            teacher_score = (
                0.6 * meta["textrank_prior"] +
                0.25 * meta["keyword_ratio"] +
                0.15 * lead_bonus
            )
            teacher_targets.append(float(min(max(teacher_score, 0.0), 1.0)))
        teacher_targets = self._normalize_scores(teacher_targets)

        bilstm_scores = teacher_targets[:]
        model_name = "bilstm-extractive-self-distilled"
        use_teacher_blend = True
        inputs = None

        if TORCH_AVAILABLE:
            inputs = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            trained_model, checkpoint_meta = self._ensure_bilstm_model(features.shape[1])
            if trained_model is not None:
                try:
                    with torch.no_grad():
                        bilstm_scores = torch.sigmoid(trained_model(inputs)).squeeze(0).cpu().numpy().tolist()
                    model_name = checkpoint_meta.get("model_name", "bilstm-trained-checkpoint")
                    top_k = max(1, min(int(checkpoint_meta.get("recommended_top_k", top_k)), len(sentences)))
                    use_teacher_blend = False
                except Exception as e:
                    logger.warning(f"Loaded BiLSTM checkpoint failed during inference, falling back: {e}")
                    self.bilstm_model = None
                    self.bilstm_model_input_dim = None
                    self.bilstm_model_meta = {}

        if use_teacher_blend and TORCH_AVAILABLE and len(sentences) > 1:
            try:
                torch.manual_seed(13)
                model = BiLSTMSentenceScorer(features.shape[1], hidden_dim=self.bilstm_hidden_dim)
                optimizer = torch.optim.Adam(model.parameters(), lr=self.bilstm_learning_rate)
                if inputs is None:
                    inputs = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                targets = torch.tensor(teacher_targets, dtype=torch.float32).unsqueeze(0)

                for _ in range(self.bilstm_epochs):
                    model.train()
                    optimizer.zero_grad()
                    logits = model(inputs)
                    probabilities = torch.sigmoid(logits)
                    loss = F.mse_loss(probabilities, targets)
                    if probabilities.shape[1] > 1:
                        smoothness = torch.mean((probabilities[:, 1:] - probabilities[:, :-1]) ** 2)
                        loss = loss + 0.05 * smoothness
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    bilstm_scores = torch.sigmoid(model(inputs)).squeeze(0).cpu().numpy().tolist()
            except Exception as e:
                logger.warning(f"BiLSTM extractive scorer failed, falling back to teacher scores: {e}")
                model_name = "bilstm-fallback"
        elif use_teacher_blend and not TORCH_AVAILABLE:
            model_name = "bilstm-unavailable-fallback"

        final_scores = []
        for index in range(len(sentences)):
            bilstm_score = bilstm_scores[index] if index < len(bilstm_scores) else 0.5
            teacher_score = teacher_targets[index] if index < len(teacher_targets) else 0.5
            if use_teacher_blend:
                final_scores.append(
                    (1.0 - self.bilstm_teacher_weight) * bilstm_score +
                    self.bilstm_teacher_weight * teacher_score
                )
            else:
                final_scores.append(bilstm_score)

        ranked = sorted(enumerate(final_scores), key=lambda item: item[1], reverse=True)
        selected_indices = sorted(index for index, _ in ranked[:top_k])
        selected_sentences = [sentences[index] for index in selected_indices]

        sources = []
        for index in selected_indices:
            sources.append({
                "snippet": sentences[index],
                "points": [
                    f"BiLSTM score: {final_scores[index]:.3f}",
                    f"Teacher prior: {teacher_targets[index]:.3f}",
                ],
                "meta": {
                    "sentence_id": index,
                    "bilstm_score": round(float(bilstm_scores[index]), 4),
                    "teacher_score": round(float(teacher_targets[index]), 4),
                },
            })

        return {
            "summary": " ".join(selected_sentences),
            "sources": sources,
            "model": model_name,
        }

    def _ensure_bilstm_model(self, input_dim: int):
        if not TORCH_AVAILABLE:
            return None, {}
        if self.bilstm_model is not None and self.bilstm_model_input_dim == input_dim:
            return self.bilstm_model, self.bilstm_model_meta
        if not self.bilstm_model_path or not os.path.exists(self.bilstm_model_path):
            return None, {}
        try:
            checkpoint = torch.load(self.bilstm_model_path, map_location="cpu")
            checkpoint_input_dim = int(checkpoint.get("input_dim", input_dim))
            if checkpoint_input_dim != input_dim:
                logger.warning(
                    "BiLSTM checkpoint input size %s does not match current feature size %s.",
                    checkpoint_input_dim,
                    input_dim,
                )
                return None, {}
            hidden_dim = int(checkpoint.get("hidden_dim", self.bilstm_hidden_dim))
            num_layers = int(checkpoint.get("num_layers", 1))
            dropout = float(checkpoint.get("dropout", 0.1))
            model = BiLSTMSentenceScorer(
                input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            self.bilstm_model = model
            self.bilstm_model_input_dim = input_dim
            self.bilstm_model_meta = checkpoint
            return model, checkpoint
        except Exception as e:
            logger.warning(f"Could not load BiLSTM checkpoint from {self.bilstm_model_path}: {e}")
            return None, {}

    # -------------------------
    # Transformers pipeline lazy loader
    # -------------------------
    def _ensure_transformers_pipeline(self):
        if not TRANSFORMERS_AVAILABLE:
            return None
        if self.abstractive_pipeline is None:
            try:
                self.abstractive_pipeline = pipeline("summarization", model=self.abstractive_model_name)
                logger.info(f"Loaded transformers summarization pipeline: {self.abstractive_model_name}")
            except Exception as e:
                logger.warning(f"Could not load transformers pipeline: {e}")
                self.abstractive_pipeline = None
        return self.abstractive_pipeline

    def abstractive_transformers(self, text: str, max_length: int = 150, min_length: int = 30) -> str:
        pipe = self._ensure_transformers_pipeline()
        if not pipe:
            raise RuntimeError("Transformers pipeline not available")
        out = pipe(text, max_length=max_length, min_length=min_length, do_sample=False)
        if isinstance(out, list) and out:
            return out[0].get("summary_text", str(out[0]))
        return str(out)

    # -------------------------
    # OpenAI abstractive (lazy)
    # -------------------------
    def abstractive_openai(self, prompt: str, max_tokens: int = 250) -> str:
        if not self.openai:
            raise RuntimeError("OpenAI not configured")
        try:
            # Use ChatCompletion if available
            resp = self.openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            if "choices" in resp and resp["choices"]:
                return resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"OpenAI call failed: {e}")
            raise

    # -------------------------
    # Sentence helper + bullet extraction
    # -------------------------
    def _sentences_from_text(self, text: str) -> List[str]:
        sents = sent_tokenize(text)
        return [s.strip() for s in sents if len(s.strip()) > 10]

    def chunk_to_bullets(self, chunk_text: str, top_k: int = 4) -> List[str]:
        """
        Convert a chunk into top_k bullet sentences.
        Prefer embedder + graph + PageRank if embedder is available; otherwise use a frequency
        based heuristic fallback.
        """
        sents = self._sentences_from_text(chunk_text)
        if not sents:
            return []

        # Try embedder + graph + PageRank (best)
        if self.embedder:
            try:
                embs = self.embedder.encode(sents, convert_to_numpy=True, normalize_embeddings=True)
                G = nx.Graph()
                for i in range(len(sents)):
                    G.add_node(i)
                for i in range(len(sents)):
                    for j in range(i + 1, len(sents)):
                        w = float(np.dot(embs[i], embs[j]))
                        if w > 0:
                            G.add_edge(i, j, weight=w)
                try:
                    scores = nx.pagerank(G, weight="weight")
                except Exception:
                    scores = {n: G.degree(n) for n in G.nodes()}
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                top_idxs = sorted([idx for idx, _ in ranked[:min(top_k, len(ranked))]])
                bullets = [sents[i] for i in top_idxs]
                return bullets
            except Exception as e:
                logger.warning(f"Embedder-based bullet extraction failed: {e}")

        # Fallback: simple frequency-based scoring
        try:
            from collections import Counter
            words = []
            for s in sents:
                for w in s.lower().split():
                    w = w.strip(".,;:()[]\"'")  # simple cleanup
                    if len(w) > 2:
                        words.append(w)
            freq = Counter(words)
            scores = []
            for s in sents:
                sc = 0.0
                for w in s.lower().split():
                    w = w.strip(".,;:()[]\"'")
                    sc += freq.get(w, 0)
                # small length normalization to favor concise sentences
                sc = sc / (len(s.split()) ** 0.3)
                scores.append(sc)
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            top_indices = sorted([i for i, _ in ranked[:min(top_k, len(ranked))]])
            bullets = [sents[i] for i in top_indices]
            return bullets
        except Exception as e:
            logger.warning(f"Fallback bullet extraction failed: {e}")
            # safe fallback: return first sentences up to top_k
            return sents[:min(top_k, len(sents))]

    # ------------------------- #
    # Main summarize orchestration
    # ------------------------- #
    def summarize(self,
                  text: str,
                  method: str = "hybrid",
                  on_premise: bool = False,
                  vector_store: Any = None) -> Any:
        """
        method: 'extractive'|'abstractive'|'hybrid'|'bilstm'
        on_premise: if True, avoid calling external APIs (OpenAI)
        vector_store: FaissStore instance containing chunks for retrieval (required for hybrid)
        returns: string or dict with summary, seed, sources
        """
        method = method.lower()
        text_clean = self._preprocess(text)
        extractive = self.textrank_extract(text_clean, top_k=6)

        if method == "bilstm":
            bilstm_result = self.bilstm_extract(text_clean, top_k=6)
            bilstm_seed = bilstm_result.get("summary", "")
            return {
                "summary": self._extractive_patient_summary(text_clean, seed=bilstm_seed),
                "seed": bilstm_seed,
                "sources": bilstm_result.get("sources", []),
                "model": bilstm_result.get("model", "bilstm-extractive-self-distilled"),
            }

        if method == "extractive":
            return {
                "summary": self._extractive_patient_summary(text_clean, seed=extractive),
                "seed": extractive,
                "sources": [],
                "model": "plain-language-extractive",
            }

        # Abstractive-only flow
        if method == "abstractive":
            # prefer OpenAI if present and allowed
            if self.openai and not on_premise:
                try:
                    prompt = self._plain_language_prompt(text_clean)
                    out = self.abstractive_openai(prompt)
                    return {
                        "summary": self._bulletize_text(out, limit=5),
                        "seed": extractive,
                        "sources": [],
                        "model": "openai-plain-language",
                    }
                except Exception as e:
                    logger.warning(f"OpenAI abstractive failed: {e}")
            # fallback to transformers pipeline if available
            pipe = self._ensure_transformers_pipeline()
            if pipe:
                transformed = self.abstractive_transformers(extractive if extractive else text_clean)
                return {
                    "summary": self._abstractive_patient_summary(transformed, seed=transformed),
                    "seed": extractive,
                    "sources": [],
                    "model": self.abstractive_model_name,
                }
            logger.warning("No abstractive model available; using extractive fallback instead.")
            return {
                "summary": self._abstractive_patient_summary(text_clean, seed=extractive),
                "seed": extractive,
                "sources": [],
                "model": "plain-language-fallback",
            }

        # Hybrid flow
        seed = extractive
        if not seed:
            return {"summary": "", "seed": "", "sources": []}

        if vector_store is None:
            return {
                "summary": self._hybrid_patient_summary(seed, []),
                "seed": seed,
                "sources": [],
                "model": "hybrid-plain-language-fallback",
                "chunk_count": 0,
            }

        # retrieve chunks relevant to the seed
        available_chunks = len(getattr(vector_store, "texts", []) or [])
        chunk_budget = self._determine_chunk_budget(text_clean, available_chunks)
        retrieved = vector_store.search(seed, top_k=chunk_budget)

        # prepare sources with point-wise bullets
        sources_with_points = []
        for t, m in retrieved:
            points = self.chunk_to_bullets(t, top_k=4)
            snippet = (t[:400] + "...") if len(t) > 400 else t
            sources_with_points.append({"snippet": snippet, "points": points, "meta": m})

        # build context for LLM if used
        ctx = ""
        for i, (chunk_text, meta) in enumerate(retrieved, start=1):
            ctx += f"[{i}] {chunk_text}\n\n"

        prompt = ("You are a clinical summarizer. Use ONLY the provided context chunks. "
                  "Do NOT hallucinate. If information is missing, say that the report does not clearly mention it. "
                  "Rewrite the summary in plain language for a patient or family member. "
                  "Use short, clear sentences and avoid jargon. Write short bullet points covering the main problem, "
                  "important findings, treatment, and follow-up.\n\n"
                  f"Context:\n{ctx}\nSummarize:")

        # Try OpenAI if available and not on-premise
        if self.openai and not on_premise:
            try:
                out = self.abstractive_openai(prompt)
                return {
                    "summary": self._bulletize_text(out, limit=6),
                    "seed": seed,
                    "sources": sources_with_points,
                    "model": "openai-plain-language",
                    "chunk_count": len(sources_with_points),
                }
            except Exception as e:
                logger.warning(f"OpenAI hybrid failed: {e}")

        # Try transformers pipeline if available
        pipe = self._ensure_transformers_pipeline()
        if pipe:
            try:
                concat = " ".join([t for t, _ in retrieved])
                out = self.abstractive_transformers(concat)
                return {
                    "summary": self._hybrid_patient_summary(out, sources_with_points),
                    "seed": seed,
                    "sources": sources_with_points,
                    "model": self.abstractive_model_name,
                    "chunk_count": len(sources_with_points),
                }
            except Exception as e:
                logger.warning(f"Transformers hybrid failed: {e}")

        # Fallback: return a plain-language version of the extractive seed
        return {
            "summary": self._hybrid_patient_summary(seed, sources_with_points),
            "seed": seed,
            "sources": sources_with_points,
            "model": "plain-language-fallback",
            "chunk_count": len(sources_with_points),
        }
