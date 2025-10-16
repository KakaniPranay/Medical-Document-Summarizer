# summarizer.py
import os
import logging
from transformers import pipeline
from typing import List, Dict, Any
import nltk
from nltk.tokenize import sent_tokenize
import networkx as nx
import numpy as np

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

    def chunk_text(self, text: str, max_words: int = 600, overlap_words: int = 100) -> List[str]:
        text = self._preprocess(text)
        return chunk_text_by_sentences(text, max_words=max_words, overlap_words=overlap_words)

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
        method: 'extractive'|'abstractive'|'hybrid'
        on_premise: if True, avoid calling external APIs (OpenAI)
        vector_store: FaissStore instance containing chunks for retrieval (required for hybrid)
        returns: string or dict with summary, seed, sources
        """
        method = method.lower()
        text_clean = self._preprocess(text)
        extractive = self.textrank_extract(text_clean, top_k=6)

        if method == "extractive":
            return extractive

        # Abstractive-only flow
        if method == "abstractive":
            # prefer OpenAI if present and allowed
            if self.openai and not on_premise:
                try:
                    prompt = f"Summarize concisely the following medical text. Do NOT hallucinate.\n\n{text_clean}"
                    return self.abstractive_openai(prompt)
                except Exception as e:
                    logger.warning(f"OpenAI abstractive failed: {e}")
            # fallback to transformers pipeline if available
            pipe = self._ensure_transformers_pipeline()
            if pipe:
                return self.abstractive_transformers(extractive if extractive else text_clean)
            raise RuntimeError("No abstractive model available (OpenAI or transformers).")

        # Hybrid flow
        seed = extractive
        if not seed:
            return {"summary": "", "seed": "", "sources": []}

        if vector_store is None:
            return {"summary": seed, "seed": seed, "sources": []}

        # retrieve chunks relevant to the seed
        retrieved = vector_store.search(seed, top_k=6)

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
                  "Do NOT hallucinate. If information is missing, say 'insufficient information'. "
                  "Produce up to 8 concise bullet points. For each bullet, list supporting chunk ids in parentheses.\n\n"
                  f"Context:\n{ctx}\nSummarize:")

        # Try OpenAI if available and not on-premise
        if self.openai and not on_premise:
            try:
                out = self.abstractive_openai(prompt)
                return {"summary": out, "seed": seed, "sources": sources_with_points, "model": "openai"}
            except Exception as e:
                logger.warning(f"OpenAI hybrid failed: {e}")

        # Try transformers pipeline if available
        pipe = self._ensure_transformers_pipeline()
        if pipe:
            try:
                concat = " ".join([t for t, _ in retrieved])
                out = self.abstractive_transformers(concat)
                return {"summary": out, "seed": seed, "sources": sources_with_points, "model": self.abstractive_model_name}
            except Exception as e:
                logger.warning(f"Transformers hybrid failed: {e}")

        # Fallback: return the extractive seed + sources (point-wise)
        return {"summary": seed, "seed": seed, "sources": sources_with_points, "model": "extractive-fallback"}
