"""
embedding_featurizer.py
------------------------
Sentence-BERT (SBERT) semantic embeddings — the PRIMARY similarity
signal in this project.

Why SBERT over plain TF-IDF for the main match score:
- Resumes and job descriptions describe the same skill in different
  words ("ML" vs "machine learning", "led a team" vs "management
  experience"). TF-IDF only sees exact token overlap; SBERT embeddings
  put semantically similar phrases close in vector space even with
  zero shared vocabulary.
- No labeled (resume, job, match) pairs exist in this dataset (see
  Module 1/3 discussion), which rules out training a supervised
  classifier from scratch. A pretrained sentence encoder gives strong
  similarity out of the box with zero training data required — exactly
  what an unsupervised matching problem like this needs.
- 'all-MiniLM-L6-v2' is the standard choice for this kind of task: 384
  dimensions, ~80MB, fast enough for CPU inference in a Streamlit app,
  and it's the most widely validated general-purpose SBERT model for
  semantic similarity.

Fallback behaviour: this environment's network policy blocks
huggingface.co, so the model weights cannot be downloaded here — verified
directly (a request to huggingface.co returns HTTP 403 from the egress
proxy). In a normal deployment (local machine, Streamlit Cloud, a server
with unrestricted internet), the download succeeds and this class uses
real SBERT embeddings. To keep the pipeline runnable in ANY environment,
`SemanticFeaturizer` automatically falls back to the TF-IDF featurizer if
the SBERT model can't be loaded, and exposes `.backend` so callers /
the UI can display which one is actually active.
"""

import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from feature_engineering.tfidf_featurizer import TfidfFeaturizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticFeaturizer:
    """Unified interface: `.encode(texts)` and `.similarity(a, b)` work the
    same way regardless of which backend ended up active."""

    def __init__(self, force_backend: str = None):
        """
        force_backend: None (default) auto-detects — tries SBERT, falls
        back to TF-IDF on failure. Pass 'tfidf_fallback' explicitly to
        skip the SBERT load attempt entirely (useful for lightweight
        deployments, CI, or environments where the model download is
        known to be unavailable, e.g. restricted network policies).
        """
        self.backend = None
        self.model = None
        self._tfidf_fallback = None
        if force_backend == "tfidf_fallback":
            self.backend = "tfidf_fallback"
            logger.info("SemanticFeaturizer forced to TF-IDF backend (SBERT load skipped).")
        else:
            self._load_sbert()

    def _load_sbert(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(MODEL_NAME)
            self.backend = "sbert"
            logger.info(f"Loaded SBERT model '{MODEL_NAME}'")
        except Exception as e:
            logger.warning(
                f"Could not load SBERT model ({e.__class__.__name__}: {e}). "
                "Falling back to TF-IDF for semantic similarity."
            )
            self.backend = "tfidf_fallback"

    def fit_fallback(self, corpus: list):
        """Only needed when backend == 'tfidf_fallback' — TF-IDF must be
        fit on the corpus before it can transform anything."""
        if self.backend == "tfidf_fallback":
            self._tfidf_fallback = TfidfFeaturizer().fit(corpus)

    def encode(self, texts: list):
        if self.backend == "sbert":
            return self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        if self._tfidf_fallback is None:
            raise RuntimeError("TF-IDF fallback not fit yet — call fit_fallback(corpus) first.")
        return self._tfidf_fallback.transform(texts)

    def similarity(self, embeddings_a, embeddings_b) -> np.ndarray:
        if self.backend == "sbert":
            return cosine_similarity(embeddings_a, embeddings_b)
        return TfidfFeaturizer.similarity(embeddings_a, embeddings_b)
