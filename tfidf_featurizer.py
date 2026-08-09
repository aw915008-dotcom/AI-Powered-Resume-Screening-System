"""
tfidf_featurizer.py
--------------------
TF-IDF representation for resumes and job descriptions.

Why TF-IDF is part of the design (not just a fallback):
- Fully offline — no model download, no GPU, cold-starts instantly in
  Streamlit.
- Interpretable — the weights map directly onto real vocabulary, which
  is exactly what we need for the "matching skills / missing skills /
  keyword overlap" outputs required in Module 7. Sentence-BERT gives a
  similarity *score* but not a natural list of "why".
- Works well here specifically because resumes and job posts are
  keyword/skill-dense text (tool names, certifications), where exact or
  near-exact token overlap is a strong, low-variance signal.

It is fit once on the UNION of resumes + job descriptions so both sides
share one vocabulary/IDF space — required for cosine similarity between
the two to be meaningful.
"""

import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfFeaturizer:
    def __init__(self, max_features: int = 20000, ngram_range=(1, 2)):
        # ngram_range=(1,2) so two-word skill phrases like "machine learning"
        # or "project management" are captured as single features, not just
        # their unigram halves.
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,          # drop terms that appear in only 1 document (likely noise)
            sublinear_tf=True,  # log-scale term frequency, standard for text with repeated keywords
        )
        self._fitted = False

    def fit(self, texts: list) -> "TfidfFeaturizer":
        self.vectorizer.fit(texts)
        self._fitted = True
        return self

    def transform(self, texts: list):
        if not self._fitted:
            raise RuntimeError("Call fit() (or fit on the combined corpus) before transform().")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: list):
        self._fitted = True
        return self.vectorizer.fit_transform(texts)

    @staticmethod
    def similarity(matrix_a, matrix_b):
        """Pairwise cosine similarity between two TF-IDF sparse matrices."""
        return cosine_similarity(matrix_a, matrix_b)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, path)

    @classmethod
    def load(cls, path: str) -> "TfidfFeaturizer":
        obj = cls()
        obj.vectorizer = joblib.load(path)
        obj._fitted = True
        return obj
