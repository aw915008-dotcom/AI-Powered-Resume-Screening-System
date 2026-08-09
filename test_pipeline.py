"""
test_pipeline.py
------------------
Module 10 — automated tests for the core, deterministic parts of the
pipeline (text cleaning, skill matching, hybrid matcher math). The
supervised classifier and SBERT loading are integration-tested manually
in the training/evaluation scripts instead, since they depend on trained
artifacts and network access respectively.

Run with:  python -m pytest tests/ -v   (from the project root)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from preprocessing.text_cleaner import clean_text, remove_urls, remove_emails
from feature_engineering.skill_matcher import SkillMatcher
from feature_engineering.embedding_featurizer import SemanticFeaturizer
from models.matcher import HybridMatcher


def test_remove_urls():
    assert remove_urls("check https://example.com now") == "check   now"


def test_remove_emails():
    assert remove_emails("contact me@test.com please") == "contact   please"


def test_clean_text_lowercases_and_strips_punctuation():
    result = clean_text("Hello, World! Visit http://x.com")
    assert "hello" in result
    assert "world" in result
    assert "," not in result
    assert "http" not in result


def test_clean_text_empty_input():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_skill_matcher_exact_match():
    sm = SkillMatcher(["python", "sql", "machine learning"])
    result = sm.compare("i know python and sql well", "we need python and machine learning")
    assert "python" in result["matching_skills"]
    assert "machine learning" in result["missing_skills"]
    assert "sql" in result["extra_skills"]


def test_skill_matcher_no_substring_false_positive():
    """'java' should not match inside 'javascript'."""
    sm = SkillMatcher(["java"])
    found = sm.extract_skills("i am a javascript developer")
    assert "java" not in found


def test_skill_matcher_handles_empty_text():
    sm = SkillMatcher(["python"])
    result = sm.compare("", "")
    assert result["matching_skills"] == []
    assert result["skill_overlap_percent"] == 0.0


def test_hybrid_matcher_weights_must_sum_to_one():
    with pytest.raises(AssertionError):
        HybridMatcher(
            semantic_featurizer=SemanticFeaturizer(force_backend="tfidf_fallback"),
            skill_matcher=SkillMatcher(["python"]),
            weight_semantic=0.5,
            weight_skill=0.6,
        )


def test_hybrid_matcher_identical_texts_score_high():
    sf = SemanticFeaturizer(force_backend="tfidf_fallback")
    sm = SkillMatcher(["python", "sql"])
    corpus = ["experienced python and sql developer"] * 2
    sf.fit_fallback(corpus)
    matcher = HybridMatcher(sf, sm)
    result = matcher.score(corpus[0], corpus[1])
    assert result.final_score > 50  # identical text should score highly


def test_hybrid_matcher_unrelated_texts_score_low():
    sf = SemanticFeaturizer(force_backend="tfidf_fallback")
    sm = SkillMatcher(["python", "sql", "welding", "forklift"])
    text_a = "python sql backend developer"
    text_b = "forklift welding warehouse safety"
    # Corpus needs >=2 docs sharing a term for TF-IDF's default min_df=2,
    # so duplicate each text once — a tiny stand-in for a real multi-resume corpus.
    corpus = [text_a, text_a, text_b, text_b]
    sf.fit_fallback(corpus)
    matcher = HybridMatcher(sf, sm)
    result = matcher.score(text_a, text_b)
    assert result.final_score < 30


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
