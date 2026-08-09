"""
text_cleaner.py
----------------
Reusable, dependency-light text cleaning functions for resumes and job
descriptions. Each function does exactly one job so the pipeline stays
testable and easy to reorder.

Design choice: we use NLTK for stopwords + lemmatization because it is
lightweight, well-tested, and does not require downloading a large
language model (unlike spaCy's `en_core_web_sm`), which keeps the
Streamlit deployment fast to cold-start.
"""

import re
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK corpora once. In a production deployment this
# should be moved to a Dockerfile / setup step rather than run at import
# time, but for this project's scale it is safe and idempotent.
for _pkg in ("stopwords", "wordnet", "omw-1.4", "punkt", "punkt_tab"):
    try:
        nltk.data.find(f"corpora/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_PHONE_RE = re.compile(
    r"(\+?\d{1,3}[\s.-]?)?(\(?\d{2,4}\)?[\s.-]?){2,4}\d{2,4}"
)
_MULTI_SPACE_RE = re.compile(r"\s+")
_SPECIAL_CHARS_RE = re.compile(r"[^a-z0-9\s]")


def remove_urls(text: str) -> str:
    """Strip http(s)/www links."""
    return _URL_RE.sub(" ", text)


def remove_emails(text: str) -> str:
    """Strip email addresses."""
    return _EMAIL_RE.sub(" ", text)


def remove_phone_numbers(text: str) -> str:
    """Strip phone-number-like digit sequences.

    Note: intentionally applied AFTER lowercasing/URL/email removal in the
    default pipeline order so it doesn't accidentally eat parts of a URL.
    """
    return _PHONE_RE.sub(" ", text)


def normalize_unicode(text: str) -> str:
    """Fold accented characters (e.g. 'Café' -> 'Cafe')."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")


def to_lowercase(text: str) -> str:
    return text.lower()


def remove_special_characters(text: str) -> str:
    """Keep only letters, digits and whitespace."""
    return _SPECIAL_CHARS_RE.sub(" ", text)


def remove_extra_whitespace(text: str) -> str:
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> list:
    return word_tokenize(text)


def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def lemmatize(tokens: list) -> list:
    return [_LEMMATIZER.lemmatize(t) for t in tokens]


def clean_text(text: str, keep_tokens: bool = False):
    """Full cleaning pipeline used across resumes and job descriptions.

    Order matters:
      1. Unicode normalize   -> stable character set before regex
      2. Remove URLs/emails  -> before special-char stripping, since '@'
                                 and '/' would otherwise mangle them
      3. Lowercase
      4. Remove phone numbers
      5. Remove special characters / punctuation
      6. Collapse whitespace
      7. Tokenize -> remove stopwords -> lemmatize

    Parameters
    ----------
    text : str
        Raw input text.
    keep_tokens : bool
        If True, return the list of cleaned tokens instead of a
        rejoined string (useful for BoW/TF-IDF debugging).
    """
    if not isinstance(text, str) or not text.strip():
        return [] if keep_tokens else ""

    text = normalize_unicode(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = to_lowercase(text)
    text = remove_phone_numbers(text)
    text = remove_special_characters(text)
    text = remove_extra_whitespace(text)

    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)

    return tokens if keep_tokens else " ".join(tokens)
