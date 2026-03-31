"""Metrics for generation-based eval: language detection and CAPS rate."""

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Make langdetect deterministic
DetectorFactory.seed = 0


def caps_rate(text: str) -> float:
    """Fraction of alphabetic characters that are uppercase."""
    alpha = sum(c.isalpha() for c in text)
    if alpha == 0:
        return 0.0
    upper = sum(c.isupper() for c in text)
    return upper / alpha


def detect_language(text: str) -> str | None:
    """Detect language of text, lowercasing first for robustness to ALL CAPS.

    Returns ISO 639-1 code (e.g. 'en', 'es', 'fr') or None if detection fails.
    """
    if not text or not any(c.isalpha() for c in text):
        return None
    try:
        return detect(text.lower())
    except LangDetectException:
        return None
