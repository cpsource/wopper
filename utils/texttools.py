# texttools.py
# Text normalization and tokenization helper functions.

import os
from dotenv import load_dotenv
import re
import string
from typing import List
from utils.logger import get_logger

log = get_logger(__name__)
log.debug("Starting texttools.py")

def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation, and extra spaces."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> List[str]:
    """Split text into tokens using whitespace."""
    return text.split()

def extract_concepts(text: str, stopwords: List[str] = None) -> List[str]:
    """Extract meaningful tokens by removing stopwords."""
    if stopwords is None:
        stopwords = []
    tokens = tokenize(normalize_text(text))
    return [token for token in tokens if token not in stopwords]

def unique_words(texts: List[str]) -> List[str]:
    """Return a sorted list of unique words from a list of texts."""
    words = set()
    for text in texts:
        words.update(extract_concepts(text))
    return sorted(words)

# -----------------------------
# 🔍 Local Test
# -----------------------------
if __name__ == "__main__":
    sample = "The quick, brown fox jumps over the lazy dog!"
    log.info("Normalized: %s", normalize_text(sample))
    log.info("Tokens: %s", tokenize(sample))
    log.info("Concepts: %s", extract_concepts(sample, stopwords=['the', 'over']))
    log.info("Unique Words: %s", unique_words([sample, "The fox ran fast."]))

