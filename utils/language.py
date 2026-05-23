from loguru import logger


def detect_language(text: str) -> str:
    """Return 'vi' or 'en' based on text content."""
    try:
        from langdetect import detect
        lang = detect(text[:500])  # sample first 500 chars for speed
        return "vi" if lang == "vi" else "en"
    except Exception:
        return "en"


def vi_tokenizer(text: str) -> list[str]:
    """Vietnamese-aware tokenizer for BM25. Falls back to whitespace split."""
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text").lower().split()
    except ImportError:
        logger.warning("underthesea not installed — falling back to whitespace tokenizer")
        return text.lower().split()
    except Exception as e:
        logger.warning(f"underthesea tokenization failed: {e} — falling back to whitespace")
        return text.lower().split()
