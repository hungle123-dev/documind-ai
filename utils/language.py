from loguru import logger


LANGUAGE_INSTRUCTION = """
Detect the language of the user's question and respond in the SAME language.
If the question is in Vietnamese, respond entirely in Vietnamese.
If the question is in English, respond entirely in English.
"""


VIETNAMESE_DIACRITICS = set(
    "ăâđêôơư"
    "áàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệ"
    "íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
)


def detect_language(text: str) -> str:
    """Return 'vi' or 'en' based on text content."""
    sample = text[:500].lower()
    if any(char in VIETNAMESE_DIACRITICS for char in sample):
        return "vi"

    try:
        from langdetect import detect

        lang = detect(sample)
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
