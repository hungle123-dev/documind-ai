from config.settings import Settings
from utils.language import detect_language, vi_tokenizer


def test_settings_exposes_groq_and_retrieval_defaults() -> None:
    settings = Settings(_env_file=None, GROQ_API_KEY="gsk-test-key")

    assert settings.GROQ_BASE_URL == "https://api.groq.com/openai/v1"
    assert settings.RELEVANCE_MODEL == "llama-3.1-8b-instant"
    assert settings.RESEARCH_MODEL == "llama-3.3-70b-versatile"
    assert settings.RESEARCH_FALLBACK_MODELS == ["llama-3.1-8b-instant"]
    assert settings.VERIFICATION_MODEL == "llama-3.1-8b-instant"
    assert settings.EMBEDDING_MODEL == "intfloat/multilingual-e5-small"
    assert settings.RERANKER_MODEL == "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    assert settings.VECTOR_SEARCH_K == 15
    assert settings.RERANKER_TOP_N == 5
    assert settings.MAX_RESEARCH_ITERATIONS == 2


def test_settings_parses_multiple_groq_api_keys() -> None:
    settings = Settings(_env_file=None, GROQ_API_KEYS="gsk-one, gsk-two,,gsk-three")

    assert settings.groq_api_keys == ["gsk-one", "gsk-two", "gsk-three"]


def test_settings_falls_back_to_single_groq_api_key() -> None:
    settings = Settings(_env_file=None, GROQ_API_KEY="gsk-single")

    assert settings.groq_api_keys == ["gsk-single"]


def test_detect_language_returns_vi_for_vietnamese_text() -> None:
    assert detect_language("Xin chào, tôi muốn hỏi về tài liệu này.") == "vi"


def test_vi_tokenizer_returns_lowercase_tokens() -> None:
    tokens = vi_tokenizer("Trung tâm dữ liệu xanh")

    assert any("trung" in token for token in tokens)
    assert all(token == token.lower() for token in tokens)
