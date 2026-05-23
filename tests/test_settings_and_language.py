from config.settings import Settings
from utils.language import detect_language, vi_tokenizer


def test_settings_exposes_grok_and_retrieval_defaults() -> None:
    settings = Settings(XAI_API_KEY="xai-test-key")

    assert settings.XAI_BASE_URL == "https://api.x.ai/v1"
    assert settings.RELEVANCE_MODEL == "grok-3-mini-fast"
    assert settings.RESEARCH_MODEL == "grok-3"
    assert settings.RESEARCH_FALLBACK_MODELS == ["grok-3-mini"]
    assert settings.VERIFICATION_MODEL == "grok-3-mini"
    assert settings.EMBEDDING_MODEL == "intfloat/multilingual-e5-small"
    assert settings.RERANKER_MODEL == "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    assert settings.VECTOR_SEARCH_K == 15
    assert settings.RERANKER_TOP_N == 5
    assert settings.MAX_RESEARCH_ITERATIONS == 2


def test_settings_parses_multiple_xai_api_keys() -> None:
    settings = Settings(XAI_API_KEYS="xai-one, xai-two,,xai-three")

    assert settings.xai_api_keys == ["xai-one", "xai-two", "xai-three"]


def test_settings_falls_back_to_single_xai_api_key() -> None:
    settings = Settings(XAI_API_KEY="xai-single")

    assert settings.xai_api_keys == ["xai-single"]


def test_detect_language_returns_vi_for_vietnamese_text() -> None:
    assert detect_language("Xin chào, tôi muốn hỏi về tài liệu này.") == "vi"


def test_vi_tokenizer_returns_lowercase_tokens() -> None:
    tokens = vi_tokenizer("Trung tâm dữ liệu xanh")

    assert any("trung" in token for token in tokens)
    assert all(token == token.lower() for token in tokens)
