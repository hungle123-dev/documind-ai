from pydantic import Field
from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .constants import ALLOWED_TYPES, MAX_FILE_SIZE, MAX_TOTAL_SIZE

class Settings(BaseSettings):
    # LLM (Groq, OpenAI-compatible)
    GROQ_API_KEY: str = Field(default="")
    GROQ_API_KEYS: str = Field(default="")
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    RELEVANCE_MODEL: str = "llama-3.1-8b-instant"
    RESEARCH_MODEL: str = "llama-3.3-70b-versatile"
    RESEARCH_FALLBACK_MODELS: list[str] = ["llama-3.1-8b-instant"]
    VERIFICATION_MODEL: str = "llama-3.3-70b-versatile"

    # Document limits
    MAX_FILE_SIZE: int = MAX_FILE_SIZE
    MAX_TOTAL_SIZE: int = MAX_TOTAL_SIZE
    ALLOWED_TYPES: list[str] = ALLOWED_TYPES

    # Embeddings and retrieval
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-small"
    RERANKER_MODEL: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    CHROMA_DB_PATH: str = "chroma_db"
    VECTOR_SEARCH_K: int = 15
    RERANKER_TOP_N: int = 5
    HYBRID_RETRIEVER_WEIGHTS: list[float] = [0.4, 0.6]
    MAX_CONTEXT_TOKENS: int = 6000
    MAX_RESEARCH_ITERATIONS: int = 2

    # Caching
    CACHE_DIR: str = "document_cache"
    CACHE_EXPIRE_DAYS: int = 7

    # Server and logging
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 7860
    GRADIO_SHARE: bool = False
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @computed_field
    @property
    def groq_api_keys(self) -> list[str]:
        raw_keys = self.GROQ_API_KEYS or self.GROQ_API_KEY
        return [key.strip() for key in raw_keys.split(",") if key.strip()]

settings = Settings()
