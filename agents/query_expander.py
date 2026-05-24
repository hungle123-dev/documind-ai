from config.settings import settings
from utils.logging import logger

from .llm_client import ChatClient, GroqClient


QUERY_EXPANSION_PROMPT = (
    "Given a user question, generate 2 alternative phrasings that capture the same intent.\n"
    "These will be used as retrieval queries, not as the final user-facing answer.\n"
    "For Vietnamese or other non-English questions, output exactly two English lines:\n"
    "1. An English search-query translation using standard academic terminology.\n"
    "2. An English keyword query preserving technical terms, acronyms, and entities.\n"
    "Output no Vietnamese unless the original term is a named entity.\n"
    "Do not answer the question.\n"
    "For English questions, output exactly two concise English search-query paraphrases.\n"
    "Output format: one query per line, no numbering.\n"
)


class QueryExpander:
    def __init__(
        self,
        client: ChatClient | None = None,
        model: str | None = None,
        fallback_models: list[str] | None = None,
    ) -> None:
        self.client = client or GroqClient()
        self.model = model or settings.RESEARCH_MODEL
        self.fallback_models = fallback_models if fallback_models is not None else settings.RESEARCH_FALLBACK_MODELS

    @property
    def models(self) -> str | list[str]:
        return [self.model, *self.fallback_models] if self.fallback_models else self.model

    def expand(self, question: str) -> list[str]:
        try:
            response = self.client.complete(
                model=self.models,
                messages=[
                    {"role": "system", "content": QUERY_EXPANSION_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=0.0,
                max_tokens=150,
            )
        except RuntimeError as exc:
            logger.warning(f"Query expansion failed; using original question only: {exc}")
            return [question]
        variants = [self._clean_variant(line) for line in response.splitlines()]
        unique: list[str] = [question]
        for variant in variants:
            if variant and variant not in unique:
                unique.append(variant)
            if len(unique) == 3:
                break
        return unique

    def _clean_variant(self, line: str) -> str:
        variant = line.strip(" -\t0123456789.")
        lower = variant.lower()
        label_prefixes = (
            "english translation:",
            "english keyword query:",
            "translation:",
            "keyword query:",
        )
        for prefix in label_prefixes:
            if lower.startswith(prefix):
                return variant[len(prefix) :].strip()
        return variant
