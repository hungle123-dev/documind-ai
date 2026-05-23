from config.settings import settings
from utils.language import LANGUAGE_INSTRUCTION

from .llm_client import ChatClient, GroqClient


QUERY_EXPANSION_PROMPT = (
    "Given a user question, generate 2 alternative phrasings that capture the same intent.\n"
    "These will be used for document retrieval. Be concise.\n"
    "Output format: one query per line, no numbering.\n"
    f"{LANGUAGE_INSTRUCTION}"
)


class QueryExpander:
    def __init__(self, client: ChatClient | None = None, model: str | None = None) -> None:
        self.client = client or GroqClient()
        self.model = model or settings.RELEVANCE_MODEL

    def expand(self, question: str) -> list[str]:
        response = self.client.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": QUERY_EXPANSION_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.3,
            max_tokens=150,
        )
        variants = [line.strip(" -\t0123456789.") for line in response.splitlines()]
        unique: list[str] = [question]
        for variant in variants:
            if variant and variant not in unique:
                unique.append(variant)
            if len(unique) == 3:
                break
        return unique
