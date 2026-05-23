import re
from typing import Protocol

from config.settings import settings
from utils.language import LANGUAGE_INSTRUCTION
from utils.logging import logger

from .llm_client import ChatClient, GroqClient


class RetrieverLike(Protocol):
    def invoke(self, question: str):
        """Return documents relevant to question."""


RELEVANCE_PROMPT = (
    "You are a document relevance classifier.\n"
    "Given retrieved document chunks and a question, classify whether the documents contain "
    "sufficient information to answer the question.\n\n"
    "Respond with EXACTLY one of: CAN_ANSWER | PARTIAL | NO_MATCH\n"
    "No explanation. No punctuation. Just the label.\n"
    f"{LANGUAGE_INSTRUCTION}"
)


class RelevanceChecker:
    VALID_LABELS = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}

    def __init__(self, client: ChatClient | None = None, model: str | None = None) -> None:
        self.client = client or GroqClient()
        self.model = model or settings.RELEVANCE_MODEL

    def check(self, question: str, retriever: RetrieverLike, k: int = 3) -> str:
        logger.debug(f"RelevanceChecker.check | question='{question}' | k={k}")
        docs = retriever.invoke(question)
        if not docs:
            return "NO_MATCH"

        passages = "\n\n".join(doc.page_content for doc in docs[:k])
        response = self.client.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": RELEVANCE_PROMPT},
                {"role": "user", "content": f"Question: {question}\n\nPassages:\n{passages}"},
            ],
            temperature=0.0,
            max_tokens=20,
        )
        label = self._normalize_label(response)
        logger.debug(f"Relevance classification: {label}")
        return label

    def _normalize_label(self, response: str) -> str:
        cleaned = re.sub(r"[^A-Z_]", "", response.strip().upper())
        return cleaned if cleaned in self.VALID_LABELS else "NO_MATCH"
