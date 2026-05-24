from collections.abc import Iterable

from langchain_core.documents import Document

from config.settings import settings
from utils.language import LANGUAGE_INSTRUCTION, detect_language
from utils.logging import logger

from .llm_client import ChatClient, GroqClient


RESEARCH_PROMPT = (
    "You are a precise document Q&A assistant.\n"
    "Answer the question using ONLY information from the provided context.\n"
    "Do not use outside knowledge.\n"
    "If the context is insufficient, say so explicitly.\n"
    "For tables and numeric facts: Match row labels, column labels, and qualifiers exactly.\n"
    "Do not mix values from different rows, columns, models, datasets, languages, or time periods.\n"
    "If the question asks about one entity, answer only for that entity; mention other entities only "
    "when explicitly needed to disambiguate.\n"
    "Never present values for another entity as alternatives to the requested entity.\n"
    "Preserve exact names for methods, formulations, systems, datasets, metrics, and acronyms "
    "from the context.\n"
    "Do not replace official names with only paraphrases; include the exact name and then explain it "
    "if useful.\n"
    "Cite which source supports each claim.\n"
    f"{LANGUAGE_INSTRUCTION}"
)


def build_context(documents: list[Document], max_tokens: int | None = None) -> str:
    budget = (max_tokens or settings.MAX_CONTEXT_TOKENS) * 4
    context_parts: list[str] = []
    total_chars = 0

    for index, doc in enumerate(documents, start=1):
        metadata = doc.metadata
        page = metadata.get("page", "?")
        source = metadata.get("source", "unknown")
        section = metadata.get("section", "N/A")
        chunk = (
            f"[Source {index}: {source}, page {page}, section {section}]\n"
            f"{doc.page_content}"
        )
        if total_chars + len(chunk) > budget:
            logger.warning(f"Token budget reached before source {index}; truncating context")
            break
        context_parts.append(chunk)
        total_chars += len(chunk)

    return "\n\n---\n\n".join(context_parts)


def language_directive(question: str) -> str:
    language = detect_language(question)
    answer_language = "Vietnamese" if language == "vi" else "English"
    return (
        f"User question language: {language}\n"
        f"Answer language: {answer_language}\n"
        "Use the answer language above even when the retrieved context is written in another language."
    )


class ResearchAgent:
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

    def generate(self, question: str, documents: list[Document]) -> dict[str, object]:
        logger.debug(f"ResearchAgent.generate | docs={len(documents)}")
        context = build_context(documents)
        answer = self.client.complete(
            model=self.models,
            messages=[
                {"role": "system", "content": RESEARCH_PROMPT},
                {
                    "role": "user",
                    "content": f"{language_directive(question)}\n\nQuestion: {question}\n\nContext:\n{context}",
                },
            ],
            temperature=0.0,
            max_tokens=1024,
        ).strip()
        if not answer:
            raise RuntimeError("ResearchAgent returned an empty answer.")
        return {"draft_answer": answer, "context_used": context, "source_docs": documents}

    def generate_stream(
        self,
        question: str,
        documents: list[Document],
    ) -> Iterable[tuple[str, list[Document]]]:
        logger.debug(f"ResearchAgent.generate_stream | docs={len(documents)}")
        context = build_context(documents)
        for token in self.client.stream(
            model=self.models,
            messages=[
                {"role": "system", "content": RESEARCH_PROMPT},
                {
                    "role": "user",
                    "content": f"{language_directive(question)}\n\nQuestion: {question}\n\nContext:\n{context}",
                },
            ],
            temperature=0.0,
            max_tokens=1024,
        ):
            yield token, documents
