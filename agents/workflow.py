from collections.abc import Iterable
from typing import Any, Protocol, TypedDict

from langchain_core.documents import Document

from config.settings import settings
from utils.logging import logger

from .relevance_checker import RelevanceChecker
from .research_agent import ResearchAgent
from .verification_agent import VerificationAgent


class RetrieverLike(Protocol):
    def invoke(self, question: str) -> list[Document]:
        """Return documents relevant to question."""


class RetrievedEvidence:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    def invoke(self, question: str) -> list[Document]:
        return self.documents


class RelevanceLike(Protocol):
    def check(self, question: str, retriever: RetrieverLike, k: int = 20) -> str:
        """Classify retrieved evidence."""


class ResearchLike(Protocol):
    def generate(self, question: str, documents: list[Document]) -> dict[str, Any]:
        """Generate an answer from source documents."""

    def generate_stream(
        self,
        question: str,
        documents: list[Document],
    ) -> Iterable[tuple[str, list[Document]]]:
        """Yield answer text while retaining its source documents."""


class VerificationLike(Protocol):
    def check(self, answer: str, documents: list[Document]) -> dict[str, str]:
        """Verify a generated answer."""


class WorkflowState(TypedDict):
    question: str
    retriever: RetrieverLike
    draft_answer: str
    source_docs: list[Document]
    verification_report: str
    relevance: str
    iteration_count: int


class AgentWorkflow:
    def __init__(
        self,
        relevance_checker: RelevanceLike | None = None,
        researcher: ResearchLike | None = None,
        verifier: VerificationLike | None = None,
        max_iterations: int | None = None,
    ) -> None:
        self.relevance_checker = relevance_checker or RelevanceChecker()
        self.researcher = researcher or ResearchAgent()
        self.verifier = verifier or VerificationAgent()
        self.max_iterations = max_iterations or settings.MAX_RESEARCH_ITERATIONS

    def full_pipeline(self, question: str, retriever: RetrieverLike) -> dict[str, Any]:
        logger.debug(f"AgentWorkflow.full_pipeline | question='{question}'")
        state: WorkflowState = {
            "question": question,
            "retriever": retriever,
            "draft_answer": "",
            "source_docs": [],
            "verification_report": "",
            "relevance": "NO_MATCH",
            "iteration_count": 0,
        }

        documents = retriever.invoke(question)
        state["relevance"] = self.relevance_checker.check(
            question,
            RetrievedEvidence(documents),
            k=20,
        )
        if state["relevance"] == "NO_MATCH":
            state["draft_answer"] = (
                "The uploaded documents do not contain enough relevant information "
                "to answer this question."
            )
            return dict(state)

        state["source_docs"] = documents
        if not documents:
            state["relevance"] = "NO_MATCH"
            state["draft_answer"] = (
                "The retriever did not return source passages for this question."
            )
            return dict(state)

        while state["iteration_count"] < self.max_iterations:
            state = self._research_step(state)
            state = self._verification_step(state)
            if not self._needs_research_retry(state):
                break
            logger.info("Verification failed; retrying research with the same sources.")

        if state["iteration_count"] >= self.max_iterations and self._needs_research_retry(state):
            logger.warning("Max research iterations reached; returning latest verified draft.")
        return dict(state)

    def stream_pipeline(self, question: str, retriever: RetrieverLike) -> Iterable[dict[str, Any]]:
        logger.debug(f"AgentWorkflow.stream_pipeline | question='{question}'")
        state: WorkflowState = {
            "question": question,
            "retriever": retriever,
            "draft_answer": "",
            "source_docs": [],
            "verification_report": "",
            "relevance": "NO_MATCH",
            "iteration_count": 0,
        }

        documents = retriever.invoke(question)
        state["relevance"] = self.relevance_checker.check(
            question,
            RetrievedEvidence(documents),
            k=20,
        )
        if state["relevance"] == "NO_MATCH":
            state["draft_answer"] = (
                "The uploaded documents do not contain enough relevant information "
                "to answer this question."
            )
            yield dict(state)
            return

        state["source_docs"] = documents
        if not state["source_docs"]:
            state["relevance"] = "NO_MATCH"
            state["draft_answer"] = "The retriever did not return source passages for this question."
            yield dict(state)
            return

        while state["iteration_count"] < self.max_iterations:
            state["draft_answer"] = ""
            state["verification_report"] = ""
            for token, source_docs in self.researcher.generate_stream(question, state["source_docs"]):
                state["draft_answer"] += token
                state["source_docs"] = source_docs
                yield dict(state)
            if not state["draft_answer"].strip():
                raise RuntimeError("ResearchAgent returned an empty streamed answer.")
            state["iteration_count"] += 1
            state = self._verification_step(state)
            yield dict(state)
            if not self._needs_research_retry(state):
                return
            logger.info("Verification failed; retrying streamed research with the same sources.")

        logger.warning("Max research iterations reached; returning latest verified draft.")

    def _research_step(self, state: WorkflowState) -> WorkflowState:
        result = self.researcher.generate(state["question"], state["source_docs"])
        state["iteration_count"] += 1
        state["draft_answer"] = str(result["draft_answer"])
        source_docs = result.get("source_docs")
        if isinstance(source_docs, list):
            state["source_docs"] = source_docs
        return state

    def _verification_step(self, state: WorkflowState) -> WorkflowState:
        try:
            result = self.verifier.check(state["draft_answer"], state["source_docs"])
            state["verification_report"] = result.get("verification_report", "")
        except RuntimeError as exc:
            logger.warning(f"Verification failed; returning unverified answer: {exc}")
            state["verification_report"] = (
                "Supported: UNKNOWN\n"
                "Unsupported Claims: Verification failed before completion\n"
                "Contradictions: Unknown\n"
                "Relevant: UNKNOWN\n"
                f"Additional Details: Verification failed: {exc}"
            )
        return state

    def _needs_research_retry(self, state: WorkflowState) -> bool:
        report = state["verification_report"].lower()
        if "supported: unknown" in report or "relevant: unknown" in report:
            return False
        return "supported: no" in report or "relevant: no" in report
