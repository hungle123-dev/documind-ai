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


class RelevanceLike(Protocol):
    def check(self, question: str, retriever: RetrieverLike, k: int = 20) -> str:
        """Classify retrieved evidence."""


class ResearchLike(Protocol):
    def generate(self, question: str, documents: list[Document]) -> dict[str, Any]:
        """Generate an answer from source documents."""


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

        state["relevance"] = self.relevance_checker.check(question, retriever, k=20)
        if state["relevance"] == "NO_MATCH":
            state["draft_answer"] = (
                "The uploaded documents do not contain enough relevant information "
                "to answer this question."
            )
            return dict(state)

        documents = retriever.invoke(question)
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

    def _research_step(self, state: WorkflowState) -> WorkflowState:
        result = self.researcher.generate(state["question"], state["source_docs"])
        state["iteration_count"] += 1
        state["draft_answer"] = str(result["draft_answer"])
        source_docs = result.get("source_docs")
        if isinstance(source_docs, list):
            state["source_docs"] = source_docs
        return state

    def _verification_step(self, state: WorkflowState) -> WorkflowState:
        result = self.verifier.check(state["draft_answer"], state["source_docs"])
        state["verification_report"] = result.get("verification_report", "")
        return state

    def _needs_research_retry(self, state: WorkflowState) -> bool:
        report = state["verification_report"].lower()
        return "supported: no" in report or "relevant: no" in report
