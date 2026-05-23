from langchain_core.documents import Document

from agents.workflow import AgentWorkflow


class FakeRetriever:
    def __init__(self) -> None:
        self.docs = [
            Document(
                page_content="ctx",
                metadata={"source": "a.pdf", "page": 1, "section": "Intro"},
            )
        ]

    def invoke(self, question: str) -> list[Document]:
        return self.docs


class FakeRelevance:
    def __init__(self, label: str) -> None:
        self.label = label

    def check(self, question: str, retriever: FakeRetriever, k: int = 20) -> str:
        return self.label


class FakeResearch:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, question: str, documents: list[Document]) -> dict[str, object]:
        self.calls += 1
        return {"draft_answer": f"answer {self.calls}", "source_docs": documents}


class FakeVerifier:
    def __init__(self, reports: list[str]) -> None:
        self.reports = list(reports)

    def check(self, answer: str, documents: list[Document]) -> dict[str, str]:
        return {"verification_report": self.reports.pop(0)}


def test_workflow_ends_without_research_on_no_match() -> None:
    research = FakeResearch()
    workflow = AgentWorkflow(
        relevance_checker=FakeRelevance("NO_MATCH"),
        researcher=research,
        verifier=FakeVerifier([]),
        max_iterations=2,
    )

    result = workflow.full_pipeline("unrelated", FakeRetriever())

    assert research.calls == 0
    assert result["relevance"] == "NO_MATCH"
    assert result["source_docs"] == []


def test_workflow_stops_after_max_research_iterations() -> None:
    research = FakeResearch()
    workflow = AgentWorkflow(
        relevance_checker=FakeRelevance("CAN_ANSWER"),
        researcher=research,
        verifier=FakeVerifier(
            [
                "Supported: NO\nRelevant: YES",
                "Supported: NO\nRelevant: YES",
            ]
        ),
        max_iterations=2,
    )

    result = workflow.full_pipeline("question", FakeRetriever())

    assert research.calls == 2
    assert result["iteration_count"] == 2
    assert result["draft_answer"] == "answer 2"
    assert result["source_docs"]
