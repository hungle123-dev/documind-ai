from collections.abc import Iterable

from langchain_core.documents import Document

from agents.query_expander import QueryExpander
from agents.relevance_checker import RelevanceChecker
from agents.research_agent import ResearchAgent, build_context
from agents.verification_agent import VerificationAgent


class FakeChatClient:
    def __init__(self, responses: list[str | list[str]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        response = self.responses.pop(0)
        assert isinstance(response, str)
        return response

    def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Iterable[str]:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        response = self.responses.pop(0)
        assert isinstance(response, list)
        yield from response


class FallbackAwareFakeClient:
    def __init__(self, response: str = "fallback response") -> None:
        self.response = response
        self.seen_models: list[list[str]] = []

    def complete(
        self,
        *,
        model: str | list[str],
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        self.seen_models.append(model if isinstance(model, list) else [model])
        return self.response

    def stream(
        self,
        *,
        model: str | list[str],
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Iterable[str]:
        self.seen_models.append(model if isinstance(model, list) else [model])
        yield self.response


class FakeRetriever:
    def __init__(self, docs: list[Document]) -> None:
        self.docs = docs

    def invoke(self, question: str) -> list[Document]:
        return self.docs


def test_relevance_checker_normalizes_valid_labels() -> None:
    client = FakeChatClient(["partial."])
    checker = RelevanceChecker(client=client, model="grok-fast")

    result = checker.check(
        "Can I answer?",
        FakeRetriever([Document(page_content="context")]),
    )

    assert result == "PARTIAL"


def test_query_expander_keeps_original_and_two_variants() -> None:
    client = FakeChatClient(["variant one\nvariant two\nvariant three"])
    expander = QueryExpander(client=client, model="grok-fast")

    assert expander.expand("original") == ["original", "variant one", "variant two"]


def test_build_context_respects_budget_and_labels_sources() -> None:
    docs = [
        Document(
            page_content="A" * 50,
            metadata={"source": "a.pdf", "page": 2, "section": "Intro"},
        ),
        Document(
            page_content="B" * 1000,
            metadata={"source": "b.pdf", "page": 3, "section": "Body"},
        ),
    ]

    context = build_context(docs, max_tokens=30)

    assert "Source 1" in context
    assert "a.pdf" in context
    assert "Source 2" not in context


def test_research_agent_streams_tokens_and_returns_sources() -> None:
    client = FakeChatClient([["Hello", " ", "world"]])
    docs = [
        Document(
            page_content="context",
            metadata={"source": "a.pdf", "page": 1, "section": "Intro"},
        )
    ]
    agent = ResearchAgent(client=client, model="grok-3")

    chunks = list(agent.generate_stream("question", docs))

    assert "".join(chunk for chunk, _ in chunks) == "Hello world"
    assert chunks[-1][1] == docs


def test_research_agent_passes_primary_and_fallback_models() -> None:
    client = FallbackAwareFakeClient()
    docs = [
        Document(
            page_content="context",
            metadata={"source": "a.pdf", "page": 1, "section": "Intro"},
        )
    ]
    agent = ResearchAgent(
        client=client,
        model="grok-3",
        fallback_models=["grok-3-mini"],
    )

    result = agent.generate("question", docs)

    assert result["draft_answer"] == "fallback response"
    assert client.seen_models == [["grok-3", "grok-3-mini"]]


def test_verification_parser_supplies_defaults_for_missing_fields() -> None:
    agent = VerificationAgent(client=FakeChatClient([]), model="grok-mini")

    parsed = agent.parse_verification_response("Supported: YES\nRelevant: YES")

    assert parsed["Supported"] == "YES"
    assert parsed["Relevant"] == "YES"
    assert parsed["Unsupported Claims"] == []
    assert parsed["Contradictions"] == []
