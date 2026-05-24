from collections.abc import Iterable

from langchain_core.documents import Document

from agents.query_expander import QUERY_EXPANSION_PROMPT, QueryExpander
from agents.relevance_checker import RelevanceChecker
from agents.research_agent import RESEARCH_PROMPT, ResearchAgent, build_context
from agents.verification_agent import VERIFICATION_PROMPT, VerificationAgent


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


class FailingChatClient:
    def complete(
        self,
        *,
        model: str | list[str],
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        raise RuntimeError("quota exceeded")

    def stream(
        self,
        *,
        model: str | list[str],
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Iterable[str]:
        raise RuntimeError("quota exceeded")


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
    checker = RelevanceChecker(client=client, model="llama-3.1-8b-instant")

    result = checker.check(
        "Can I answer?",
        FakeRetriever([Document(page_content="context")]),
    )

    assert result == "PARTIAL"


def test_query_expander_keeps_original_and_two_variants() -> None:
    client = FakeChatClient(["variant one\nvariant two\nvariant three"])
    expander = QueryExpander(client=client, model="llama-3.1-8b-instant")

    assert expander.expand("original") == ["original", "variant one", "variant two"]


def test_query_expander_defaults_to_stronger_model_for_cross_lingual_retrieval() -> None:
    expander = QueryExpander(client=FakeChatClient([]))

    assert expander.model == "llama-3.3-70b-versatile"


def test_query_expander_passes_primary_and_fallback_models() -> None:
    client = FallbackAwareFakeClient(response="expanded one\nexpanded two")
    expander = QueryExpander(
        client=client,
        model="llama-3.3-70b-versatile",
        fallback_models=["llama-3.1-8b-instant"],
    )

    assert expander.expand("original") == ["original", "expanded one", "expanded two"]
    assert client.seen_models == [["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]]


def test_query_expander_falls_back_to_original_question_when_expansion_fails() -> None:
    expander = QueryExpander(client=FailingChatClient())

    assert expander.expand("original") == ["original"]


def test_query_expander_strips_language_labels() -> None:
    client = FakeChatClient(["English translation: translated query\nEnglish keyword query: keyword query"])
    expander = QueryExpander(client=client, model="llama-3.1-8b-instant")

    assert expander.expand("original") == ["original", "translated query", "keyword query"]


def test_query_expander_prompt_supports_cross_lingual_retrieval() -> None:
    assert "English search-query translation" in QUERY_EXPANSION_PROMPT
    assert "English keyword query" in QUERY_EXPANSION_PROMPT
    assert "standard academic terminology" in QUERY_EXPANSION_PROMPT
    assert "retrieval queries" in QUERY_EXPANSION_PROMPT
    assert "respond entirely" not in QUERY_EXPANSION_PROMPT


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


def test_research_prompt_requires_exact_table_qualifiers() -> None:
    assert "Match row labels, column labels, and qualifiers exactly" in RESEARCH_PROMPT
    assert "Do not mix values from different rows" in RESEARCH_PROMPT
    assert "answer only for that entity" in RESEARCH_PROMPT
    assert "Never present values for another entity as alternatives" in RESEARCH_PROMPT


def test_research_prompt_preserves_named_methods_and_acronyms() -> None:
    assert "Preserve exact names" in RESEARCH_PROMPT
    assert "methods, formulations, systems, datasets, metrics, and acronyms" in RESEARCH_PROMPT
    assert "Do not replace official names with only paraphrases" in RESEARCH_PROMPT


def test_research_agent_uses_deterministic_temperature_for_grounded_answers() -> None:
    client = FakeChatClient(["answer"])
    agent = ResearchAgent(client=client, model="llama-3.3-70b-versatile")

    agent.generate(
        "question",
        [Document(page_content="context", metadata={"source": "a.pdf", "page": 1})],
    )

    assert client.calls[0]["temperature"] == 0.0


def test_research_agent_explicitly_passes_user_question_language() -> None:
    client = FakeChatClient(["câu trả lời"])
    agent = ResearchAgent(client=client, model="llama-3.3-70b-versatile")

    agent.generate(
        "Mô hình RAG dùng bộ nhớ nào?",
        [Document(page_content="context", metadata={"source": "a.pdf", "page": 1})],
    )

    user_message = client.calls[0]["messages"][1]["content"]
    assert "User question language: vi" in user_message
    assert "Answer language: Vietnamese" in user_message


def test_research_agent_streams_tokens_and_returns_sources() -> None:
    client = FakeChatClient([["Hello", " ", "world"]])
    docs = [
        Document(
            page_content="context",
            metadata={"source": "a.pdf", "page": 1, "section": "Intro"},
        )
    ]
    agent = ResearchAgent(client=client, model="llama-3.3-70b-versatile")

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
        model="llama-3.3-70b-versatile",
        fallback_models=["llama-3.1-8b-instant"],
    )

    result = agent.generate("question", docs)

    assert result["draft_answer"] == "fallback response"
    assert client.seen_models == [["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]]


def test_verification_parser_supplies_defaults_for_missing_fields() -> None:
    agent = VerificationAgent(client=FakeChatClient([]), model="llama-3.3-70b-versatile")

    parsed = agent.parse_verification_response("Supported: YES\nRelevant: YES")

    assert parsed["Supported"] == "YES"
    assert parsed["Relevant"] == "YES"
    assert parsed["Unsupported Claims"] == []
    assert parsed["Contradictions"] == []


def test_verification_parser_corrects_self_contradictory_supported_label() -> None:
    agent = VerificationAgent(client=FakeChatClient([]), model="llama-3.1-8b-instant")

    parsed = agent.parse_verification_response(
        "Supported: NO\n"
        "Unsupported Claims: The answer says X.\n"
        "Contradictions: None\n"
        "Relevant: YES\n"
        "Additional Details: The context states that the answer says X."
    )

    assert parsed["Supported"] == "YES"
    assert parsed["Unsupported Claims"] == []


def test_verification_prompt_treats_grounded_insufficiency_as_supported() -> None:
    assert "correctly states that the context is insufficient" in VERIFICATION_PROMPT


def test_verification_prompt_requires_consistent_table_fact_checks() -> None:
    assert "linearized table rows" in VERIFICATION_PROMPT
    assert "base vs big" in VERIFICATION_PROMPT
    assert "Supported: YES" in VERIFICATION_PROMPT
