import sys
from types import SimpleNamespace

from langchain_core.documents import Document

from retriever.builder import (
    ChromaVectorFactory,
    RetrieverBuilder,
    combined_file_hash,
    configure_system_trust_store,
    dedupe_documents,
)


def test_combined_file_hash_is_order_independent() -> None:
    assert combined_file_hash(["b", "a"]) == combined_file_hash(["a", "b"])


def test_dedupe_documents_preserves_first_seen_metadata() -> None:
    docs = [
        Document(page_content="same", metadata={"source": "a"}),
        Document(page_content="same", metadata={"source": "b"}),
    ]

    result = dedupe_documents(docs)

    assert len(result) == 1
    assert result[0].metadata["source"] == "a"


class FakeVectorStore:
    def __init__(self, docs: list[Document]) -> None:
        self.docs = docs

    def as_retriever(self, search_kwargs: dict) -> "FakeVectorStore":
        return self

    def invoke(self, query: str) -> list[Document]:
        return self.docs


class FakeFactory:
    def __init__(self) -> None:
        self.built: list[str] = []

    def load_or_build(self, docs: list[Document], persist_directory: str) -> FakeVectorStore:
        self.built.append(persist_directory)
        return FakeVectorStore(docs)


def test_builder_returns_top_reranked_documents(tmp_path) -> None:
    docs = [
        Document(page_content="alpha", metadata={"file_hash": "1"}),
        Document(page_content="beta", metadata={"file_hash": "1"}),
    ]
    builder = RetrieverBuilder(
        vector_factory=FakeFactory(),
        reranker=lambda query, items, top_n: items[:1],
        chroma_root=tmp_path,
    )

    retriever = builder.build_hybrid_retriever(docs)

    assert len(retriever.invoke("alpha")) == 1


def test_builder_cache_key_changes_with_processor_schema(tmp_path) -> None:
    factory = FakeFactory()
    builder = RetrieverBuilder(
        vector_factory=factory,
        reranker=lambda query, items, top_n: items[:top_n],
        chroma_root=tmp_path,
    )

    builder.build_hybrid_retriever(
        [Document(page_content="alpha", metadata={"file_hash": "1", "processor_schema_version": 2})]
    )
    builder.build_hybrid_retriever(
        [Document(page_content="alpha", metadata={"file_hash": "1", "processor_schema_version": 3})]
    )

    assert factory.built[0] != factory.built[1]


def test_hybrid_retriever_expands_queries_before_reranking(tmp_path) -> None:
    docs = [Document(page_content="alpha", metadata={"file_hash": "1"})]
    vector_store = FakeVectorStore(docs)
    invoked_queries: list[str] = []

    def expand(query: str) -> list[str]:
        return [query, "expanded query"]

    def rerank(query: str, items: list[Document], top_n: int) -> list[Document]:
        invoked_queries.append(query)
        return items[:top_n]

    class TrackingFactory:
        def load_or_build(self, _docs: list[Document], _persist_directory: str):
            original_invoke = vector_store.invoke

            def track(query: str) -> list[Document]:
                invoked_queries.append(query)
                return original_invoke(query)

            vector_store.invoke = track
            return vector_store

    retriever = RetrieverBuilder(
        vector_factory=TrackingFactory(),
        reranker=rerank,
        query_expander=expand,
        chroma_root=tmp_path,
    ).build_hybrid_retriever(docs)

    retriever.invoke("original query")

    assert "original query" in invoked_queries
    assert "expanded query" in invoked_queries
    assert invoked_queries[-1] == "expanded query"


def test_hybrid_retriever_preserves_evidence_from_expanded_query(tmp_path) -> None:
    original_doc = Document(page_content="original-only", metadata={"file_hash": "1"})
    expanded_doc = Document(page_content="expanded-only", metadata={"file_hash": "1"})
    vector_store = FakeVectorStore([original_doc, expanded_doc])

    def expand(query: str) -> list[str]:
        return [query, "expanded query"]

    def rerank(query: str, items: list[Document], top_n: int) -> list[Document]:
        if query == "expanded query":
            return [doc for doc in items if doc.page_content == "expanded-only"]
        return [doc for doc in items if doc.page_content == "original-only"]

    class QueryAwareFactory:
        def load_or_build(self, _docs: list[Document], _persist_directory: str):
            def invoke(query: str) -> list[Document]:
                return [expanded_doc] if query == "expanded query" else [original_doc]

            vector_store.invoke = invoke
            return vector_store

    retriever = RetrieverBuilder(
        vector_factory=QueryAwareFactory(),
        reranker=rerank,
        query_expander=expand,
        chroma_root=tmp_path,
    ).build_hybrid_retriever([original_doc, expanded_doc])

    results = retriever.invoke("original query")

    assert [doc.page_content for doc in results] == ["original-only", "expanded-only"]


def test_hybrid_retriever_keeps_primary_ranking_for_english_queries(tmp_path) -> None:
    primary_docs = [
        Document(page_content=f"primary-{index}", metadata={"file_hash": "1"})
        for index in range(5)
    ]
    expanded_doc = Document(page_content="expanded-only", metadata={"file_hash": "1"})
    vector_store = FakeVectorStore([*primary_docs, expanded_doc])

    def expand(query: str) -> list[str]:
        return [query, "expanded query"]

    def rerank(query: str, items: list[Document], top_n: int) -> list[Document]:
        if query == "expanded query":
            return [expanded_doc]
        return primary_docs[:top_n]

    class QueryAwareFactory:
        def load_or_build(self, _docs: list[Document], _persist_directory: str):
            def invoke(query: str) -> list[Document]:
                return [expanded_doc] if query == "expanded query" else primary_docs

            vector_store.invoke = invoke
            return vector_store

    retriever = RetrieverBuilder(
        vector_factory=QueryAwareFactory(),
        reranker=rerank,
        query_expander=expand,
        chroma_root=tmp_path,
    ).build_hybrid_retriever([*primary_docs, expanded_doc])

    results = retriever.invoke("What is the answer?")

    assert [doc.page_content for doc in results] == [f"primary-{index}" for index in range(5)]


def test_hybrid_retriever_interleaves_expanded_queries_for_vietnamese_queries(tmp_path) -> None:
    primary_docs = [
        Document(page_content=f"primary-{index}", metadata={"file_hash": "1"})
        for index in range(5)
    ]
    expanded_doc = Document(page_content="expanded-only", metadata={"file_hash": "1"})
    vector_store = FakeVectorStore([*primary_docs, expanded_doc])

    def expand(query: str) -> list[str]:
        return [query, "expanded query"]

    def rerank(query: str, items: list[Document], top_n: int) -> list[Document]:
        if query == "expanded query":
            return [expanded_doc]
        return primary_docs[:top_n]

    class QueryAwareFactory:
        def load_or_build(self, _docs: list[Document], _persist_directory: str):
            def invoke(query: str) -> list[Document]:
                return [expanded_doc] if query == "expanded query" else primary_docs

            vector_store.invoke = invoke
            return vector_store

    retriever = RetrieverBuilder(
        vector_factory=QueryAwareFactory(),
        reranker=rerank,
        query_expander=expand,
        chroma_root=tmp_path,
    ).build_hybrid_retriever([*primary_docs, expanded_doc])

    results = retriever.invoke("Bộ nhớ này là gì?")

    assert "expanded-only" in [doc.page_content for doc in results]


def test_vector_factory_uses_supported_chroma_package(monkeypatch, tmp_path) -> None:
    calls: list[str] = []

    class FakeChroma:
        @classmethod
        def from_documents(cls, **kwargs):
            calls.append(kwargs["persist_directory"])
            return "vector-store"

    monkeypatch.setitem(sys.modules, "langchain_chroma", SimpleNamespace(Chroma=FakeChroma))
    factory = ChromaVectorFactory(embeddings=object())

    result = factory.load_or_build(
        [Document(page_content="evidence", metadata={"file_hash": "1"})],
        str(tmp_path / "db"),
    )

    assert result == "vector-store"
    assert calls == [str(tmp_path / "db")]


def test_default_builder_does_not_load_models_until_build(monkeypatch, tmp_path) -> None:
    import retriever.builder as builder_module

    def fail_if_loaded(*args, **kwargs):
        raise AssertionError("Heavy model loaded during RetrieverBuilder construction")

    monkeypatch.setattr(builder_module, "ChromaVectorFactory", fail_if_loaded)
    monkeypatch.setattr(builder_module, "CrossEncoderReranker", fail_if_loaded)

    builder = RetrieverBuilder(chroma_root=tmp_path)

    assert builder.chroma_root == tmp_path


def test_configure_system_trust_store_injects_platform_certificates(monkeypatch) -> None:
    calls: list[bool] = []
    fake_truststore = SimpleNamespace(inject_into_ssl=lambda: calls.append(True))
    monkeypatch.setitem(sys.modules, "truststore", fake_truststore)

    configure_system_trust_store()

    assert calls == [True]
