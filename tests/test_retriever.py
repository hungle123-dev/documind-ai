from langchain_core.documents import Document

from retriever.builder import RetrieverBuilder, combined_file_hash, dedupe_documents


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


def test_default_builder_does_not_load_models_until_build(monkeypatch, tmp_path) -> None:
    import retriever.builder as builder_module

    def fail_if_loaded(*args, **kwargs):
        raise AssertionError("Heavy model loaded during RetrieverBuilder construction")

    monkeypatch.setattr(builder_module, "ChromaVectorFactory", fail_if_loaded)
    monkeypatch.setattr(builder_module, "CrossEncoderReranker", fail_if_loaded)

    builder = RetrieverBuilder(chroma_root=tmp_path)

    assert builder.chroma_root == tmp_path
