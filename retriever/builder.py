import hashlib
from pathlib import Path
from typing import Callable, Protocol

from langchain_core.documents import Document

from config.settings import settings
from utils.language import vi_tokenizer
from utils.logging import logger


class VectorFactory(Protocol):
    def load_or_build(self, docs: list[Document], persist_directory: str):
        """Return a vector store for the documents."""


Reranker = Callable[[str, list[Document], int], list[Document]]


def combined_file_hash(file_hashes: list[str]) -> str:
    joined = "|".join(sorted(file_hashes))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def dedupe_documents(documents: list[Document]) -> list[Document]:
    seen: set[str] = set()
    deduped: list[Document] = []
    for doc in documents:
        if doc.page_content in seen:
            continue
        seen.add(doc.page_content)
        deduped.append(doc)
    return deduped


class E5Embeddings:
    def __init__(self, model_name: str | None = None) -> None:
        from langchain_huggingface import HuggingFaceEmbeddings

        self.embeddings = HuggingFaceEmbeddings(
            model=model_name or settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            query_encode_kwargs={"normalize_embeddings": True},
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings.embed_documents([f"passage: {text}" for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.embeddings.embed_query(f"query: {text}")


class ChromaVectorFactory:
    def __init__(self, embeddings=None) -> None:
        self.embeddings = embeddings or E5Embeddings()

    def load_or_build(self, docs: list[Document], persist_directory: str):
        from langchain_community.vectorstores import Chroma

        persist_path = Path(persist_directory)
        if persist_path.exists() and any(persist_path.iterdir()):
            logger.info(f"Loading cached Chroma DB: {persist_path.name}")
            return Chroma(
                persist_directory=str(persist_path),
                embedding_function=self.embeddings,
            )

        logger.info(f"Building Chroma DB: {persist_path.name}")
        persist_path.mkdir(parents=True, exist_ok=True)
        return Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=str(persist_path),
        )


class BM25Index:
    def __init__(self, docs: list[Document]) -> None:
        from rank_bm25 import BM25Okapi

        self.docs = docs
        self.tokens = [vi_tokenizer(doc.page_content) for doc in docs]
        self.index = BM25Okapi(self.tokens) if self.tokens else None

    def invoke(self, query: str, k: int) -> list[Document]:
        if self.index is None:
            return []
        scores = self.index.get_scores(vi_tokenizer(query))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [self.docs[index] for index, score in ranked[:k] if score > 0]


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name or settings.RERANKER_MODEL)

    def __call__(self, query: str, items: list[Document], top_n: int) -> list[Document]:
        if not items:
            return []
        pairs = [(query, item.page_content) for item in items]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(items, scores), key=lambda item: float(item[1]), reverse=True)
        return [doc for doc, _ in ranked[:top_n]]


class HybridRetriever:
    def __init__(
        self,
        docs: list[Document],
        vector_store,
        reranker: Reranker,
        search_k: int,
        top_n: int,
    ) -> None:
        self.docs = docs
        self.vector_retriever = vector_store.as_retriever(search_kwargs={"k": search_k})
        self.bm25 = BM25Index(docs)
        self.reranker = reranker
        self.search_k = search_k
        self.top_n = top_n

    def invoke(self, query: str) -> list[Document]:
        candidates: list[Document] = []
        candidates.extend(self.bm25.invoke(query, self.search_k))
        candidates.extend(self.vector_retriever.invoke(query))
        deduped = dedupe_documents(candidates)
        return self.reranker(query, deduped, self.top_n)


class RetrieverBuilder:
    def __init__(
        self,
        embeddings=None,
        vector_factory: VectorFactory | None = None,
        reranker: Reranker | None = None,
        chroma_root: Path | str | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.vector_factory = vector_factory
        self.reranker = reranker
        self.chroma_root = Path(chroma_root or settings.CHROMA_DB_PATH)

    def build_hybrid_retriever(self, docs: list[Document]) -> HybridRetriever:
        if not docs:
            raise ValueError("Cannot build retriever without documents.")

        file_hashes = [str(doc.metadata.get("file_hash", "")) for doc in docs]
        persist_directory = self.chroma_root / combined_file_hash(file_hashes)
        vector_factory = self.vector_factory or ChromaVectorFactory(embeddings=self.embeddings)
        reranker = self.reranker or CrossEncoderReranker()
        vector_store = vector_factory.load_or_build(docs, str(persist_directory))

        return HybridRetriever(
            docs=docs,
            vector_store=vector_store,
            reranker=reranker,
            search_k=settings.VECTOR_SEARCH_K,
            top_n=settings.RERANKER_TOP_N,
        )
