import hashlib
import os
from pathlib import Path
from typing import Callable, Protocol

from langchain_core.documents import Document

from config.settings import settings
from utils.language import detect_language, vi_tokenizer
from utils.logging import logger
from utils.network import configure_system_trust_store


class VectorFactory(Protocol):
    def load_or_build(self, docs: list[Document], persist_directory: str):
        """Return a vector store for the documents."""


Reranker = Callable[[str, list[Document], int], list[Document]]
QueryExpansionFunction = Callable[[str], list[str]]


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


def configure_torch_cpu_runtime() -> None:
    thread_count = settings.RERANKER_TORCH_THREADS
    thread_value = str(thread_count)
    os.environ.setdefault("OMP_NUM_THREADS", thread_value)
    os.environ.setdefault("MKL_NUM_THREADS", thread_value)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch

    torch.set_num_threads(thread_count)
    try:
        torch.set_num_interop_threads(thread_count)
    except RuntimeError as exc:
        logger.debug(f"Torch interop threads already initialized: {exc}")


class E5Embeddings:
    def __init__(self, model_name: str | None = None) -> None:
        configure_system_trust_store()
        configure_torch_cpu_runtime()
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
        from langchain_chroma import Chroma

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
        configure_system_trust_store()
        configure_torch_cpu_runtime()
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
        query_expander: QueryExpansionFunction | None = None,
    ) -> None:
        self.docs = docs
        self.vector_retriever = vector_store.as_retriever(search_kwargs={"k": search_k})
        self.bm25 = BM25Index(docs)
        self.reranker = reranker
        self.search_k = search_k
        self.top_n = top_n
        self.query_expander = query_expander

    def invoke(self, query: str) -> list[Document]:
        queries = self.query_expander(query) if self.query_expander else [query]
        ranked_per_query: list[list[Document]] = []
        for candidate_query in dict.fromkeys(queries):
            candidates: list[Document] = []
            candidates.extend(self.bm25.invoke(candidate_query, self.search_k))
            candidates.extend(self.vector_retriever.invoke(candidate_query))
            deduped = dedupe_documents(candidates)
            ranked_per_query.append(self.reranker(candidate_query, deduped, self.top_n))
        if detect_language(query) == "en":
            return self._merge_primary_first(ranked_per_query)
        return self._merge_round_robin(ranked_per_query)

    def _merge_primary_first(self, ranked_per_query: list[list[Document]]) -> list[Document]:
        merged: list[Document] = []
        seen: set[str] = set()
        for ranked_items in ranked_per_query:
            for doc in ranked_items:
                if doc.page_content in seen:
                    continue
                merged.append(doc)
                seen.add(doc.page_content)
                if len(merged) == self.top_n:
                    return merged
        return merged

    def _merge_round_robin(self, ranked_per_query: list[list[Document]]) -> list[Document]:
        merged: list[Document] = []
        seen: set[str] = set()
        max_length = max((len(items) for items in ranked_per_query), default=0)
        for rank_index in range(max_length):
            for ranked_items in ranked_per_query:
                if rank_index >= len(ranked_items):
                    continue
                doc = ranked_items[rank_index]
                if doc.page_content in seen:
                    continue
                merged.append(doc)
                seen.add(doc.page_content)
                if len(merged) == self.top_n:
                    return merged
        return merged


class RetrieverBuilder:
    def __init__(
        self,
        embeddings=None,
        vector_factory: VectorFactory | None = None,
        reranker: Reranker | None = None,
        query_expander: QueryExpansionFunction | None = None,
        chroma_root: Path | str | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.vector_factory = vector_factory
        self.reranker = reranker
        self.query_expander = query_expander
        self.chroma_root = Path(chroma_root or settings.CHROMA_DB_PATH)
        self._default_vector_factory: VectorFactory | None = None
        self._default_reranker: Reranker | None = None

    def build_hybrid_retriever(self, docs: list[Document]) -> HybridRetriever:
        if not docs:
            raise ValueError("Cannot build retriever without documents.")

        file_hashes = [
            f"{doc.metadata.get('file_hash', '')}:schema:{doc.metadata.get('processor_schema_version', '')}"
            for doc in docs
        ]
        persist_directory = self.chroma_root / combined_file_hash(file_hashes)
        vector_factory = self.vector_factory or self._get_default_vector_factory()
        reranker = self.reranker or self._get_default_reranker()
        vector_store = vector_factory.load_or_build(docs, str(persist_directory))

        return HybridRetriever(
            docs=docs,
            vector_store=vector_store,
            reranker=reranker,
            search_k=settings.VECTOR_SEARCH_K,
            top_n=settings.RERANKER_TOP_N,
            query_expander=self.query_expander,
        )

    def _get_default_vector_factory(self) -> VectorFactory:
        if self._default_vector_factory is None:
            self._default_vector_factory = ChromaVectorFactory(embeddings=self.embeddings)
        return self._default_vector_factory

    def _get_default_reranker(self) -> Reranker:
        if self._default_reranker is None:
            self._default_reranker = CrossEncoderReranker()
        return self._default_reranker
