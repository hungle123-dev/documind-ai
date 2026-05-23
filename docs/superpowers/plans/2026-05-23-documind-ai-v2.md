# DocuMind AI v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current baseline into a production-ready bilingual RAG document assistant with Groq agents, lightweight document ingestion, multilingual retrieval, streaming UI, tests, CI, Docker, and evaluation/deployment preparation.

**Architecture:** Implement the app as small contracts around settings, LLM clients, document chunks, retriever stages, workflow state, and UI helpers. External services and heavyweight ML components are injected or wrapped so unit tests stay offline while runtime code can use real Groq, Chroma, embeddings, and rerankers.

**Tech Stack:** Python 3.11+, Gradio, LangGraph, LangChain, OpenAI-compatible Groq API, pymupdf4llm, python-docx, ChromaDB, sentence-transformers, rank-bm25, underthesea, pytest, Docker, GitHub Actions.

---

## File Structure

- Create `tests/` as the real pytest suite; leave old exploratory `test/` assets only as samples until they are replaced or moved.
- Create `agents/llm_client.py` for the Groq client boundary and retry behavior.
- Create `agents/query_expander.py` for retrieval query variants.
- Rewrite `agents/relevance_checker.py`, `agents/research_agent.py`, `agents/verification_agent.py`, and `agents/workflow.py` around injectable clients and typed results.
- Rewrite `config/settings.py` to remove legacy OpenAI/WatsonX assumptions and expose all PRD settings.
- Rewrite `document_processor/file_handler.py` for pymupdf4llm/python-docx/text parsing, chunking, metadata, and cache behavior.
- Rewrite `retriever/builder.py` around staged multilingual retrieval with persistence and test seams.
- Refactor `app.py` to isolate UI helpers from Gradio wiring.
- Add `.github/workflows/test.yml`, `Dockerfile`, `docker-compose.yml`, `README.md`, and `eval/` assets.

---

### Task 1: Test Harness and Settings Contract

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_settings_and_language.py`
- Modify: `config/settings.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Write failing tests for settings and language helpers**

```python
# tests/test_settings_and_language.py
from config.settings import Settings
from utils.language import detect_language, vi_tokenizer


def test_settings_exposes_groq_and_retrieval_defaults():
    settings = Settings(GROQ_API_KEY="gsk-test-key")

    assert settings.GROQ_BASE_URL == "https://api.groq.com/openai/v1"
    assert settings.RELEVANCE_MODEL == "llama-3.1-8b-instant"
    assert settings.RESEARCH_MODEL == "llama-3.3-70b-versatile"
    assert settings.VERIFICATION_MODEL == "llama-3.1-8b-instant"
    assert settings.EMBEDDING_MODEL == "intfloat/multilingual-e5-small"
    assert settings.RERANKER_MODEL == "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    assert settings.VECTOR_SEARCH_K == 15
    assert settings.RERANKER_TOP_N == 5
    assert settings.MAX_RESEARCH_ITERATIONS == 2


def test_detect_language_returns_vi_for_vietnamese_text():
    assert detect_language("Xin chao, toi muon hoi ve tai lieu nay.") == "vi"


def test_vi_tokenizer_returns_lowercase_tokens():
    tokens = vi_tokenizer("Trung tam du lieu xanh")
    assert "trung" in tokens
    assert all(token == token.lower() for token in tokens)
```

- [ ] **Step 2: Run red tests**

Run: `python -m pytest tests/test_settings_and_language.py -q`

Expected: FAIL because `Settings` still requires `OPENAI_API_KEY` and does not expose the v2 fields.

- [ ] **Step 3: Implement settings defaults**

```python
# config/settings.py
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .constants import ALLOWED_TYPES, MAX_FILE_SIZE, MAX_TOTAL_SIZE


class Settings(BaseSettings):
    GROQ_API_KEY: str = Field(default="")
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    RELEVANCE_MODEL: str = "llama-3.1-8b-instant"
    RESEARCH_MODEL: str = "llama-3.3-70b-versatile"
    VERIFICATION_MODEL: str = "llama-3.1-8b-instant"
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-small"
    RERANKER_MODEL: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    VECTOR_SEARCH_K: int = 15
    RERANKER_TOP_N: int = 5
    HYBRID_RETRIEVER_WEIGHTS: list[float] = [0.4, 0.6]
    MAX_CONTEXT_TOKENS: int = 6000
    MAX_RESEARCH_ITERATIONS: int = 2
    MAX_FILE_SIZE: int = MAX_FILE_SIZE
    MAX_TOTAL_SIZE: int = MAX_TOTAL_SIZE
    ALLOWED_TYPES: list[str] = ALLOWED_TYPES
    CACHE_DIR: str = "document_cache"
    CACHE_EXPIRE_DAYS: int = 7
    CHROMA_DB_PATH: str = "chroma_db"
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 7860
    GRADIO_SHARE: bool = False
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
```

- [ ] **Step 4: Run green tests**

Run: `python -m pytest tests/test_settings_and_language.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add config/settings.py tests/__init__.py tests/conftest.py tests/test_settings_and_language.py requirements.txt
git commit -m "test: establish settings and language contracts"
```

---

### Task 2: Groq Client Boundary and Agents

**Files:**
- Create: `agents/llm_client.py`
- Create: `agents/query_expander.py`
- Create: `tests/test_agents.py`
- Modify: `agents/relevance_checker.py`
- Modify: `agents/research_agent.py`
- Modify: `agents/verification_agent.py`
- Modify: `agents/__init__.py`

- [ ] **Step 1: Write failing tests for agent behavior**

```python
# tests/test_agents.py
from langchain.schema import Document

from agents.query_expander import QueryExpander
from agents.relevance_checker import RelevanceChecker
from agents.research_agent import ResearchAgent, build_context
from agents.verification_agent import VerificationAgent


class FakeChatClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, *, model, messages, temperature, max_tokens):
        self.calls.append({"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens})
        return self.responses.pop(0)

    def stream(self, *, model, messages, temperature, max_tokens):
        self.calls.append({"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens})
        for token in self.responses.pop(0):
            yield token


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    def invoke(self, question):
        return self.docs


def test_relevance_checker_normalizes_valid_labels(settings):
    client = FakeChatClient(["partial."])
    checker = RelevanceChecker(client=client, model="llama-3.1-8b-instant")

    result = checker.check("Can I answer?", FakeRetriever([Document(page_content="context")]))

    assert result == "PARTIAL"


def test_query_expander_keeps_original_and_two_variants():
    client = FakeChatClient(["variant one\nvariant two\nvariant three"])
    expander = QueryExpander(client=client, model="llama-3.1-8b-instant")

    assert expander.expand("original") == ["original", "variant one", "variant two"]


def test_build_context_respects_budget_and_labels_sources():
    docs = [
        Document(page_content="A" * 50, metadata={"source": "a.pdf", "page": 2, "section": "Intro"}),
        Document(page_content="B" * 1000, metadata={"source": "b.pdf", "page": 3, "section": "Body"}),
    ]

    context = build_context(docs, max_tokens=20)

    assert "Source 1" in context
    assert "a.pdf" in context
    assert "Source 2" not in context


def test_research_agent_streams_tokens_and_returns_sources():
    client = FakeChatClient([["Hello", " ", "world"]])
    docs = [Document(page_content="context", metadata={"source": "a.pdf", "page": 1, "section": "Intro"})]
    agent = ResearchAgent(client=client, model="llama-3.3-70b-versatile")

    chunks = list(agent.generate_stream("question", docs))

    assert "".join(chunk for chunk, _ in chunks) == "Hello world"
    assert chunks[-1][1] == docs


def test_verification_parser_supplies_defaults_for_missing_fields():
    agent = VerificationAgent(client=FakeChatClient([]), model="llama-3.3-70b-versatile")

    parsed = agent.parse_verification_response("Supported: YES\nRelevant: YES")

    assert parsed["Supported"] == "YES"
    assert parsed["Relevant"] == "YES"
    assert parsed["Unsupported Claims"] == []
    assert parsed["Contradictions"] == []
```

- [ ] **Step 2: Run red tests**

Run: `python -m pytest tests/test_agents.py -q`

Expected: FAIL due missing `QueryExpander`, missing client boundary, and old WatsonX imports.

- [ ] **Step 3: Implement `agents/llm_client.py`**

```python
from collections.abc import Iterable
from typing import Protocol

from openai import APIConnectionError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config.settings import settings


class ChatClient(Protocol):
    def complete(self, *, model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str: ...
    def stream(self, *, model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> Iterable[str]: ...


class GroqClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        key = api_key if api_key is not None else settings.GROQ_API_KEY
        if not key:
            raise RuntimeError("GROQ_API_KEY is required for live Groq requests.")
        self.client = OpenAI(api_key=key, base_url=base_url or settings.GROQ_BASE_URL)

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
    )
    def complete(self, *, model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Groq returned an empty response.")
        return content

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
    )
    def stream(self, *, model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> Iterable[str]:
        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                yield token
```

- [ ] **Step 4: Rewrite agents around injected clients**

Implement each agent with `client: ChatClient | None = None`; default to `GroqClient()` only at runtime. Prompts must append the shared language instruction and use settings-backed defaults.

- [ ] **Step 5: Run green tests**

Run: `python -m pytest tests/test_agents.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add agents tests/test_agents.py
git commit -m "feat: migrate agents to Groq client boundary"
```

---

### Task 3: Bounded Workflow

**Files:**
- Create: `tests/test_workflow.py`
- Modify: `agents/workflow.py`

- [ ] **Step 1: Write failing workflow tests**

```python
# tests/test_workflow.py
from langchain.schema import Document

from agents.workflow import AgentWorkflow


class FakeRetriever:
    def __init__(self):
        self.docs = [Document(page_content="ctx", metadata={"source": "a.pdf", "page": 1, "section": "Intro"})]

    def invoke(self, question):
        return self.docs


class FakeRelevance:
    def __init__(self, label):
        self.label = label

    def check(self, question, retriever, k=20):
        return self.label


class FakeResearch:
    def __init__(self):
        self.calls = 0

    def generate(self, question, documents):
        self.calls += 1
        return {"draft_answer": f"answer {self.calls}", "source_docs": documents}


class FakeVerifier:
    def __init__(self, reports):
        self.reports = list(reports)

    def check(self, answer, documents):
        return {"verification_report": self.reports.pop(0)}


def test_workflow_ends_without_research_on_no_match():
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


def test_workflow_stops_after_max_research_iterations():
    research = FakeResearch()
    workflow = AgentWorkflow(
        relevance_checker=FakeRelevance("CAN_ANSWER"),
        researcher=research,
        verifier=FakeVerifier(["Supported: NO\nRelevant: YES", "Supported: NO\nRelevant: YES"]),
        max_iterations=2,
    )

    result = workflow.full_pipeline("question", FakeRetriever())

    assert research.calls == 2
    assert result["iteration_count"] == 2
    assert result["draft_answer"] == "answer 2"
```

- [ ] **Step 2: Run red tests**

Run: `python -m pytest tests/test_workflow.py -q`

Expected: FAIL because the current workflow constructor is not injectable and has no iteration counter.

- [ ] **Step 3: Implement typed workflow state and iteration guard**

Update `AgentWorkflow.__init__` to accept injected agents and `max_iterations`. Track `source_docs`, `relevance`, and `iteration_count` in state; `_decide_next_step` returns `end` when the max is reached.

- [ ] **Step 4: Run green tests**

Run: `python -m pytest tests/test_workflow.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/workflow.py tests/test_workflow.py
git commit -m "feat: bound multi-agent workflow iterations"
```

---

### Task 4: Lightweight Document Processing

**Files:**
- Create: `tests/test_document_processor.py`
- Modify: `document_processor/file_handler.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Write failing document processor tests**

```python
# tests/test_document_processor.py
from types import SimpleNamespace

import pytest

from document_processor.file_handler import DocumentProcessor


def uploaded(path):
    return SimpleNamespace(name=str(path))


def test_text_file_returns_chunks_with_required_metadata(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("# Intro\nXin chao tai lieu.\n" * 20, encoding="utf-8")
    processor = DocumentProcessor(cache_dir=tmp_path / "cache")

    chunks = processor.process([uploaded(path)])

    assert chunks
    metadata = chunks[0].metadata
    assert metadata["source"] == "sample.txt"
    assert metadata["file_hash"]
    assert metadata["chunk_index"] == 0
    assert "page" in metadata
    assert metadata["section"]
    assert metadata["language"] in {"vi", "en"}


def test_cache_hit_reuses_previous_chunks(tmp_path):
    path = tmp_path / "sample.md"
    path.write_text("# Title\nContent " * 100, encoding="utf-8")
    processor = DocumentProcessor(cache_dir=tmp_path / "cache")

    first = processor.process([uploaded(path)])
    path.write_text("# Title\nChanged", encoding="utf-8")
    second = processor.process([uploaded(path)])

    assert first != second


def test_unsupported_file_type_raises_value_error(tmp_path):
    path = tmp_path / "bad.exe"
    path.write_bytes(b"not allowed")
    processor = DocumentProcessor(cache_dir=tmp_path / "cache")

    with pytest.raises(ValueError, match="Unsupported file type"):
        processor.process([uploaded(path)])
```

- [ ] **Step 2: Run red tests**

Run: `python -m pytest tests/test_document_processor.py -q`

Expected: FAIL because the current processor imports Docling and does not expose injectable cache/type behavior.

- [ ] **Step 3: Implement parser/chunker/cache**

Implement `DocumentProcessor(cache_dir: Path | str | None = None)` with:

```python
def process(self, files: list[object]) -> list[Document]: ...
def _parse_pdf(self, path: Path) -> str: ...
def _parse_docx(self, path: Path) -> str: ...
def _parse_text(self, path: Path) -> str: ...
def _build_chunks(self, text: str, source: str, file_hash: str) -> list[Document]: ...
```

Use `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100, separators=[...])`. Metadata must include `source`, `file_hash`, `chunk_index`, `page`, `section`, and `language`.

- [ ] **Step 4: Run green tests**

Run: `python -m pytest tests/test_document_processor.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add document_processor/file_handler.py tests/test_document_processor.py requirements.txt
git commit -m "feat: replace document ingestion with lightweight parser"
```

---

### Task 5: Multilingual Retriever and Persistence

**Files:**
- Create: `tests/test_retriever.py`
- Modify: `retriever/builder.py`

- [ ] **Step 1: Write failing retriever tests**

```python
# tests/test_retriever.py
from langchain.schema import Document

from retriever.builder import RetrieverBuilder, combined_file_hash, dedupe_documents


def test_combined_file_hash_is_order_independent():
    assert combined_file_hash(["b", "a"]) == combined_file_hash(["a", "b"])


def test_dedupe_documents_preserves_first_seen_metadata():
    docs = [
        Document(page_content="same", metadata={"source": "a"}),
        Document(page_content="same", metadata={"source": "b"}),
    ]

    result = dedupe_documents(docs)

    assert len(result) == 1
    assert result[0].metadata["source"] == "a"


class FakeVectorStore:
    def __init__(self, docs):
        self.docs = docs

    def as_retriever(self, search_kwargs):
        return self

    def invoke(self, query):
        return self.docs


class FakeFactory:
    def __init__(self):
        self.loaded = []
        self.built = []

    def load_or_build(self, docs, persist_directory):
        self.built.append(persist_directory)
        return FakeVectorStore(docs)


def test_builder_returns_top_reranked_documents(tmp_path):
    docs = [
        Document(page_content="alpha", metadata={"file_hash": "1"}),
        Document(page_content="beta", metadata={"file_hash": "1"}),
    ]
    builder = RetrieverBuilder(vector_factory=FakeFactory(), reranker=lambda query, items, top_n: items[:1], chroma_root=tmp_path)

    retriever = builder.build_hybrid_retriever(docs)

    assert len(retriever.invoke("alpha")) == 1
```

- [ ] **Step 2: Run red tests**

Run: `python -m pytest tests/test_retriever.py -q`

Expected: FAIL because helper functions and injected test seams do not exist.

- [ ] **Step 3: Implement retriever helpers and staged retriever**

Expose:

```python
def combined_file_hash(file_hashes: list[str]) -> str: ...
def dedupe_documents(documents: list[Document]) -> list[Document]: ...
class RetrieverBuilder:
    def __init__(self, embeddings=None, vector_factory=None, reranker=None, chroma_root=None) -> None: ...
    def build_hybrid_retriever(self, docs: list[Document]): ...
```

Runtime default uses HuggingFace E5 embeddings, Chroma persistence, BM25 with `vi_tokenizer`, and multilingual cross-encoder reranking. Tests use injected fakes.

- [ ] **Step 4: Run green tests**

Run: `python -m pytest tests/test_retriever.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add retriever/builder.py tests/test_retriever.py
git commit -m "feat: add multilingual retrieval with persistence seam"
```

---

### Task 6: Gradio UI Helpers and Streaming Integration

**Files:**
- Create: `tests/test_app_helpers.py`
- Modify: `app.py`

- [ ] **Step 1: Write failing UI helper tests**

```python
# tests/test_app_helpers.py
from types import SimpleNamespace

from langchain.schema import Document

from app import format_sources, get_file_hashes


def test_format_sources_renders_filename_page_section_and_excerpt():
    docs = [
        Document(
            page_content="This is a long source excerpt used by the answer.",
            metadata={"source": "report.pdf", "page": 3, "section": "Summary"},
        )
    ]

    rendered = format_sources(docs)

    assert "report.pdf" in rendered
    assert "3" in rendered
    assert "Summary" in rendered
    assert "This is a long source excerpt" in rendered


def test_get_file_hashes_hashes_content_not_filename(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("same", encoding="utf-8")
    b.write_text("same", encoding="utf-8")

    hashes = get_file_hashes([SimpleNamespace(name=str(a)), SimpleNamespace(name=str(b))])

    assert len(hashes) == 1
```

- [ ] **Step 2: Run red tests**

Run: `python -m pytest tests/test_app_helpers.py -q`

Expected: FAIL because helpers are private or not source-aware.

- [ ] **Step 3: Refactor UI**

Expose `get_file_hashes`, `format_sources`, and `process_question_stream`. Update Gradio outputs to answer markdown, verification markdown, source markdown, and session state. Use `settings.SERVER_HOST`, `settings.SERVER_PORT`, and `settings.GRADIO_SHARE`.

- [ ] **Step 4: Run green tests**

Run: `python -m pytest tests/test_app_helpers.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app_helpers.py
git commit -m "feat: add streaming UI helpers and source citations"
```

---

### Task 7: Repository Quality, CI, Docker, README, and Eval Assets

**Files:**
- Create: `.github/workflows/test.yml`
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `README.md`
- Create: `eval/qa_pairs_vi.json`
- Create: `eval/qa_pairs_en.json`
- Create: `eval/run_ragas_eval.py`
- Create: `tests/test_release_artifacts.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing artifact tests**

```python
# tests/test_release_artifacts.py
from pathlib import Path


def test_release_artifacts_exist():
    for path in [
        ".github/workflows/test.yml",
        "Dockerfile",
        "docker-compose.yml",
        "README.md",
        "eval/qa_pairs_vi.json",
        "eval/qa_pairs_en.json",
        "eval/run_ragas_eval.py",
    ]:
        assert Path(path).exists()


def test_readme_does_not_claim_unverified_ragas_results():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "TBD" not in readme
    assert "not yet run" in readme.lower()
    assert "fake" not in readme.lower()
```

- [ ] **Step 2: Run red tests**

Run: `python -m pytest tests/test_release_artifacts.py -q`

Expected: FAIL because release artifacts do not exist.

- [ ] **Step 3: Add CI and packaging**

Use GitHub Actions with Python 3.11 and `python -m pytest tests -q`. Docker should expose port `7860`, install `requirements.txt`, and run `python app.py`. Compose should mount `document_cache` and `chroma_db`.

- [ ] **Step 4: Add README and eval assets**

README must include quick start, environment setup, architecture, features, limitations, test command, Docker command, eval instructions, and HuggingFace deployment notes. Evaluation section must state results are not yet run until real credentials/data are used.

- [ ] **Step 5: Run green tests**

Run: `python -m pytest tests/test_release_artifacts.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add .github Dockerfile docker-compose.yml README.md eval tests/test_release_artifacts.py .gitignore
git commit -m "chore: add CI Docker docs and evaluation assets"
```

---

### Task 8: Final Verification and Push

**Files:**
- Verify all changed files.

- [ ] **Step 1: Run full offline test suite**

Run: `python -m pytest tests -q`

Expected: all collected tests PASS.

- [ ] **Step 2: Run import smoke checks**

Run:

```bash
python -c "from config.settings import settings; from agents.workflow import AgentWorkflow; from document_processor.file_handler import DocumentProcessor; from retriever.builder import RetrieverBuilder; print('imports ok')"
```

Expected: prints `imports ok`.

- [ ] **Step 3: Inspect for legacy dependencies and prints**

Run:

```bash
rg -n "ibm_watsonx|Watsonx|docling|DocumentConverter|print\(" . -g "*.py" -g "!test/**"
```

Expected: no production hits.

- [ ] **Step 4: Review git diff**

Run: `git status --short --branch` and `git log --oneline --decorate --max-count=8`.

Expected: clean or only intentional uncommitted changes before final commit.

- [ ] **Step 5: Push**

Run: `git push origin main`

Expected: remote `main` receives all completed commits.

---

## Self-Review Notes

- Spec coverage: settings, Groq agents, workflow guard, document parsing, multilingual retrieval, UI streaming/sources, tests, CI, Docker, README, eval assets, and truthful deployment/evaluation reporting each map to a task.
- Placeholder scan: no implementation task relies on unspecified files or unnamed future behavior.
- Type consistency: public names used by tests are defined in the corresponding implementation tasks before later tasks depend on them.
