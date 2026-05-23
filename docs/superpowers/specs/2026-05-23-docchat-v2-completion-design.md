# DocChat v2 Completion Design

**Date:** 2026-05-23  
**Scope decision:** Complete all production-ready repository work first. Evaluation results and hosted deployment must only be reported after they have actually been executed with the required credentials and data.

## 1. Purpose

DocChat v2 is a bilingual Vietnamese-English document question-answering application. It should ingest common document formats on modest hardware, retrieve evidence effectively in both languages, produce grounded answers through a bounded multi-agent workflow, and show the evidence used to the user.

The existing repository is a partial v1 implementation. It still uses IBM WatsonX and Docling at runtime, builds an English-oriented retriever without durable embedding reuse, has no collected automated tests, has no streaming/source display, and can repeat research indefinitely after a failed verification. The completed v2 replaces those weaknesses rather than patching individual symptoms.

## 2. Delivery Boundary

### Included

- Grok xAI client migration for relevance classification, answer generation, query expansion, and verification.
- Bounded LangGraph workflow with explicit typed state and source propagation.
- Lightweight PDF/DOCX/TXT/MD processing with metadata and content-hash caching.
- Vietnamese-aware hybrid retrieval, multilingual embeddings, multilingual reranking, and persisted Chroma indexes.
- Streaming Gradio answer rendering and visible source citations.
- Automated tests that do not contact live LLM services.
- CI, Docker packaging, README documentation, and evaluation inputs/runner or notebook that can be executed later.
- HuggingFace Spaces-compatible repository packaging and deployment instructions.

### Excluded Until Externally Verifiable

- Publishing RAGAS metric numbers before the evaluation has been executed against real prepared data and configured model access.
- Claiming a hosted HuggingFace deployment exists before deployment credentials are available and the hosted app has been tested.
- New product features outside the PRD, such as authentication, persistent chat history, fine-tuning, or OCR guarantees for non-Unicode Vietnamese fonts.

## 3. Approach Selection

The implementation will follow vertical operational slices, each protected by tests before the next layer is connected:

1. Establish contracts, settings, and test seams.
2. Migrate LLM-facing agents and bound the workflow.
3. Replace document ingestion and caching.
4. Build multilingual retrieval and persistence.
5. Integrate streaming UI and source presentation.
6. Add repository quality and release artifacts.

This approach is preferred over a bulk rewrite because the installed environment may expose newer LangChain and Gradio APIs than examples in the PRD. Small verified slices make dependency/API incompatibilities diagnosable and keep the user-facing pipeline continuously understandable.

## 4. Architecture

### 4.1 Primary Runtime Flow

1. The user uploads one or more supported documents and asks a question.
2. The application hashes the file contents to determine document-cache and vector-index reuse.
3. `DocumentProcessor` validates inputs, parses text, splits content, and returns `Document` chunks with complete citation metadata.
4. `RetrieverBuilder` loads or creates the Chroma collection for the selected file set, combines semantic and BM25 retrieval, applies query expansion, and reranks multilingual candidate chunks.
5. `AgentWorkflow` retrieves evidence and runs relevance checking, research generation, and verification with a strict research-attempt limit.
6. The UI streams answer text while retaining the exact source chunks used; it renders source filenames, page/section context, and excerpts in a dedicated panel.

### 4.2 Component Boundaries

| Component | Responsibility | Must Not Do |
| --- | --- | --- |
| `config/settings.py` | Environment-backed settings and default model/retrieval/server limits | Instantiate network clients or hardcode runtime decisions outside settings |
| `agents/*` | Prompts, Grok calls, response parsing, context budgeting | Parse files or create vector stores |
| `agents/workflow.py` | Typed LangGraph transitions and iteration bounds | Hide retrieval metadata or loop without limit |
| `document_processor/file_handler.py` | Validation, parsing, splitting, chunk metadata, document cache | Embed documents or invoke LLMs |
| `retriever/builder.py` | Vector persistence, BM25, expansion, reranking | Own UI state or answer generation |
| `app.py` | Gradio composition, session lifecycle, streaming rendering, source formatting | Contain retrieval/model implementation details |
| `tests/` | Behavioral verification with fakes/mocks at external boundaries | Depend on live xAI requests or downloaded heavyweight models |

## 5. Stable Contracts

### 5.1 Settings

All configurable runtime behavior must be exposed through `settings`, including:

- `XAI_API_KEY`, `XAI_BASE_URL`, `RELEVANCE_MODEL`, `RESEARCH_MODEL`, `VERIFICATION_MODEL`.
- `EMBEDDING_MODEL`, `RERANKER_MODEL`, `VECTOR_SEARCH_K`, `RERANKER_TOP_N`, `HYBRID_RETRIEVER_WEIGHTS`.
- `MAX_CONTEXT_TOKENS`, `MAX_RESEARCH_ITERATIONS`.
- `MAX_FILE_SIZE`, `MAX_TOTAL_SIZE`, `ALLOWED_TYPES`, `CACHE_DIR`, `CACHE_EXPIRE_DAYS`, `CHROMA_DB_PATH`.
- `SERVER_HOST`, `SERVER_PORT`, and a setting governing whether a public Gradio share URL is enabled.

The application must fail with an actionable configuration error when a live Grok call is requested without a usable key. Unit tests inject fake clients rather than requiring an environment secret.

### 5.2 Chunk Metadata

Every retrievable `Document` chunk must carry:

```python
{
    "source": "filename.pdf",
    "file_hash": "sha256...",
    "chunk_index": 0,
    "page": 1,
    "section": "Nearest heading or N/A",
    "language": "vi",
}
```

For formats with no meaningful page number, `page` is set to a consistent sentinel such as `None`; downstream formatting renders it deliberately rather than guessing.

### 5.3 Workflow Result

The synchronous pipeline result must expose:

```python
{
    "draft_answer": str,
    "verification_report": str,
    "source_docs": list[Document],
    "relevance": str,
    "iteration_count": int,
}
```

Streaming must yield incremental answer content while preserving source documents for final source rendering and verification status for the completed interaction.

## 6. Component Design

### 6.1 Grok Agents

The agents use the OpenAI-compatible xAI endpoint through a client boundary that can be injected in tests. Every live API operation uses retry/backoff for transient rate-limit and connection failures. Model identifiers and token/temperature configuration come from settings.

- `RelevanceChecker` returns only `CAN_ANSWER`, `PARTIAL`, or `NO_MATCH`, treating invalid model output as a controlled failure rather than silently proceeding.
- `QueryExpander` generates concise retrieval variants and always retains the original question.
- `ResearchAgent` builds budgeted, citation-labelled context and supports both normal completion and streamed completion.
- `VerificationAgent` parses a structured verification contract and returns safe negative verification on malformed model responses.

All prompts require answering in the question's language. They must state that answers are grounded only in provided context.

### 6.2 Workflow

The workflow starts from retrieved evidence, not from unbounded repeated retrieval. `NO_MATCH` terminates with a language-appropriate insufficiency message and no invented evidence. `CAN_ANSWER` and `PARTIAL` proceed to research and verification.

When verification finds unsupported or irrelevant output, research can repeat only until `MAX_RESEARCH_ITERATIONS` is reached. Once the bound is reached, the workflow returns the latest answer plus its verification report so the UI can communicate uncertainty without hanging.

### 6.3 Document Ingestion

`DocumentProcessor` accepts `.pdf`, `.docx`, `.txt`, and `.md` after both per-file and total-size validation. It uses:

- `pymupdf4llm` for PDF text-to-markdown extraction.
- `python-docx` for paragraph and table extraction.
- UTF-8 text reading for TXT and Markdown, with clear errors for unreadable content.
- `RecursiveCharacterTextSplitter` with section-friendly separators for bounded chunks.

For PDF content that extracts fewer than a practical minimum of readable characters, the processor logs that the document appears scanned and returns no misleading searchable chunks. Suspect Vietnamese legacy-font extraction produces a warning and is documented as a limitation. Parsed chunk lists are cached by file hash and cache age, with metadata preserved.

### 6.4 Retrieval

Retrieval consists of explicit stages:

1. Instantiate multilingual E5 embeddings with correct query/document prefixes and normalized vectors.
2. Compute a deterministic combined hash for the active input files.
3. Load the persisted Chroma vector store for that hash or build it once from the document chunks.
4. Build Vietnamese-aware BM25 using `underthesea` tokenization with a controlled fallback.
5. Retrieve a broad candidate pool from semantic and lexical paths.
6. Apply query variants and deduplicate candidates.
7. Rerank with a multilingual cross-encoder and return the configured top evidence chunks.

Tests use fakes around embeddings, vector storage, expansion, and reranking to validate composition and cache behavior without downloading models.

### 6.5 User Interface

The first screen remains the usable DocChat tool rather than a marketing page. It contains document upload, question input, submit action, streamed answer area, verification status, and sources area.

User-facing behavior:

- Empty questions and missing uploads are rejected with clear messages.
- A new file set rebuilds or reloads its retriever; repeated questions against the same file hashes reuse the session retriever.
- Answers are rendered progressively as text arrives.
- Sources are presented with file name, page or section where available, and a concise excerpt.
- Runtime model/parser failures are logged in detail and translated into actionable UI messages.

The UI removes references to Docling and does not automatically create a public share URL. Host, port, and sharing are settings-driven.

## 7. Error Handling and Observability

- Use Loguru for application logging; production source files contain no `print()` calls.
- Validation errors are shown to the user and do not trigger downstream processing.
- Parser errors identify the affected file while allowing independently valid files to be processed when meaningful.
- An empty usable document set blocks retrieval with a direct explanation.
- Network/LLM failures retry only transient cases, then surface a controlled failure.
- Persistence corruption or vector-store load failures are logged and may trigger a rebuild only for the affected deterministic cache path.
- Cache cleanup applies to expired document and vector artifacts without touching active paths.

## 8. Testing Strategy

Behavioral tests will be written before the related implementation changes and organized under `tests/`.

| Area | Required Behavior |
| --- | --- |
| Settings/language | Environment defaults and overrides, Vietnamese/English handling, tokenizer fallback |
| Agents | Prompt language instruction, valid/invalid relevance parsing, retry-call boundary, context budgeting, stream assembly, verification parsing |
| Workflow | `NO_MATCH` termination, successful sequence, failed verification retry, hard maximum research iterations, source retention |
| Document processor | Each supported format, metadata completeness, hash cache hit, limits/unsupported type, scanned PDF warning |
| Retriever | Combined-hash persistence reuse, BM25 preprocessing, deduplication, reranker top-N behavior, no heavyweight/network dependence in unit tests |
| UI helpers | File hashing, source formatting, state reuse and user-facing validation paths |

Integration-level tests will compose faked agents/retrievers and real lightweight document samples. Live API calls, full model download, real RAGAS evaluation, Docker image build, and browser smoke testing are separate verification actions, reported truthfully according to what the environment supports.

## 9. Repository and Release Artifacts

- `.github/workflows/test.yml` runs the offline test suite with a mock API-key environment value.
- `Dockerfile` and `docker-compose.yml` define local/container operation and persisted cache volumes.
- `README.md` documents setup, architecture, capabilities, limitations, tests, Docker, evaluation execution, and HuggingFace Spaces deployment steps.
- `eval/` contains bilingual evaluation input sets and an executable evaluation workflow. Results in README remain explicitly unreported until produced by an actual evaluation run.
- HuggingFace compatibility files and instructions prepare deployment; they do not assert a live deployment.

## 10. Implementation Sequence

1. Create `tests/` fixtures and core contract tests; migrate settings and shared utility contracts.
2. Replace WatsonX agent code with injectable Grok-based agents and implement bounded workflow tests.
3. Replace Docling ingestion with lightweight typed parsing, chunk metadata, and document-cache tests.
4. Replace English-only retrieval with multilingual staged retrieval, persistence, and isolated tests.
5. Refactor `app.py` around session reuse, streamed answers, sources, and UI helper tests.
6. Add CI, documentation, evaluation assets, and deployment packaging.
7. Run available offline tests and static/import/runtime smoke checks; execute live/model/container checks only when their dependencies and credentials exist.

## 11. Acceptance Criteria

- No runtime imports or dependencies on WatsonX or Docling remain.
- All new application functions use type hints and application diagnostics use Loguru.
- Both Vietnamese and English content flow through metadata-preserving parsing, multilingual retrieval, language-aware prompting, and source rendering.
- Re-research cannot run indefinitely.
- Repeated file sets reuse content and embedding persistence as designed.
- Offline automated tests execute without a real xAI key or network model calls.
- CI, Docker, README, and evaluation/deployment preparation are present and aligned with actual behavior.
- No fabricated metric, deployment, or live-service success statement appears in repository documentation.
