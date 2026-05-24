from types import SimpleNamespace

from langchain_core.documents import Document

from app import format_sources, get_file_hashes, process_question_stream


def test_format_sources_renders_filename_page_section_and_excerpt() -> None:
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


def test_get_file_hashes_hashes_content_not_filename(tmp_path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("same", encoding="utf-8")
    b.write_text("same", encoding="utf-8")

    hashes = get_file_hashes([SimpleNamespace(name=str(a)), SimpleNamespace(name=str(b))])

    assert len(hashes) == 1


def test_process_question_stream_yields_incremental_workflow_answers(tmp_path) -> None:
    path = tmp_path / "paper.txt"
    path.write_text("evidence", encoding="utf-8")
    doc = Document(
        page_content="evidence",
        metadata={"source": "paper.txt", "page": None, "section": "Intro"},
    )

    class FakeProcessor:
        def process(self, _files):
            return [doc]

    class FakeBuilder:
        def build_hybrid_retriever(self, _chunks):
            return object()

    class FakeWorkflow:
        def stream_pipeline(self, question, retriever):
            yield {"draft_answer": "partial", "verification_report": "", "source_docs": [doc]}
            yield {"draft_answer": "partial answer", "verification_report": "Supported: YES", "source_docs": [doc]}

    outputs = list(
        process_question_stream(
            "question",
            [SimpleNamespace(name=str(path))],
            {"file_hashes": frozenset(), "retriever": None},
            FakeProcessor(),
            FakeBuilder(),
            FakeWorkflow(),
        )
    )

    assert outputs[-2][0] == "partial"
    assert outputs[-1][0] == "partial answer"
    assert outputs[-1][1] == "Supported: YES"
