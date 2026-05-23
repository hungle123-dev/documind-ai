from types import SimpleNamespace

from langchain_core.documents import Document

from app import format_sources, get_file_hashes


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
