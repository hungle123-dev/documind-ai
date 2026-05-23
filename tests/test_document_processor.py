from types import SimpleNamespace

import pytest

from document_processor.file_handler import DocumentProcessor


def uploaded(path):
    return SimpleNamespace(name=str(path))


def test_text_file_returns_chunks_with_required_metadata(tmp_path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("# Intro\nXin chào tài liệu.\n" * 40, encoding="utf-8")
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


def test_file_hash_change_reprocesses_document(tmp_path) -> None:
    path = tmp_path / "sample.md"
    path.write_text("# Title\nOriginal content " * 80, encoding="utf-8")
    processor = DocumentProcessor(cache_dir=tmp_path / "cache")

    first = processor.process([uploaded(path)])
    path.write_text("# Title\nChanged content " * 80, encoding="utf-8")
    second = processor.process([uploaded(path)])

    assert first[0].page_content != second[0].page_content
    assert first[0].metadata["file_hash"] != second[0].metadata["file_hash"]


def test_unsupported_file_type_raises_value_error(tmp_path) -> None:
    path = tmp_path / "bad.exe"
    path.write_bytes(b"not allowed")
    processor = DocumentProcessor(cache_dir=tmp_path / "cache")

    with pytest.raises(ValueError, match="Unsupported file type"):
        processor.process([uploaded(path)])
