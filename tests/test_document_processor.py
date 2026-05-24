import sys
import pickle
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

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
    assert metadata["processor_schema_version"] == DocumentProcessor.CACHE_SCHEMA_VERSION
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


def test_cache_directory_is_recreated_before_saving(tmp_path) -> None:
    path = tmp_path / "sample.md"
    path.write_text("# Title\nContent " * 80, encoding="utf-8")
    cache_dir = tmp_path / "cache"
    processor = DocumentProcessor(cache_dir=cache_dir)
    cache_dir.rmdir()

    chunks = processor.process([uploaded(path)])

    assert chunks
    assert cache_dir.exists()


def test_processor_accepts_path_objects(tmp_path) -> None:
    path = tmp_path / "sample.md"
    path.write_text("# Title\nPath input content " * 80, encoding="utf-8")
    processor = DocumentProcessor(cache_dir=tmp_path / "cache")

    chunks = processor.process([path])

    assert chunks
    assert chunks[0].metadata["source"] == "sample.md"


def test_pdf_chunks_preserve_page_numbers(monkeypatch, tmp_path) -> None:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF fake document")
    page_chunk_calls: list[bool] = []

    def to_markdown(_path: str, *, page_chunks: bool = False):
        page_chunk_calls.append(page_chunks)
        if page_chunks:
            return [
                {"text": "# Introduction\n" + "page one evidence " * 20, "metadata": {"page_number": 1}},
                {"text": "# Results\n" + "page two evidence " * 20, "metadata": {"page_number": 2}},
            ]
        return "page one evidence " * 20 + "page two evidence " * 20

    monkeypatch.setitem(sys.modules, "pymupdf4llm", SimpleNamespace(to_markdown=to_markdown))
    processor = DocumentProcessor(cache_dir=tmp_path / "cache")

    chunks = processor.process([path])

    assert page_chunk_calls == [True]
    assert {chunk.metadata["page"] for chunk in chunks} == {1, 2}


def test_markdown_tables_stay_with_header_and_rows(tmp_path) -> None:
    repeated_rows = "\n".join(
        f"|Model {index}|{20 + index / 10:.1f}<br>{30 + index / 10:.1f}|"
        for index in range(35)
    )
    table = (
        "# Results\n"
        "|Model|BLEU<br>EN-DE<br>EN-FR|\n"
        "|---|---|\n"
        f"{repeated_rows}\n"
        "|Transformer (base model)<br>Transformer (big)|27.3<br>38.1<br>28.4<br>41.8|\n"
        "\nThe surrounding prose can be split normally."
    )
    processor = DocumentProcessor(cache_dir=tmp_path / "cache")

    chunks = processor._build_chunks(
        table,
        source="paper.pdf",
        file_hash="hash",
        page=8,
        language="en",
    )

    assert any(
        "Model|BLEU" in chunk.page_content
        and "Transformer (base model)" in chunk.page_content
        and "27.3" in chunk.page_content
        for chunk in chunks
    )


def test_markdown_tables_are_linearized_for_retrieval(tmp_path) -> None:
    table = (
        "|Model|BLEU<br>EN-DE<br>EN-FR|Training Cost|\n"
        "|---|---|---|\n"
        "|Transformer (base model)<br>Transformer (big)|27.3<br>38.1<br>28.4<br>41.8|12 hours<br>3.5 days|\n"
    )
    processor = DocumentProcessor(cache_dir=tmp_path / "cache")

    chunks = processor._build_chunks(
        table,
        source="paper.pdf",
        file_hash="hash",
        page=8,
        language="en",
    )

    assert any(
        "Transformer (base model): BLEU EN-DE = 27.3" in chunk.page_content
        and "Transformer (big): BLEU EN-DE = 28.4" in chunk.page_content
        for chunk in chunks
    )


def test_markdown_table_chunks_include_preceding_caption(tmp_path) -> None:
    text = (
        "Table 2: BLEU scores on English-to-German and English-to-French tests.\n"
        "|Model|BLEU<br>EN-DE<br>EN-FR|\n"
        "|---|---|\n"
        "|Transformer (base model)<br>Transformer (big)|27.3<br>38.1<br>28.4<br>41.8|\n"
    )
    processor = DocumentProcessor(cache_dir=tmp_path / "cache")

    chunks = processor._build_chunks(
        text,
        source="paper.pdf",
        file_hash="hash",
        page=8,
        language="en",
    )

    assert any(
        chunk.page_content.startswith("Table 2: BLEU scores")
        and "Transformer (base model): BLEU EN-DE = 27.3" in chunk.page_content
        for chunk in chunks
    )


def test_plain_text_overlap_starts_at_sentence_boundary(tmp_path) -> None:
    text = (
        "A short introduction sentence. "
        + "The big Transformer model established the result and training took 3.5 days on 8 P100 GPUs. "
        + "Even the base model was competitive at lower cost. "
        + "Additional explanatory text follows. "
    ) * 6
    processor = DocumentProcessor(cache_dir=tmp_path / "cache")

    chunks = processor._split_plain_text(text, chunk_size=220, chunk_overlap=80)

    assert len(chunks) > 1
    assert all(not chunk.startswith("training took 3.5 days") for chunk in chunks[1:])
    assert all(chunk[0].isupper() for chunk in chunks[1:] if chunk)


def test_default_overlap_preserves_disambiguating_context(tmp_path) -> None:
    text = (
        "We trained our models on one machine with 8 NVIDIA P100 GPUs. "
        "For our base models using the hyperparameters described throughout the paper, "
        "each training step took about 0.4 seconds. "
        "We trained the base models for a total of 100,000 steps or 12 hours. "
        "For our big models, step time was 1.0 seconds. "
        "The big models were trained for 300,000 steps (3.5 days). "
    ) * 4
    processor = DocumentProcessor(cache_dir=tmp_path / "cache")

    chunks = processor._split_plain_text(text, chunk_size=220)

    assert any(
        "8 NVIDIA P100 GPUs" in chunk
        and "base models" in chunk
        and "12 hours" in chunk
        for chunk in chunks
    )


def test_legacy_cache_is_reprocessed_after_metadata_schema_change(monkeypatch, tmp_path) -> None:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF cached document")
    processor = DocumentProcessor(cache_dir=tmp_path / "cache")
    cache_path = processor.cache_dir / f"{processor._generate_hash(path.read_bytes())}.pkl"
    with cache_path.open("wb") as cached_file:
        pickle.dump(
            {"chunks": [Document(page_content="legacy", metadata={"page": None})]},
            cached_file,
        )

    fresh = Document(page_content="fresh", metadata={"page": 1})
    monkeypatch.setattr(processor, "_process_file", lambda _path, _hash: [fresh])

    chunks = processor.process([path])

    assert chunks == [fresh]


@pytest.mark.parametrize("schema_version", [2, 3, 4, 5, 6])
def test_old_schema_cache_is_reprocessed_after_table_chunking_change(
    monkeypatch,
    tmp_path,
    schema_version: int,
) -> None:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF cached document")
    processor = DocumentProcessor(cache_dir=tmp_path / "cache")
    cache_path = processor.cache_dir / f"{processor._generate_hash(path.read_bytes())}.pkl"
    with cache_path.open("wb") as cached_file:
        pickle.dump(
            {
                "schema_version": schema_version,
                "chunks": [Document(page_content="split legacy table", metadata={"page": 8})],
            },
            cached_file,
        )

    fresh = Document(page_content="fresh table chunk", metadata={"page": 8})
    monkeypatch.setattr(processor, "_process_file", lambda _path, _hash: [fresh])

    chunks = processor.process([path])

    assert chunks == [fresh]


def test_unsupported_file_type_raises_value_error(tmp_path) -> None:
    path = tmp_path / "bad.exe"
    path.write_bytes(b"not allowed")
    processor = DocumentProcessor(cache_dir=tmp_path / "cache")

    with pytest.raises(ValueError, match="Unsupported file type"):
        processor.process([uploaded(path)])
