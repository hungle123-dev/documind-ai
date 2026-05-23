import hashlib
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from config.settings import settings
from utils.language import detect_language
from utils.logging import logger


class DocumentProcessor:
    def __init__(self, cache_dir: Path | str | None = None) -> None:
        self.cache_dir = Path(cache_dir or settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def validate_files(self, files: list[Any]) -> None:
        total_size = 0
        for file in files:
            path = self._file_path(file)
            if path.suffix.lower() not in settings.ALLOWED_TYPES:
                raise ValueError(f"Unsupported file type: {path.suffix}")
            size = path.stat().st_size
            if size > settings.MAX_FILE_SIZE:
                limit_mb = settings.MAX_FILE_SIZE // 1024 // 1024
                raise ValueError(f"File {path.name} exceeds {limit_mb}MB limit")
            total_size += size

        if total_size > settings.MAX_TOTAL_SIZE:
            limit_mb = settings.MAX_TOTAL_SIZE // 1024 // 1024
            raise ValueError(f"Total size exceeds {limit_mb}MB limit")

    def process(self, files: list[Any]) -> list[Document]:
        self.validate_files(files)
        chunks: list[Document] = []
        seen_content_hashes: set[str] = set()

        for file in files:
            path = self._file_path(file)
            file_hash = self._generate_hash(path.read_bytes())
            cache_path = self.cache_dir / f"{file_hash}.pkl"

            if self._is_cache_valid(cache_path):
                logger.info(f"Loading cached chunks for {path.name}")
                file_chunks = self._load_from_cache(cache_path)
            else:
                logger.info(f"Processing document {path.name}")
                file_chunks = self._process_file(path, file_hash)
                self._save_to_cache(file_chunks, cache_path)

            for chunk in file_chunks:
                chunk_hash = self._generate_hash(chunk.page_content.encode("utf-8"))
                if chunk_hash not in seen_content_hashes:
                    chunks.append(chunk)
                    seen_content_hashes.add(chunk_hash)

        logger.info(f"Total unique chunks: {len(chunks)}")
        return chunks

    def _process_file(self, path: Path, file_hash: str) -> list[Document]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            text = self._parse_pdf(path)
        elif suffix == ".docx":
            text = self._parse_docx(path)
        elif suffix in {".txt", ".md"}:
            text = self._parse_text(path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        if not text.strip():
            logger.warning(f"No readable text extracted from {path.name}")
            return []

        return self._build_chunks(text, source=path.name, file_hash=file_hash)

    def _parse_pdf(self, path: Path) -> str:
        try:
            import pymupdf4llm
        except ImportError as exc:
            raise RuntimeError("pymupdf4llm is required to parse PDF files.") from exc

        markdown = pymupdf4llm.to_markdown(str(path))
        if len(markdown.strip()) < 100:
            logger.warning(f"{path.name} appears to be scanned or has no extractable text.")
            return ""
        return markdown

    def _parse_docx(self, path: Path) -> str:
        from docx import Document as DocxDocument

        document = DocxDocument(str(path))
        parts: list[str] = []
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if text:
                parts.append(text)
        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    parts.append(" | ".join(cells))
        return "\n\n".join(parts)

    def _parse_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def _build_chunks(self, text: str, source: str, file_hash: str) -> list[Document]:
        language = detect_language(text)
        raw_chunks = self._split_text(text)
        documents: list[Document] = []

        for index, chunk in enumerate(raw_chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": source,
                        "file_hash": file_hash,
                        "chunk_index": index,
                        "page": None,
                        "section": self._nearest_section(chunk),
                        "language": language,
                    },
                )
            )
        return documents

    def _split_text(self, text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> list[str]:
        cleaned = text.strip()
        if not cleaned:
            return []
        chunks: list[str] = []
        start = 0
        while start < len(cleaned):
            end = min(start + chunk_size, len(cleaned))
            if end < len(cleaned):
                split_at = max(
                    cleaned.rfind("\n\n", start, end),
                    cleaned.rfind("\n", start, end),
                    cleaned.rfind(". ", start, end),
                    cleaned.rfind(" ", start, end),
                )
                if split_at > start + chunk_size // 2:
                    end = split_at + 1
            chunk = cleaned[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(cleaned):
                break
            start = max(0, end - chunk_overlap)
        return chunks

    def _nearest_section(self, chunk: str) -> str:
        for line in chunk.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip() or "N/A"
        first_line = chunk.splitlines()[0].strip() if chunk.splitlines() else ""
        return first_line[:80] if first_line else "N/A"

    def _file_path(self, file: Any) -> Path:
        name = getattr(file, "name", file)
        return Path(name)

    def _generate_hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def _save_to_cache(self, chunks: list[Document], cache_path: Path) -> None:
        with cache_path.open("wb") as file:
            pickle.dump({"timestamp": datetime.now().timestamp(), "chunks": chunks}, file)

    def _load_from_cache(self, cache_path: Path) -> list[Document]:
        with cache_path.open("rb") as file:
            data = pickle.load(file)
        return data["chunks"]

    def _is_cache_valid(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return cache_age < timedelta(days=settings.CACHE_EXPIRE_DAYS)
