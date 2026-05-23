import hashlib
from collections.abc import Generator
from pathlib import Path
from typing import Any

import gradio as gr
from langchain_core.documents import Document

from agents.workflow import AgentWorkflow
from config.settings import settings
from document_processor.file_handler import DocumentProcessor
from retriever.builder import RetrieverBuilder
from utils.logging import logger


SessionState = dict[str, Any]


def get_file_hashes(uploaded_files: list[Any]) -> frozenset[str]:
    hashes: set[str] = set()
    for file in uploaded_files:
        path = Path(getattr(file, "name", file))
        hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
    return frozenset(hashes)


def format_sources(docs: list[Document]) -> str:
    if not docs:
        return "No sources used."

    rendered: list[str] = []
    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata
        source = metadata.get("source", "unknown")
        page = metadata.get("page")
        page_text = str(page) if page not in {None, ""} else "N/A"
        section = metadata.get("section") or "N/A"
        excerpt = " ".join(doc.page_content.split())[:300]
        rendered.append(
            f"**[{index}] {source}**  \n"
            f"Page: {page_text} | Section: {section}\n\n"
            f"> {excerpt}"
        )
    return "\n\n".join(rendered)


def process_question_stream(
    question_text: str,
    uploaded_files: list[Any],
    state: SessionState,
    processor: DocumentProcessor,
    retriever_builder: RetrieverBuilder,
    workflow: AgentWorkflow,
) -> Generator[tuple[str, str, str, SessionState], None, None]:
    try:
        question = question_text.strip()
        if not question:
            raise ValueError("Question cannot be empty.")
        if not uploaded_files:
            raise ValueError("Upload at least one document.")

        current_hashes = get_file_hashes(uploaded_files)
        if state.get("retriever") is None or current_hashes != state.get("file_hashes"):
            yield "Processing documents...", "", "", state
            chunks = processor.process(uploaded_files)
            if not chunks:
                raise ValueError("No readable text was extracted from the uploaded documents.")
            state["retriever"] = retriever_builder.build_hybrid_retriever(chunks)
            state["file_hashes"] = current_hashes

        yield "Retrieving evidence and generating answer...", "", "", state
        result = workflow.full_pipeline(question=question, retriever=state["retriever"])
        answer = str(result.get("draft_answer", ""))
        verification = str(result.get("verification_report", ""))
        sources = format_sources(result.get("source_docs", []))
        yield answer, verification, sources, state
    except Exception as exc:
        logger.exception(f"Processing error: {exc}")
        yield f"Error: {exc}", "", "", state


def main() -> None:
    processor = DocumentProcessor()
    retriever_builder = RetrieverBuilder()
    workflow = AgentWorkflow()

    css = """
    .app-title { text-align: center; margin-bottom: 0.25rem; }
    .app-subtitle { text-align: center; color: #555; margin-bottom: 1.5rem; }
    """

    theme = gr.themes.Soft()

    with gr.Blocks(title="DocuMind AI") as demo:
        gr.Markdown("# DocuMind AI", elem_classes="app-title")
        gr.Markdown(
            "Multilingual document Q&A for Vietnamese and English files.",
            elem_classes="app-subtitle",
        )

        session_state = gr.State({"file_hashes": frozenset(), "retriever": None})

        with gr.Row():
            with gr.Column(scale=1):
                files = gr.Files(
                    label="Documents",
                    file_types=settings.ALLOWED_TYPES,
                )
                question = gr.Textbox(label="Question", lines=4)
                submit_btn = gr.Button("Ask", variant="primary")

            with gr.Column(scale=1):
                answer_output = gr.Markdown(label="Answer")
                verification_output = gr.Markdown(label="Verification")
                sources_output = gr.Markdown(label="Sources")

        submit_btn.click(
            fn=lambda q, f, s: process_question_stream(
                q,
                f,
                s,
                processor,
                retriever_builder,
                workflow,
            ),
            inputs=[question, files, session_state],
            outputs=[answer_output, verification_output, sources_output, session_state],
        )

    demo.launch(
        server_name=settings.SERVER_HOST,
        server_port=settings.SERVER_PORT,
        share=settings.GRADIO_SHARE,
        theme=theme,
        css=css,
    )


if __name__ == "__main__":
    main()
