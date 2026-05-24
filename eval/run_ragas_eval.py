import argparse
import json
from pathlib import Path

from loguru import logger


def load_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    required = {"question", "answer", "contexts", "ground_truth"}
    missing = [index for index, row in enumerate(rows, start=1) if not required.issubset(row)]
    if missing:
        raise ValueError(
            "Evaluation input must contain question, answer, contexts, and ground_truth "
            f"for every row. Missing fields in rows: {missing}"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation for DocuMind AI outputs.")
    parser.add_argument("--input", required=True, help="Path to JSON rows with question, answer, contexts, ground_truth.")
    parser.add_argument("--output", default="eval/ragas_results.json", help="Where to write metric results.")
    args = parser.parse_args()

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError as exc:
        raise SystemExit("Install evaluation extras first: pip install ragas datasets") from exc

    rows = load_rows(Path(args.input))
    dataset = Dataset.from_list(rows)
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote RAGAS results to {}", output)


if __name__ == "__main__":
    main()
