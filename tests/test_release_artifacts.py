import json
from pathlib import Path


def test_release_artifacts_exist() -> None:
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


def test_readme_does_not_claim_unverified_ragas_results() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "TBD" not in readme
    assert "not yet run" in readme.lower()
    assert "fake" not in readme.lower()


def test_eval_sets_have_at_least_fifteen_pairs_each() -> None:
    for path in ["eval/qa_pairs_vi.json", "eval/qa_pairs_en.json"]:
        pairs = json.loads(Path(path).read_text(encoding="utf-8"))
        assert len(pairs) >= 15
        assert all("question" in pair and "ground_truth" in pair for pair in pairs)
