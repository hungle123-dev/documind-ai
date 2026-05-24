import json
from pathlib import Path

from config.settings import Settings


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


def test_ci_uses_groq_test_configuration() -> None:
    workflow = Path(".github/workflows/test.yml").read_text(encoding="utf-8")

    assert "GROQ_API_KEY" in workflow
    assert "XAI_API_KEY" not in workflow


def test_documented_model_defaults_match_settings() -> None:
    settings = Settings(_env_file=None, GROQ_API_KEY="gsk-test-key")
    env_example = Path(".env.example").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")

    for variable in [
        "RELEVANCE_MODEL",
        "RESEARCH_MODEL",
        "VERIFICATION_MODEL",
    ]:
        value = getattr(settings, variable)
        assert f"{variable}={value}" in env_example
        assert f"`{variable}`: default `{value}`." in readme
