import json
from pathlib import Path

from config.settings import Settings


def test_release_artifacts_exist() -> None:
    for path in [
        ".github/workflows/test.yml",
        ".github/workflows/deploy-space.yml",
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


def test_readme_has_huggingface_spaces_metadata() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert readme.startswith("---\n")
    assert "sdk: gradio" in readme
    assert "app_file: app.py" in readme


def test_huggingface_space_deploy_workflow_uses_token_secret() -> None:
    workflow = Path(".github/workflows/deploy-space.yml").read_text(encoding="utf-8")

    assert "HF_TOKEN" in workflow
    assert "huggingface.co/spaces" in workflow
    assert "requirements.txt" in workflow
    assert "GROQ_API_KEY" not in workflow


def test_dockerignore_excludes_local_runtime_and_model_artifacts() -> None:
    dockerignore = Path(".dockerignore").read_text(encoding="utf-8")

    for pattern in [
        ".venv/",
        ".pytest-run*/",
        ".test-tmp/",
        ".runtime-validation/",
        ".x/",
        "document_cache/",
        "chroma_db/",
        "app_smoke*.log",
    ]:
        assert pattern in dockerignore


def test_dockerfile_installs_certificate_bundle_before_pip() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "ca-certificates" in dockerfile
    assert "PIP_TRUSTED_HOSTS" in dockerfile
    assert "--default-timeout=120" in dockerfile
    assert "--retries=10" in dockerfile
    assert dockerfile.index("ca-certificates") < dockerfile.index("pip install")


def test_compose_sets_local_pip_trusted_hosts_for_docker_desktop_proxy() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "PIP_TRUSTED_HOSTS" in compose
    assert "download.pytorch.org" in compose
    assert "download-r2.pytorch.org" in compose
    assert "files.pythonhosted.org" in compose


def test_documented_model_defaults_match_settings() -> None:
    settings = Settings(_env_file=None, GROQ_API_KEY="gsk-test-key")
    env_example = Path(".env.example").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")

    for variable in [
        "RELEVANCE_MODEL",
        "RESEARCH_MODEL",
        "VERIFICATION_MODEL",
        "RERANKER_TORCH_THREADS",
    ]:
        value = getattr(settings, variable)
        assert f"{variable}={value}" in env_example
        assert f"`{variable}`: default `{value}`." in readme
