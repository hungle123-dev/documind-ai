from types import SimpleNamespace

import pytest
from openai import RateLimitError

from agents.llm_client import GroqClient


class FakeCompletions:
    def __init__(self) -> None:
        self.models: list[str] = []

    def create(self, **kwargs):
        self.models.append(kwargs["model"])
        if kwargs["model"] == "llama-3.3-70b-versatile":
            raise RateLimitError(
                "quota exceeded",
                response=SimpleNamespace(status_code=429, headers={}, request=None),
                body=None,
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="fallback answer"),
                )
            ]
        )


class FakeOpenAI:
    def __init__(self) -> None:
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def test_groq_client_falls_back_to_second_model_on_rate_limit() -> None:
    fake = FakeOpenAI()
    client = GroqClient(api_key="gsk-test")
    client.client = fake

    result = client.complete(
        model=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.1,
        max_tokens=20,
    )

    assert result == "fallback answer"
    assert fake.completions.models == ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]


class KeyAwareCompletions:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.calls: list[str] = []

    def create(self, **kwargs):
        self.calls.append(self.api_key)
        if self.api_key == "gsk-first":
            raise RateLimitError(
                "quota exceeded",
                response=SimpleNamespace(status_code=429, headers={}, request=None),
                body=None,
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=f"answer from {self.api_key}"),
                )
            ]
        )


class KeyAwareOpenAI:
    instances: list["KeyAwareOpenAI"] = []

    def __init__(self, *, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.completions = KeyAwareCompletions(api_key)
        self.chat = SimpleNamespace(completions=self.completions)
        self.instances.append(self)


def test_groq_client_falls_back_to_second_api_key_on_rate_limit(monkeypatch) -> None:
    KeyAwareOpenAI.instances = []
    monkeypatch.setattr("agents.llm_client.OpenAI", KeyAwareOpenAI)

    client = GroqClient(api_keys=["gsk-first", "gsk-second"], base_url="https://api.groq.com/openai/v1")

    result = client.complete(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.1,
        max_tokens=20,
    )

    assert result == "answer from gsk-second"
    assert [instance.api_key for instance in KeyAwareOpenAI.instances] == [
        "gsk-first",
        "gsk-second",
    ]
