from types import SimpleNamespace

import pytest
from openai import RateLimitError

from agents.llm_client import GrokClient


class FakeCompletions:
    def __init__(self) -> None:
        self.models: list[str] = []

    def create(self, **kwargs):
        self.models.append(kwargs["model"])
        if kwargs["model"] == "grok-3":
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


def test_grok_client_falls_back_to_second_model_on_rate_limit() -> None:
    fake = FakeOpenAI()
    client = GrokClient(api_key="xai-test")
    client.client = fake

    result = client.complete(
        model=["grok-3", "grok-3-mini"],
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.1,
        max_tokens=20,
    )

    assert result == "fallback answer"
    assert fake.completions.models == ["grok-3", "grok-3-mini"]


class KeyAwareCompletions:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.calls: list[str] = []

    def create(self, **kwargs):
        self.calls.append(self.api_key)
        if self.api_key == "xai-first":
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


def test_grok_client_falls_back_to_second_api_key_on_rate_limit(monkeypatch) -> None:
    KeyAwareOpenAI.instances = []
    monkeypatch.setattr("agents.llm_client.OpenAI", KeyAwareOpenAI)

    client = GrokClient(api_keys=["xai-first", "xai-second"], base_url="https://api.x.ai/v1")

    result = client.complete(
        model="grok-3-mini",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.1,
        max_tokens=20,
    )

    assert result == "answer from xai-second"
    assert [instance.api_key for instance in KeyAwareOpenAI.instances] == [
        "xai-first",
        "xai-second",
    ]
