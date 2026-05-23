from collections.abc import Iterable
from typing import Protocol

from openai import APIConnectionError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config.settings import settings


class ChatClient(Protocol):
    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Return a complete chat response."""

    def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Iterable[str]:
        """Yield streamed chat response tokens."""


class GrokClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        key = api_key if api_key is not None else settings.XAI_API_KEY
        if not key:
            raise RuntimeError("XAI_API_KEY is required for live Grok requests.")
        self.client = OpenAI(api_key=key, base_url=base_url or settings.XAI_BASE_URL)

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
    )
    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Grok returned an empty response.")
        return content

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
    )
    def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Iterable[str]:
        response_stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in response_stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                yield token
