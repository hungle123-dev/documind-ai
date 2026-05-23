from collections.abc import Iterable
from typing import Protocol

from openai import APIConnectionError, AuthenticationError, OpenAI, PermissionDeniedError, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config.settings import settings


class ChatClient(Protocol):
    def complete(
        self,
        *,
        model: str | list[str],
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Return a complete chat response."""

    def stream(
        self,
        *,
        model: str | list[str],
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Iterable[str]:
        """Yield streamed chat response tokens."""


class GrokClient:
    FALLBACK_EXCEPTIONS = (AuthenticationError, PermissionDeniedError, RateLimitError)

    def __init__(
        self,
        api_key: str | None = None,
        api_keys: list[str] | str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.base_url = base_url or settings.XAI_BASE_URL
        self.api_keys = self._resolve_api_keys(api_key=api_key, api_keys=api_keys)
        if not self.api_keys:
            raise RuntimeError("XAI_API_KEY or XAI_API_KEYS is required for live Grok requests.")

        self.clients = [OpenAI(api_key=key, base_url=self.base_url) for key in self.api_keys]
        # Backward-compatible test seam: tests may replace `client` directly.
        self.client = self.clients[0]

    def _resolve_api_keys(
        self,
        api_key: str | None,
        api_keys: list[str] | str | None,
    ) -> list[str]:
        if api_keys is not None:
            if isinstance(api_keys, str):
                return [key.strip() for key in api_keys.split(",") if key.strip()]
            return [key.strip() for key in api_keys if key.strip()]
        if api_key is not None:
            return [api_key] if api_key else []
        return settings.xai_api_keys

    def _models(self, model: str | list[str]) -> list[str]:
        return model if isinstance(model, list) else [model]

    @retry(
        retry=retry_if_exception_type(APIConnectionError),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
    )
    def _complete_once(
        self,
        client,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Grok returned an empty response.")
        return content

    def complete(
        self,
        *,
        model: str | list[str],
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        errors: list[str] = []
        clients = [self.client] if self.client is not self.clients[0] else self.clients
        for client_index, client in enumerate(clients, start=1):
            for candidate_model in self._models(model):
                try:
                    return self._complete_once(
                        client,
                        model=candidate_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except self.FALLBACK_EXCEPTIONS as exc:
                    errors.append(f"key #{client_index}, model {candidate_model}: {type(exc).__name__}")
                    continue
        raise RuntimeError(f"All configured xAI API keys failed. Attempts: {'; '.join(errors)}")

    @retry(
        retry=retry_if_exception_type(APIConnectionError),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
    )
    def _stream_once(
        self,
        client,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ):
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

    def stream(
        self,
        *,
        model: str | list[str],
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Iterable[str]:
        errors: list[str] = []
        clients = [self.client] if self.client is not self.clients[0] else self.clients
        for client_index, client in enumerate(clients, start=1):
            for candidate_model in self._models(model):
                try:
                    response_stream = self._stream_once(
                        client,
                        model=candidate_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    for chunk in response_stream:
                        token = chunk.choices[0].delta.content or ""
                        if token:
                            yield token
                    return
                except self.FALLBACK_EXCEPTIONS as exc:
                    errors.append(f"key #{client_index}, model {candidate_model}: {type(exc).__name__}")
                    continue
        raise RuntimeError(f"All configured xAI API keys failed. Attempts: {'; '.join(errors)}")
