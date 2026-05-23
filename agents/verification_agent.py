from langchain_core.documents import Document

from config.settings import settings
from utils.language import LANGUAGE_INSTRUCTION
from utils.logging import logger

from .llm_client import ChatClient, GroqClient
from .research_agent import build_context


VERIFICATION_PROMPT = (
    "You are a fact-checking agent. Verify the draft answer against the source context.\n"
    "Respond in this EXACT format:\n"
    "Supported: YES/NO\n"
    "Unsupported Claims: [list or None]\n"
    "Contradictions: [list or None]\n"
    "Relevant: YES/NO\n"
    "Additional Details: [optional]\n"
    f"{LANGUAGE_INSTRUCTION}"
)


class VerificationAgent:
    EXPECTED_KEYS = {
        "Supported",
        "Unsupported Claims",
        "Contradictions",
        "Relevant",
        "Additional Details",
    }

    def __init__(self, client: ChatClient | None = None, model: str | None = None) -> None:
        self.client = client or GroqClient()
        self.model = model or settings.VERIFICATION_MODEL

    def check(self, answer: str, documents: list[Document]) -> dict[str, str]:
        logger.debug(f"VerificationAgent.check | answer_len={len(answer)} | docs={len(documents)}")
        context = build_context(documents)
        response = self.client.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": VERIFICATION_PROMPT},
                {"role": "user", "content": f"Answer: {answer}\n\nContext:\n{context}"},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        parsed = self.parse_verification_response(response)
        return {
            "verification_report": self.format_verification_report(parsed),
            "context_used": context,
        }

    def parse_verification_response(self, response_text: str) -> dict[str, object]:
        parsed: dict[str, object] = {}
        aliases = {
            "unsupported claims": "Unsupported Claims",
            "additional details": "Additional Details",
        }

        for line in response_text.splitlines():
            if ":" not in line:
                continue
            raw_key, raw_value = line.split(":", 1)
            normalized_key = aliases.get(raw_key.strip().lower(), raw_key.strip().title())
            if normalized_key not in self.EXPECTED_KEYS:
                continue
            value = raw_value.strip()
            if normalized_key in {"Unsupported Claims", "Contradictions"}:
                parsed[normalized_key] = self._parse_list(value)
            elif normalized_key in {"Supported", "Relevant"}:
                parsed[normalized_key] = "YES" if value.upper().startswith("YES") else "NO"
            else:
                parsed[normalized_key] = value

        parsed.setdefault("Supported", "NO")
        parsed.setdefault("Unsupported Claims", [])
        parsed.setdefault("Contradictions", [])
        parsed.setdefault("Relevant", "NO")
        parsed.setdefault("Additional Details", "")
        return parsed

    def _parse_list(self, value: str) -> list[str]:
        if not value or value.lower() == "none":
            return []
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if not inner or inner.lower() == "none":
                return []
            return [item.strip().strip("\"'") for item in inner.split(",") if item.strip()]
        return [value]

    def format_verification_report(self, verification: dict[str, object]) -> str:
        unsupported = verification.get("Unsupported Claims") or []
        contradictions = verification.get("Contradictions") or []
        return (
            f"Supported: {verification.get('Supported', 'NO')}\n"
            f"Unsupported Claims: {', '.join(unsupported) if unsupported else 'None'}\n"
            f"Contradictions: {', '.join(contradictions) if contradictions else 'None'}\n"
            f"Relevant: {verification.get('Relevant', 'NO')}\n"
            f"Additional Details: {verification.get('Additional Details') or 'None'}"
        )
