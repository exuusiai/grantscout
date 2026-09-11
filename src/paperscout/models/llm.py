import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from paperscout.models.endpoints import normalize_local_base_url


class ModelClientError(RuntimeError):
    """Raised when an OpenAI-compatible model service cannot complete a request."""


def parse_json_content(content: str) -> Any:
    """Parse JSON from a model response, tolerating thinking text and code fences."""
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
            return value
        except json.JSONDecodeError:
            continue
    raise ModelClientError("Model response did not contain a valid JSON object or array")


@dataclass(frozen=True)
class ChatResponse:
    content: str
    model: str | None
    usage: dict[str, Any]


class OpenAICompatibleClient:
    """Small synchronous client for local OpenAI-compatible inference servers."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.base_url = normalize_local_base_url(base_url)
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def healthcheck(self) -> dict[str, Any]:
        try:
            response = httpx.get(
                f"{self.base_url}/models",
                headers=self._headers(),
                timeout=self.timeout_seconds,
                follow_redirects=False,
            )
            response.raise_for_status()
            payload = response.json()
            return {"ok": True, "models": payload.get("data", [])}
        except (httpx.HTTPError, ValueError) as error:
            return {"ok": False, "error": str(error)}

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> ChatResponse:
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if chat_template_kwargs is not None:
            body["chat_template_kwargs"] = chat_template_kwargs
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=body,
                timeout=self.timeout_seconds,
                follow_redirects=False,
            )
            response.raise_for_status()
            payload = response.json()
            choice = payload.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content")
            if not isinstance(content, str):
                raise ModelClientError("Model response did not contain message content")
            return ChatResponse(
                content=content,
                model=payload.get("model"),
                usage=payload.get("usage") or {},
            )
        except httpx.HTTPStatusError as error:
            detail = error.response.text[:500]
            raise ModelClientError(
                f"Model request failed with HTTP {error.response.status_code}: {detail}"
            ) from error
        except (httpx.HTTPError, ValueError) as error:
            raise ModelClientError(f"Model request failed: {error}") from error

    def chat_json(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> Any:
        """Call the local chat endpoint and parse a JSON response safely."""
        response = self.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            chat_template_kwargs={"enable_thinking": False},
        )
        return parse_json_content(response.content)
