from __future__ import annotations

from typing import Any, Sequence

import httpx

from neutrons.config import settings


class ModelGatewayError(RuntimeError):
    """Raised when the local model gateway cannot complete the request."""


class OllamaChatGateway:
    """Thin abstraction over the local Ollama HTTP API."""

    def __init__(self, base_url: str | None = None, model_name: str | None = None) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model_name = model_name or settings.model_name
        self.timeout = settings.request_timeout_seconds

    def _build_messages(
        self,
        user_message: str,
        system_prompt: str | None = None,
        history: Sequence[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return messages

    def chat(
        self,
        user_message: str,
        system_prompt: str | None = None,
        history: Sequence[dict[str, str]] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": self._build_messages(user_message, system_prompt, history),
            "think": False,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 512},
        }

        try:
            response = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - depends on runtime service
            raise ModelGatewayError(f"Failed to reach local Ollama at {self.base_url}: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover - protocol-level validation
            raise ModelGatewayError("Received invalid JSON from Ollama") from exc

        message = data.get("message", {})
        content = message.get("content", "") if isinstance(message, dict) else ""
        if isinstance(content, str) and "</think>" in content:
            content = content.rsplit("</think>", maxsplit=1)[-1]
        if not isinstance(content, str) or not content.strip():
            raise ModelGatewayError("Ollama returned no final answer within the generation limit")
        return content.strip()

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
        }

        try:
            response = httpx.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
            response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - depends on runtime service
            raise ModelGatewayError(f"Failed to reach local Ollama at {self.base_url}: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover - protocol-level validation
            raise ModelGatewayError("Received invalid JSON from Ollama") from exc

        content = data.get("response", "")
        if not isinstance(content, str) or not content.strip():
            raise ModelGatewayError("Ollama returned an empty response")
        return content.strip()
