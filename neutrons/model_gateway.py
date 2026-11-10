from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import httpx

from neutrons.config import Settings, settings

logger = logging.getLogger(__name__)
_UNRESOLVED = object()


class ModelGatewayError(RuntimeError):
    """Base error for model-provider failures."""


class ModelUnavailableError(ModelGatewayError):
    """Raised when the configured model service cannot be used."""


class ModelNotFoundError(ModelUnavailableError):
    """Raised when the configured model is not installed in Ollama."""


class ModelEndpointError(ModelGatewayError):
    """Raised when Ollama reports a missing or invalid API endpoint."""


class ModelTimeoutError(ModelGatewayError):
    """Raised when Ollama does not respond before the configured timeout."""


class InvalidModelResponseError(ModelGatewayError):
    """Raised when Ollama returns an invalid or incomplete response."""


class EmptyModelResponseError(InvalidModelResponseError):
    """Raised when the model did not produce user-facing text."""


class ModelCapabilityError(ModelGatewayError):
    """Raised when a configured model does not support a requested capability."""


@dataclass(frozen=True, slots=True)
class ModelInfo:
    name: str
    digest: str | None
    size_bytes: int | None
    modified_at: str | None


class ChatModelGateway(Protocol):
    model_name: str

    def chat(
        self,
        user_message: str,
        system_prompt: str | None = None,
        history: Sequence[Mapping[str, str]] | None = None,
    ) -> str: ...

    def chat_json(
        self,
        user_message: str,
        system_prompt: str,
        history: Sequence[Mapping[str, str]] | None = None,
    ) -> str: ...

    def check_health(self, timeout: float | None = None) -> bool: ...

    def get_model_info(self, timeout: float | None = None) -> ModelInfo | None: ...


class OllamaChatGateway:
    """Provider-specific HTTP adapter for the local Ollama API."""

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        configuration: Settings = settings,
    ) -> None:
        self.base_url = (base_url or configuration.ollama_base_url).rstrip("/")
        self.model_name = model_name or configuration.model_name
        self.configuration = configuration
        self.timeout = configuration.request_timeout_seconds
        self._thinking_value: bool | None | object = _UNRESOLVED
        self._thinking_lock = threading.Lock()

    def _resolve_thinking_value(self) -> bool | None:
        with self._thinking_lock:
            if self._thinking_value is not _UNRESOLVED:
                if self._thinking_value is None or isinstance(self._thinking_value, bool):
                    return self._thinking_value
                raise RuntimeError("Cached Ollama thinking capability is invalid")

            data = self._request_json(
                "POST", "api/show", {"model": self.model_name}
            )
            thinking = data.get("thinking")
            if thinking is None:
                values: list[object] = []
            elif isinstance(thinking, dict) and isinstance(thinking.get("values"), list):
                values = thinking["values"]
            else:
                raise InvalidModelResponseError("Ollama thinking capability metadata is malformed")

            if any(not isinstance(value, (bool, str)) for value in values):
                raise InvalidModelResponseError("Ollama thinking capability values are malformed")
            mode = self.configuration.thinking_mode
            if mode == "disabled":
                if False not in values:
                    raise ModelCapabilityError(
                        "The configured model does not support disabling thinking"
                    )
                resolved: bool | None = False
            elif mode == "enabled":
                if True not in values:
                    raise ModelCapabilityError(
                        "The configured model does not support enabling thinking"
                    )
                resolved = True
            elif False in values:
                resolved = False
            elif True in values:
                resolved = True
            else:
                resolved = None

            self._thinking_value = resolved
            return resolved

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        request_timeout = self.timeout if timeout is None else timeout
        try:
            if method == "POST":
                response = httpx.post(url, json=payload, timeout=request_timeout)
            elif method == "GET":
                response = httpx.get(url, timeout=request_timeout)
            else:
                raise ValueError(f"Unsupported Ollama HTTP method: {method}")
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            logger.warning(
                "Ollama request timed out",
                extra={"event": "model_request_timeout", "context": {"model": self.model_name}},
            )
            raise ModelTimeoutError("Ollama did not respond before the configured timeout") from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                try:
                    error_body = exc.response.json()
                except ValueError:
                    error_body = None
                provider_error = error_body.get("error") if isinstance(error_body, dict) else None
                if (
                    path.lstrip("/") in {"api/chat", "api/generate"}
                    and isinstance(provider_error, str)
                    and "model" in provider_error.casefold()
                    and any(
                        marker in provider_error.casefold()
                        for marker in ("not found", "does not exist", "not available")
                    )
                ):
                    raise ModelNotFoundError("The configured Ollama model was not found") from exc
                raise ModelEndpointError("Ollama returned 404 for an API endpoint") from exc
            raise ModelUnavailableError(
                f"Ollama returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            logger.warning(
                "Ollama service is unreachable",
                extra={"event": "model_service_unreachable", "context": {"model": self.model_name}},
            )
            raise ModelUnavailableError("Could not connect to the configured Ollama service") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise InvalidModelResponseError("Ollama returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise InvalidModelResponseError("Ollama returned an unexpected response type")
        if "error" in data:
            raise ModelUnavailableError("Ollama reported an error while processing the request")
        return data

    def _build_messages(
        self,
        user_message: str,
        system_prompt: str | None = None,
        history: Sequence[Mapping[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(
                {"role": message["role"], "content": message["content"]}
                for message in history
            )
        messages.append({"role": "user", "content": user_message})
        return messages

    def chat(
        self,
        user_message: str,
        system_prompt: str | None = None,
        history: Sequence[Mapping[str, str]] | None = None,
    ) -> str:
        return self._chat(user_message, system_prompt, history)

    def chat_json(
        self,
        user_message: str,
        system_prompt: str,
        history: Sequence[Mapping[str, str]] | None = None,
    ) -> str:
        return self._chat(user_message, system_prompt, history, json_mode=True)

    def _chat(
        self,
        user_message: str,
        system_prompt: str | None,
        history: Sequence[Mapping[str, str]] | None,
        *,
        json_mode: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": self._build_messages(user_message, system_prompt, history),
            "stream": False,
            "options": {
                "temperature": self.configuration.temperature,
                "num_predict": self.configuration.num_predict,
                "num_ctx": self.configuration.num_ctx,
            },
        }
        thinking_value = self._resolve_thinking_value()
        if thinking_value is not None:
            payload["think"] = thinking_value
        if json_mode:
            payload["format"] = "json"
        data = self._request_json("POST", "api/chat", payload)
        message = data.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise InvalidModelResponseError("Ollama returned a malformed assistant message")
        content = message.get("content")
        if not isinstance(content, str):
            raise InvalidModelResponseError("Ollama assistant content must be text")
        thinking_field = message.get("thinking", "")
        if not isinstance(thinking_field, str):
            raise InvalidModelResponseError("Ollama assistant thinking field must be text")
        if "<think>" in content or "</think>" in content:
            raise InvalidModelResponseError("Ollama mixed reasoning markers into assistant content")
        if not content.strip():
            raise EmptyModelResponseError("Ollama returned no final answer")
        self._validate_completion(data)
        return content.strip()

    @staticmethod
    def _validate_completion(data: Mapping[str, Any]) -> None:
        if data.get("done") is not True:
            raise InvalidModelResponseError("Ollama response did not complete generation")
        done_reason = data.get("done_reason")
        if done_reason is not None and done_reason != "stop":
            raise InvalidModelResponseError("Ollama generation ended before a normal stop")

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": self.configuration.temperature,
                "num_predict": self.configuration.num_predict,
                "num_ctx": self.configuration.num_ctx,
            },
        }
        thinking_value = self._resolve_thinking_value()
        if thinking_value is not None:
            payload["think"] = thinking_value
        data = self._request_json("POST", "api/generate", payload)
        content = data.get("response")
        if not isinstance(content, str):
            raise InvalidModelResponseError("Ollama generated response must be text")
        thinking_field = data.get("thinking", "")
        if not isinstance(thinking_field, str):
            raise InvalidModelResponseError("Ollama generated thinking field must be text")
        if "<think>" in content or "</think>" in content:
            raise InvalidModelResponseError("Ollama mixed reasoning markers into generated content")
        if not content.strip():
            raise EmptyModelResponseError("Ollama returned an empty generated response")
        self._validate_completion(data)
        return content.strip()

    def check_health(self, timeout: float | None = None) -> bool:
        data = self._request_json("GET", "api/tags", timeout=timeout)
        if not isinstance(data.get("models"), list):
            raise InvalidModelResponseError("Ollama model-list response is malformed")
        return True

    def get_model_info(self, timeout: float | None = None) -> ModelInfo | None:
        data = self._request_json("GET", "api/tags", timeout=timeout)
        models = data.get("models")
        if not isinstance(models, list):
            raise InvalidModelResponseError("Ollama model-list response is malformed")

        for item in models:
            if not isinstance(item, dict):
                raise InvalidModelResponseError("Ollama model-list contains an invalid entry")
            name = item.get("name")
            if not isinstance(name, str):
                raise InvalidModelResponseError("Ollama model-list entry is missing its name")
            if name == self.model_name:
                digest = item.get("digest")
                size = item.get("size")
                modified_at = item.get("modified_at")
                if digest is not None and not isinstance(digest, str):
                    raise InvalidModelResponseError("Ollama model digest must be text")
                if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 0):
                    raise InvalidModelResponseError("Ollama model size must be a non-negative integer")
                if modified_at is not None and not isinstance(modified_at, str):
                    raise InvalidModelResponseError("Ollama model timestamp must be text")
                return ModelInfo(name, digest, size, modified_at)
        return None
