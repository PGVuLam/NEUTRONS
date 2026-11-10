from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Settings:
    app_name: str = "NEUTRONS"
    app_version: str = "0.1.0"
    environment: str = "development"
    model_name: str = "qwen3:4b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    request_timeout_seconds: float = 120.0
    readiness_timeout_seconds: float = 3.0
    temperature: float = 0.2
    thinking_mode: str = "auto"
    num_predict: int = 512
    num_ctx: int = 4096
    api_port: int = 8000
    max_history_messages: int = 40
    max_history_chars: int = 12_000
    max_request_chars: int = 50_000
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("NEUTRONS_MODEL must not be empty")
        if not math.isfinite(self.request_timeout_seconds) or self.request_timeout_seconds <= 0:
            raise ValueError("NEUTRONS_REQUEST_TIMEOUT must be a positive finite number")
        if (
            not math.isfinite(self.readiness_timeout_seconds)
            or not 0 < self.readiness_timeout_seconds <= 30
        ):
            raise ValueError("NEUTRONS_READINESS_TIMEOUT must be greater than 0 and at most 30 seconds")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("NEUTRONS_TEMPERATURE must be between 0 and 2")
        if self.thinking_mode not in {"auto", "enabled", "disabled"}:
            raise ValueError("NEUTRONS_THINKING must be auto, enabled, or disabled")
        if self.num_predict < 1:
            raise ValueError("NEUTRONS_NUM_PREDICT must be at least 1")
        if self.num_ctx < 1:
            raise ValueError("NEUTRONS_NUM_CTX must be at least 1")
        if not 1 <= self.api_port <= 65535:
            raise ValueError("NEUTRONS_API_PORT must be between 1 and 65535")
        if self.max_history_messages < 0:
            raise ValueError("NEUTRONS_MAX_HISTORY_MESSAGES must not be negative")
        if self.max_history_chars < 0:
            raise ValueError("NEUTRONS_MAX_HISTORY_CHARS must not be negative")
        if self.max_request_chars < 1:
            raise ValueError("NEUTRONS_MAX_REQUEST_CHARS must be at least 1")
        if self.log_level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("NEUTRONS_LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")

        try:
            parsed_url = urlsplit(self.ollama_base_url)
            port = parsed_url.port
        except ValueError as exc:
            raise ValueError("OLLAMA_BASE_URL must contain a valid port number") from exc
        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.hostname
            or parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise ValueError("OLLAMA_BASE_URL must be an HTTP(S) URL without credentials or query parameters")
        if port is not None and not 1 <= port <= 65535:
            raise ValueError("OLLAMA_BASE_URL port must be between 1 and 65535")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        values = os.environ if environ is None else environ

        def integer(name: str, default: int) -> int:
            try:
                return int(values.get(name, str(default)))
            except ValueError as exc:
                raise ValueError(f"{name} must be an integer") from exc

        def decimal(name: str, default: float) -> float:
            try:
                return float(values.get(name, str(default)))
            except ValueError as exc:
                raise ValueError(f"{name} must be a number") from exc

        return cls(
            environment=values.get("NEUTRONS_ENV", "development"),
            model_name=values.get("NEUTRONS_MODEL", "qwen3:4b"),
            ollama_base_url=values.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            request_timeout_seconds=decimal("NEUTRONS_REQUEST_TIMEOUT", 120.0),
            readiness_timeout_seconds=decimal("NEUTRONS_READINESS_TIMEOUT", 3.0),
            temperature=decimal("NEUTRONS_TEMPERATURE", 0.2),
            thinking_mode=values.get("NEUTRONS_THINKING", "auto").casefold(),
            num_predict=integer("NEUTRONS_NUM_PREDICT", 512),
            num_ctx=integer("NEUTRONS_NUM_CTX", 4096),
            api_port=integer("NEUTRONS_API_PORT", 8000),
            max_history_messages=integer("NEUTRONS_MAX_HISTORY_MESSAGES", 40),
            max_history_chars=integer("NEUTRONS_MAX_HISTORY_CHARS", 12_000),
            max_request_chars=integer("NEUTRONS_MAX_REQUEST_CHARS", 50_000),
            log_level=values.get("NEUTRONS_LOG_LEVEL", "INFO").upper(),
        )


settings = Settings.from_env()
