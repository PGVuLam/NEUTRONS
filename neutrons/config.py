from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str = "NEUTRONS"
    app_version: str = "0.1.0"
    environment: str = os.getenv("NEUTRONS_ENV", "development")
    model_name: str = os.getenv("NEUTRONS_MODEL", "qwen3:4b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    request_timeout_seconds: float = float(os.getenv("NEUTRONS_REQUEST_TIMEOUT", "120"))


settings = Settings()
