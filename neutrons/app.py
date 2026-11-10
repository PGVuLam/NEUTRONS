from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from neutrons.config import settings
from neutrons.logging_config import configure_logging
from neutrons.model_gateway import (
    ModelGatewayError,
    ModelNotFoundError,
    ModelTimeoutError,
    ModelUnavailableError,
)
from neutrons.orchestrator import AgentOrchestrator

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info(
        "Application started",
        extra={
            "event": "application_started",
            "context": {"app": _app.title, "model": settings.model_name},
        },
    )
    yield
    logger.info("Application stopped", extra={"event": "application_stopped"})


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=20_000)


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=settings.max_request_chars,
        description="Nội dung tin nhắn từ người dùng",
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=settings.max_history_messages,
    )

    @model_validator(mode="after")
    def validate_request_size(self) -> ChatRequest:
        total_chars = len(self.message) + sum(len(item.content) for item in self.history)
        if total_chars > settings.max_request_chars:
            raise ValueError(
                f"Tin nhắn và lịch sử không được vượt quá {settings.max_request_chars} ký tự"
            )
        if len(self.history) > settings.max_history_messages:
            raise ValueError(
                f"Lịch sử không được vượt quá {settings.max_history_messages} tin nhắn"
            )
        return self


class ChatResponse(BaseModel):
    response: str
    model: str
    status: str = "ok"


orchestrator = AgentOrchestrator()


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    return {
        "status": "ok",
        "app": settings.app_name,
        "model": settings.model_name,
        "environment": settings.environment,
    }


@app.get("/readyz")
def readiness() -> dict[str, str]:
    try:
        model = orchestrator.gateway.get_model_info(timeout=settings.readiness_timeout_seconds)
    except ModelGatewayError as exc:
        logger.warning(
            "Readiness check could not reach the configured model",
            extra={"event": "readiness_provider_unavailable"},
        )
        raise HTTPException(
            status_code=503,
            detail="Local Ollama is unavailable or did not respond to the readiness check.",
        ) from exc

    if model is None:
        logger.warning(
            "Configured model is not installed",
            extra={"event": "readiness_model_missing", "context": {"model": settings.model_name}},
        )
        raise HTTPException(
            status_code=503,
            detail="The configured model was not found in Ollama.",
        )

    return {"status": "ready", "app": settings.app_name, "model": model.name}


@app.get("/")
def root() -> dict[str, str]:
    return {"message": f"{settings.app_name} is running", "version": settings.app_version}


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    started_at = time.monotonic()
    logger.info(
        "Chat request received",
        extra={
            "event": "chat_request_received",
            "context": {"history_count": len(request.history), "message_chars": len(request.message)},
        },
    )
    try:
        history = [message.model_dump() for message in request.history]
        result = orchestrator.process_user_message(request.message, history=history)
    except ValueError as exc:
        logger.warning(
            "Chat request validation failed",
            extra={"event": "chat_request_invalid"},
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelTimeoutError as exc:
        logger.warning("Model request timed out", extra={"event": "chat_request_timeout"})
        raise HTTPException(
            status_code=504,
            detail="Local model did not respond before the configured timeout.",
        ) from exc
    except ModelNotFoundError as exc:
        logger.warning("Configured local model was not found", extra={"event": "model_not_found"})
        raise HTTPException(
            status_code=503,
            detail="The configured model was not found in Ollama. Check the model name with 'ollama list'.",
        ) from exc
    except ModelUnavailableError as exc:
        logger.warning("Local model service is unavailable", extra={"event": "model_unavailable"})
        raise HTTPException(
            status_code=503,
            detail="Local Ollama is unavailable. Check that the Ollama service is running.",
        ) from exc
    except ModelGatewayError as exc:
        logger.error("Local model returned an invalid response", extra={"event": "model_response_invalid"})
        raise HTTPException(
            status_code=502,
            detail="The local model returned an invalid response.",
        ) from exc

    logger.info(
        "Chat request completed",
        extra={
            "event": "chat_request_completed",
            "context": {
                "model": result.model,
                "duration_ms": round((time.monotonic() - started_at) * 1000),
            },
        },
    )
    return ChatResponse(response=result.response, model=result.model, status=result.status)
