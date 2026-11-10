from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from neutrons.config import settings
from neutrons.model_gateway import ModelGatewayError
from neutrons.orchestrator import AgentOrchestrator

app = FastAPI(title=settings.app_name, version=settings.app_version)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=20_000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=20_000, description="Nội dung tin nhắn từ người dùng")
    history: list[ChatMessage] = Field(default_factory=list, max_length=40)


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


@app.get("/")
def root() -> dict[str, str]:
    return {"message": f"{settings.app_name} is running", "version": settings.app_version}


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        history = [message.model_dump() for message in request.history]
        result = orchestrator.process_user_message(request.message, history=history)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelGatewayError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ChatResponse(response=result.response, model=result.model, status=result.status)
