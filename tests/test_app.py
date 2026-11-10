from __future__ import annotations

from fastapi.testclient import TestClient

from neutrons.app import app, orchestrator
from neutrons.model_gateway import (
    InvalidModelResponseError,
    ModelInfo,
    ModelTimeoutError,
    ModelUnavailableError,
)
from neutrons.orchestrator import OrchestratorResult

client = TestClient(app)


def test_healthcheck() -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["app"] == "NEUTRONS"


def test_healthcheck_remains_healthy_when_model_provider_is_unavailable(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        assert not args
        assert not kwargs
        raise AssertionError("liveness must not call the model provider")

    monkeypatch.setattr(orchestrator.gateway, "get_model_info", fail_if_called)

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_returns_configured_model_when_available(monkeypatch) -> None:
    received: dict[str, float | None] = {}

    def get_model_info(timeout: float | None = None) -> ModelInfo:
        received["timeout"] = timeout
        return ModelInfo("qwen3:4b", "abc", 100, None)

    monkeypatch.setattr(orchestrator.gateway, "get_model_info", get_model_info)

    response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "app": "NEUTRONS", "model": "qwen3:4b"}
    assert received["timeout"] is not None
    assert 0 < received["timeout"] <= 30


def test_readiness_returns_503_when_configured_model_is_missing(monkeypatch) -> None:
    def missing_model(timeout: float | None = None) -> None:
        assert timeout is not None and timeout > 0
        return None

    monkeypatch.setattr(orchestrator.gateway, "get_model_info", missing_model)

    response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["detail"] == "The configured model was not found in Ollama."


def test_readiness_returns_503_when_provider_is_unreachable(monkeypatch) -> None:
    def fail(timeout: float | None = None):
        assert timeout is not None and timeout > 0
        raise ModelUnavailableError("sensitive provider details")

    monkeypatch.setattr(orchestrator.gateway, "get_model_info", fail)

    response = client.get("/readyz")

    assert response.status_code == 503
    assert "sensitive" not in response.text


def test_chat_route(monkeypatch) -> None:
    def fake_process_user_message(message: str, history=None) -> OrchestratorResult:
        assert message == "Xin chào, NEUTRONS."
        assert history == []
        return OrchestratorResult(response="Xin chào!", model="qwen3:4b")

    monkeypatch.setattr("neutrons.app.orchestrator.process_user_message", fake_process_user_message)
    response = client.post("/api/v1/chat", json={"message": "Xin chào, NEUTRONS."})
    assert response.status_code == 200
    payload = response.json()
    assert payload["response"] == "Xin chào!"
    assert payload["model"] == "qwen3:4b"
    assert payload["status"] == "ok"


def test_chat_route_passes_conversation_history(monkeypatch) -> None:
    received: dict[str, object] = {}

    def fake_process_user_message(message: str, history=None) -> OrchestratorResult:
        received["message"] = message
        received["history"] = history
        return OrchestratorResult(response="Bạn vừa nói xin chào.", model="qwen3:4b")

    monkeypatch.setattr("neutrons.app.orchestrator.process_user_message", fake_process_user_message)
    response = client.post(
        "/api/v1/chat",
        json={
            "message": "Tôi tên là An",
            "history": [{"role": "user", "content": "Xin chào"}, {"role": "assistant", "content": "Chào bạn"}],
        },
    )
    assert response.status_code == 200
    assert received == {
        "message": "Tôi tên là An",
        "history": [{"role": "user", "content": "Xin chào"}, {"role": "assistant", "content": "Chào bạn"}],
    }


def test_chat_route_rejects_empty_message() -> None:
    response = client.post("/api/v1/chat", json={"message": ""})
    assert response.status_code == 422


def test_chat_route_rejects_oversized_total_request() -> None:
    response = client.post(
        "/api/v1/chat",
        json={
            "message": "a" * 20_000,
            "history": [
                {"role": "user", "content": "b" * 20_000},
                {"role": "assistant", "content": "c" * 20_000},
            ],
        },
    )
    assert response.status_code == 422


def test_chat_route_maps_model_timeout_to_gateway_timeout(monkeypatch) -> None:
    def fake_process_user_message(message: str, history=None) -> OrchestratorResult:
        assert message == "Xin chào"
        assert history == []
        raise ModelTimeoutError("do not expose this internal detail")

    monkeypatch.setattr("neutrons.app.orchestrator.process_user_message", fake_process_user_message)

    response = client.post("/api/v1/chat", json={"message": "Xin chào"})

    assert response.status_code == 504
    assert response.json()["detail"] == "Local model did not respond before the configured timeout."
    assert "internal detail" not in response.text


def test_chat_route_does_not_expose_malformed_provider_response(monkeypatch) -> None:
    def fake_process_user_message(message: str, history=None) -> OrchestratorResult:
        assert message == "Xin chào"
        assert history == []
        raise InvalidModelResponseError("sensitive provider payload detail")

    monkeypatch.setattr("neutrons.app.orchestrator.process_user_message", fake_process_user_message)

    response = client.post("/api/v1/chat", json={"message": "Xin chào"})

    assert response.status_code == 502
    assert response.json()["detail"] == "The local model returned an invalid response."
    assert "sensitive" not in response.text
