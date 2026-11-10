from __future__ import annotations

from fastapi.testclient import TestClient

from neutrons.app import app
from neutrons.orchestrator import OrchestratorResult

client = TestClient(app)


def test_healthcheck() -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["app"] == "NEUTRONS"


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
