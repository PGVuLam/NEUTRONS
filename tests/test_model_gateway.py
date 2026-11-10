from __future__ import annotations

import httpx
import pytest

from neutrons.model_gateway import ModelGatewayError, OllamaChatGateway


def test_chat_uses_bounded_generation_and_returns_final_answer(monkeypatch) -> None:
    request_data: dict[str, object] = {}

    def fake_post(url: str, *, json: dict[str, object], timeout: float) -> httpx.Response:
        request_data.update(json)
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "The user said hello. I should greet them.</think>\n\nXin chào!",
                }
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("neutrons.model_gateway.httpx.post", fake_post)
    gateway = OllamaChatGateway()

    response = gateway.chat("Xin chào")

    assert response == "Xin chào!"
    assert request_data["think"] is False
    assert request_data["options"] == {"temperature": 0.2, "num_predict": 512}


def test_chat_rejects_response_with_no_final_answer(monkeypatch) -> None:
    def fake_post(url: str, *, json: dict[str, object], timeout: float) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "The user said hello.</think>"}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("neutrons.model_gateway.httpx.post", fake_post)
    gateway = OllamaChatGateway()

    with pytest.raises(ModelGatewayError, match="no final answer"):
        gateway.chat("Xin chào")
