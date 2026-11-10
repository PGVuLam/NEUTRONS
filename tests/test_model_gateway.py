from __future__ import annotations

import httpx
import pytest

from neutrons.config import Settings
from neutrons.model_gateway import (
    EmptyModelResponseError,
    InvalidModelResponseError,
    ModelInfo,
    ModelNotFoundError,
    ModelTimeoutError,
    OllamaChatGateway,
)


def test_chat_uses_bounded_generation_and_returns_final_answer(monkeypatch) -> None:
    request_data: dict[str, object] = {}

    def fake_post(url: str, *, json: dict[str, object], timeout: float) -> httpx.Response:
        assert timeout > 0
        request_data.update(json)
        return httpx.Response(
            200,
            json={
                "done": True,
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
    assert request_data["options"] == {
        "temperature": 0.2,
        "num_predict": 512,
        "num_ctx": 4096,
    }


def test_chat_rejects_response_with_no_final_answer(monkeypatch) -> None:
    def fake_post(url: str, *, json: dict[str, object], timeout: float) -> httpx.Response:
        assert json["model"] == "qwen3:4b"
        assert timeout > 0
        return httpx.Response(
            200,
            json={
                "done": True,
                "message": {"role": "assistant", "content": "The user said hello.</think>"},
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("neutrons.model_gateway.httpx.post", fake_post)
    gateway = OllamaChatGateway()

    with pytest.raises(EmptyModelResponseError, match="no final answer"):
        gateway.chat("Xin chào")


@pytest.mark.parametrize("body", [[], {"message": []}, {"message": {"role": "user", "content": "no"}}])
def test_chat_rejects_malformed_response(monkeypatch, body) -> None:
    def fake_post(url: str, *, json: dict[str, object], timeout: float) -> httpx.Response:
        assert json["model"] == "qwen3:4b"
        assert timeout > 0
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr("neutrons.model_gateway.httpx.post", fake_post)
    with pytest.raises(InvalidModelResponseError):
        OllamaChatGateway().chat("Xin chào")


def test_chat_rejects_incomplete_generation(monkeypatch) -> None:
    def fake_post(url: str, *, json: dict[str, object], timeout: float) -> httpx.Response:
        assert json["model"] == "qwen3:4b"
        assert timeout > 0
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "partial"}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("neutrons.model_gateway.httpx.post", fake_post)

    with pytest.raises(InvalidModelResponseError, match="did not complete"):
        OllamaChatGateway().chat("Xin chào")


def test_chat_maps_read_timeout_to_model_timeout(monkeypatch) -> None:
    def fake_post(url: str, *, json: dict[str, object], timeout: float) -> httpx.Response:
        assert json["model"] == "qwen3:4b"
        assert timeout > 0
        request = httpx.Request("POST", url)
        raise httpx.ReadTimeout("internal detail", request=request)

    monkeypatch.setattr("neutrons.model_gateway.httpx.post", fake_post)
    with pytest.raises(ModelTimeoutError, match="configured timeout"):
        OllamaChatGateway().chat("Xin chào")


def test_chat_uses_configured_generation_parameters(monkeypatch) -> None:
    configuration = Settings(temperature=0.5, num_predict=128, num_ctx=2048)
    request_data: dict[str, object] = {}

    def fake_post(url: str, *, json: dict[str, object], timeout: float) -> httpx.Response:
        assert timeout > 0
        request_data.update(json)
        return httpx.Response(
            200,
            json={"done": True, "message": {"role": "assistant", "content": "Xin chào!"}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("neutrons.model_gateway.httpx.post", fake_post)
    OllamaChatGateway(configuration=configuration).chat("Xin chào")

    assert request_data["options"] == {
        "temperature": 0.5,
        "num_predict": 128,
        "num_ctx": 2048,
    }


def test_get_model_info_returns_configured_model(monkeypatch) -> None:
    def fake_get(url: str, *, timeout: float) -> httpx.Response:
        assert timeout > 0
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "qwen3:4b",
                        "digest": "abc",
                        "size": 123,
                        "modified_at": "2026-11-10T00:00:00Z",
                    }
                ]
            },
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr("neutrons.model_gateway.httpx.get", fake_get)

    assert OllamaChatGateway().get_model_info() == ModelInfo(
        "qwen3:4b", "abc", 123, "2026-11-10T00:00:00Z"
    )


def test_missing_model_returns_none_from_model_lookup(monkeypatch) -> None:
    def fake_get(url: str, *, timeout: float) -> httpx.Response:
        assert timeout > 0
        return httpx.Response(200, json={"models": []}, request=httpx.Request("GET", url))

    monkeypatch.setattr("neutrons.model_gateway.httpx.get", fake_get)

    assert OllamaChatGateway().get_model_info() is None


def test_model_lookup_rejects_malformed_model_list(monkeypatch) -> None:
    def fake_get(url: str, *, timeout: float) -> httpx.Response:
        assert timeout > 0
        return httpx.Response(200, json={"models": [None]}, request=httpx.Request("GET", url))

    monkeypatch.setattr("neutrons.model_gateway.httpx.get", fake_get)

    with pytest.raises(InvalidModelResponseError):
        OllamaChatGateway().get_model_info()


def test_not_found_http_status_maps_to_model_not_found(monkeypatch) -> None:
    def fake_post(url: str, *, json: dict[str, object], timeout: float) -> httpx.Response:
        assert json["model"] == "qwen3:4b"
        assert timeout > 0
        return httpx.Response(404, request=httpx.Request("POST", url))

    monkeypatch.setattr("neutrons.model_gateway.httpx.post", fake_post)

    with pytest.raises(ModelNotFoundError):
        OllamaChatGateway().chat("Xin chào")
