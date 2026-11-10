from __future__ import annotations

from collections.abc import Mapping, Sequence

from neutrons.agent.router import IntentRouter
from neutrons.agent.schemas import Intent


class RouterGateway:
    model_name = "test-model"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def chat(
        self,
        user_message: str,
        system_prompt: str | None = None,
        history: Sequence[Mapping[str, str]] | None = None,
    ) -> str:
        self.calls += 1
        assert user_message
        assert "Do not plan or execute actions" in system_prompt
        return self.response

    def check_health(self, timeout: float | None = None) -> bool:
        return True

    def get_model_info(self, timeout: float | None = None):
        return None


def test_router_returns_valid_chat_intent_and_response() -> None:
    gateway = RouterGateway(
        '{"intent":"chat","confidence":0.95,"reason":"Greeting","response":"Xin chào!"}'
    )

    decision = IntentRouter(gateway).route("Xin chào")

    assert decision.intent is Intent.CHAT
    assert decision.response == "Xin chào!"
    assert gateway.calls == 1


def test_router_distinguishes_task_intent_without_execution() -> None:
    gateway = RouterGateway(
        '{"intent":"task","confidence":0.9,"reason":"Requires multiple steps","response":null}'
    )

    decision = IntentRouter(gateway).route("Hãy xử lý việc này")

    assert decision.intent is Intent.TASK
    assert decision.response is None


def test_router_classifies_unsupported_requests_without_actions() -> None:
    gateway = RouterGateway(
        '{"intent":"unsupported","confidence":0.95,"reason":"Unavailable capability","response":null}'
    )

    decision = IntentRouter(gateway).route("Open a browser")

    assert decision.intent is Intent.UNSUPPORTED


def test_router_falls_back_to_legacy_chat_for_malformed_output() -> None:
    decision = IntentRouter(RouterGateway("not json")).route("Do something")

    assert decision.intent is Intent.CHAT
    assert decision.response is None


def test_router_falls_back_to_clarification_for_low_confidence() -> None:
    gateway = RouterGateway(
        '{"intent":"task","confidence":0.2,"reason":"Unsure","response":null}'
    )

    decision = IntentRouter(gateway).route("Maybe do this")

    assert decision.intent is Intent.CLARIFICATION
