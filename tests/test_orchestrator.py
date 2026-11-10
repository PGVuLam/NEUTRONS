from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from neutrons.config import Settings
from neutrons.model_gateway import ChatModelGateway
from neutrons.orchestrator import AgentOrchestrator


class FakeGateway:
    model_name = "test-model"

    def __init__(self) -> None:
        self.history: Sequence[Mapping[str, str]] | None = None

    def chat(
        self,
        user_message: str,
        system_prompt: str | None = None,
        history: Sequence[Mapping[str, str]] | None = None,
    ) -> str:
        assert user_message
        assert system_prompt
        self.history = history
        return "Xin chào!"

    def check_health(self) -> bool:
        return True

    def get_model_info(self):
        return None


def test_orchestrator_uses_injected_gateway() -> None:
    gateway: ChatModelGateway = FakeGateway()

    result = AgentOrchestrator(gateway=gateway).process_user_message("Xin chào")

    assert result.response == "Xin chào!"
    assert result.model == "test-model"


def test_orchestrator_limits_history_by_message_count_and_characters() -> None:
    gateway = FakeGateway()
    configuration = Settings(max_history_messages=3, max_history_chars=8)
    orchestrator = AgentOrchestrator(gateway=gateway, configuration=configuration)

    orchestrator.process_user_message(
        "Câu hỏi mới",
        history=[
            {"role": "user", "content": "cũ nhất"},
            {"role": "assistant", "content": "cũ"},
            {"role": "user", "content": "mới"},
            {"role": "assistant", "content": "abcdefghijk"},
        ],
    )

    assert gateway.history == [{"role": "assistant", "content": "abcdefgh"}]


def test_orchestrator_rejects_empty_message() -> None:
    with pytest.raises(ValueError, match="không được rỗng"):
        AgentOrchestrator(gateway=FakeGateway()).process_user_message("  ")
