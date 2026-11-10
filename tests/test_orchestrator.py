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

    def check_health(self, timeout: float | None = None) -> bool:
        assert timeout is None or timeout > 0
        return True

    def get_model_info(self, timeout: float | None = None):
        assert timeout is None or timeout > 0
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

    assert gateway.history == [{"role": "assistant", "content": "defghijk"}]


def test_history_within_limits_preserves_chronological_order() -> None:
    gateway = FakeGateway()
    orchestrator = AgentOrchestrator(
        gateway=gateway,
        configuration=Settings(max_history_messages=4, max_history_chars=100),
    )
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
    ]

    orchestrator.process_user_message("current", history)

    assert gateway.history == history
    assert gateway.history is not history


def test_history_obeys_message_count_limit() -> None:
    gateway = FakeGateway()
    orchestrator = AgentOrchestrator(
        gateway=gateway,
        configuration=Settings(max_history_messages=2, max_history_chars=100),
    )
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "middle"},
        {"role": "user", "content": "new"},
    ]

    orchestrator.process_user_message("current", history)

    assert gateway.history == history[-2:]


def test_history_includes_only_whole_older_messages_that_fit_budget() -> None:
    gateway = FakeGateway()
    orchestrator = AgentOrchestrator(
        gateway=gateway,
        configuration=Settings(max_history_messages=5, max_history_chars=8),
    )
    history = [
        {"role": "user", "content": "older"},  # 5 chars
        {"role": "assistant", "content": "new"},  # 3 chars
    ]

    orchestrator.process_user_message("current", history)

    assert gateway.history == history


def test_history_does_not_partially_include_older_message() -> None:
    gateway = FakeGateway()
    orchestrator = AgentOrchestrator(
        gateway=gateway,
        configuration=Settings(max_history_messages=5, max_history_chars=7),
    )
    history = [
        {"role": "user", "content": "older"},  # 5 chars
        {"role": "assistant", "content": "new"},  # 3 chars
    ]

    orchestrator.process_user_message("current", history)

    assert gateway.history == [history[-1]]


def test_single_oversized_newest_message_keeps_its_suffix() -> None:
    gateway = FakeGateway()
    orchestrator = AgentOrchestrator(
        gateway=gateway,
        configuration=Settings(max_history_messages=5, max_history_chars=5),
    )
    history = [{"role": "user", "content": "beginning-important-ending"}]

    orchestrator.process_user_message("current", history)

    assert gateway.history == [{"role": "user", "content": "nding"}]
    assert history == [{"role": "user", "content": "beginning-important-ending"}]


def test_newest_message_uses_budget_before_any_older_message() -> None:
    gateway = FakeGateway()
    orchestrator = AgentOrchestrator(
        gateway=gateway,
        configuration=Settings(max_history_messages=5, max_history_chars=10),
    )
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "newest123"},  # 9 chars
    ]

    orchestrator.process_user_message("current", history)

    assert gateway.history == [history[-1]]


def test_zero_history_message_limit_returns_empty_history() -> None:
    gateway = FakeGateway()
    orchestrator = AgentOrchestrator(
        gateway=gateway,
        configuration=Settings(max_history_messages=0, max_history_chars=100),
    )

    orchestrator.process_user_message("current", [{"role": "user", "content": "old"}])

    assert gateway.history == []


def test_zero_history_character_budget_returns_empty_history() -> None:
    gateway = FakeGateway()
    orchestrator = AgentOrchestrator(
        gateway=gateway,
        configuration=Settings(max_history_messages=5, max_history_chars=0),
    )

    orchestrator.process_user_message("current", [{"role": "user", "content": "old"}])

    assert gateway.history == []


def test_history_exact_character_budget_is_included() -> None:
    gateway = FakeGateway()
    orchestrator = AgentOrchestrator(
        gateway=gateway,
        configuration=Settings(max_history_messages=5, max_history_chars=8),
    )
    history = [
        {"role": "user", "content": "four"},
        {"role": "assistant", "content": "more"},
    ]

    orchestrator.process_user_message("current", history)

    assert gateway.history == history


def test_history_selection_does_not_mutate_input() -> None:
    gateway = FakeGateway()
    orchestrator = AgentOrchestrator(
        gateway=gateway,
        configuration=Settings(max_history_messages=5, max_history_chars=5),
    )
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "123456789"},
    ]
    original = [message.copy() for message in history]

    orchestrator.process_user_message("current", history)

    assert history == original


def test_orchestrator_rejects_empty_message() -> None:
    with pytest.raises(ValueError, match="không được rỗng"):
        AgentOrchestrator(gateway=FakeGateway()).process_user_message("  ")
