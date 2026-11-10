from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

from neutrons.agent.core import AgentCore
from neutrons.agent.executor import ActionRegistry, PermissionPolicy
from neutrons.agent.response_policy import build_system_prompt
from neutrons.agent.schemas import TaskExecutionResult
from neutrons.config import Settings, settings
from neutrons.model_gateway import ChatModelGateway, ModelGatewayError, OllamaChatGateway

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OrchestratorResult:
    response: str
    model: str
    status: str = "ok"
    task_result: TaskExecutionResult | None = None


class AgentOrchestrator:
    """Coordinates user intent with the local model gateway."""

    def __init__(
        self,
        gateway: ChatModelGateway | None = None,
        configuration: Settings = settings,
        action_registry: ActionRegistry | None = None,
        permission_policy: PermissionPolicy | None = None,
    ) -> None:
        self.gateway = gateway if gateway is not None else OllamaChatGateway(configuration=configuration)
        self.configuration = configuration
        self.agent_core = AgentCore(
            self.gateway,
            registry=action_registry,
            permission_policy=permission_policy,
            chat_system_prompt=self.build_system_prompt(),
        )

    def build_system_prompt(self) -> str:
        return build_system_prompt(
            "Bạn là NEUTRONS, một trợ lý AI cá nhân chạy cục bộ. "
            "Hãy trả lời rõ ràng, ngắn gọn, có cấu trúc, và ưu tiên an toàn, "
            "tính thực tế và hướng dẫn thực thi rõ ràng khi phù hợp."
        )

    def process_user_message(
        self,
        user_message: str,
        history: Sequence[Mapping[str, str]] | None = None,
    ) -> OrchestratorResult:
        if not user_message or not user_message.strip():
            raise ValueError("Tin nhắn người dùng không được rỗng")

        bounded_history = self._limit_history(history or ())
        logger.info(
            "Processing user request",
            extra={
                "event": "agent_request_started",
                "context": {"model": self.gateway.model_name, "history_count": len(bounded_history)},
            },
        )
        core_result = self.agent_core.process(user_message, history=bounded_history)
        if core_result.status in {"needs_clarification", "unsupported"}:
            response_text = core_result.response
        elif core_result.intent.value == "chat" and core_result.response is None:
            response_text = self.gateway.chat(
                user_message,
                system_prompt=self.build_system_prompt(),
                history=bounded_history,
            )
        else:
            response_text = core_result.response
        if not isinstance(response_text, str) or not response_text.strip():
            raise ModelGatewayError("Model gateway returned no response text")
        return OrchestratorResult(
            response=response_text,
            model=self.gateway.model_name,
            status=core_result.status,
            task_result=core_result.task_result,
        )

    def _limit_history(
        self,
        history: Sequence[Mapping[str, str]],
    ) -> list[dict[str, str]]:
        """Keep whole newest messages; truncate only the newest message from its start."""
        max_messages = self.configuration.max_history_messages
        max_chars = self.configuration.max_history_chars
        if max_messages == 0 or max_chars == 0:
            return []

        bounded: list[dict[str, str]] = []
        used_chars = 0
        candidates = history[-max_messages:]
        for item in reversed(candidates):
            role = item.get("role")
            content = item.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str) or not content:
                continue
            available_chars = max_chars - used_chars
            if available_chars <= 0:
                break
            if not bounded and len(content) > available_chars:
                bounded.append({"role": role, "content": content[-available_chars:]})
                break
            if len(content) > available_chars:
                break
            bounded.append({"role": role, "content": content})
            used_chars += len(content)

        bounded.reverse()
        return bounded

    def process_command(self, user_message: str) -> str:
        try:
            result = self.process_user_message(user_message)
            return result.response
        except (ModelGatewayError, ValueError) as exc:  # pragma: no cover - error path
            raise RuntimeError(str(exc)) from exc
