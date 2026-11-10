from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from neutrons.model_gateway import ModelGatewayError, OllamaChatGateway


@dataclass(slots=True)
class OrchestratorResult:
    response: str
    model: str
    status: str = "ok"


class AgentOrchestrator:
    """Coordinates user intent with the local model gateway."""

    def __init__(self, gateway: OllamaChatGateway | None = None) -> None:
        self.gateway = gateway or OllamaChatGateway()

    def build_system_prompt(self) -> str:
        return (
            "Bạn là NEUTRONS, một trợ lý AI cá nhân chạy cục bộ. "
            "Hãy trả lời rõ ràng, ngắn gọn, có cấu trúc, và ưu tiên an toàn, "
            "tính thực tế và hướng dẫn thực thi rõ ràng khi phù hợp."
        )

    def process_user_message(
        self,
        user_message: str,
        history: Sequence[dict[str, str]] | None = None,
    ) -> OrchestratorResult:
        if not user_message or not user_message.strip():
            raise ValueError("Tin nhắn người dùng không được rỗng")

        response_text = self.gateway.chat(
            user_message,
            system_prompt=self.build_system_prompt(),
            history=history,
        )
        return OrchestratorResult(response=response_text, model=self.gateway.model_name)

    def process_command(self, user_message: str) -> str:
        try:
            result = self.process_user_message(user_message)
            return result.response
        except (ModelGatewayError, ValueError) as exc:  # pragma: no cover - error path
            raise RuntimeError(str(exc)) from exc
