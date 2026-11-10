from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence

from pydantic import ValidationError

from neutrons.agent.schemas import Intent, IntentDecision
from neutrons.model_gateway import ChatModelGateway

logger = logging.getLogger(__name__)
MAX_INTENT_OUTPUT_CHARS = 20_000


class IntentRouter:
    """Classifies requests; it never creates or executes actions."""

    def __init__(
        self,
        gateway: ChatModelGateway,
        confidence_threshold: float = 0.65,
        chat_system_prompt: str | None = None,
    ) -> None:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self._gateway = gateway
        self._confidence_threshold = confidence_threshold
        self._chat_system_prompt = chat_system_prompt

    def route(
        self,
        user_message: str,
        history: Sequence[Mapping[str, str]] | None = None,
    ) -> IntentDecision:
        classification_prompt = (
            "You are NEUTRONS, a local personal assistant. "
            "Classify the request as chat, task, clarification, or unsupported. "
            "Do not plan or execute actions. For chat, provide the direct user-facing "
            "answer in response. For ambiguous requests, choose clarification. "
            "Return only a JSON object with intent, confidence (0 to 1), reason, "
            "and response (string for chat, otherwise null)."
        )
        system_prompt = " ".join(
            part for part in (self._chat_system_prompt, classification_prompt) if part
        )
        raw = self._gateway.chat(
            user_message,
            system_prompt=system_prompt,
            history=history,
        )
        if not isinstance(raw, str) or len(raw) > MAX_INTENT_OUTPUT_CHARS:
            logger.warning(
                "Intent router output exceeded its limit",
                extra={"event": "intent_output_invalid"},
            )
            return IntentDecision(
                intent=Intent.CHAT,
                confidence=1,
                reason="Fallback to the existing chat path because classification was invalid.",
            )
        try:
            value = json.loads(raw)
            decision = IntentDecision.model_validate(value)
        except (json.JSONDecodeError, ValidationError, TypeError, RecursionError):
            logger.warning(
                "Intent router returned malformed output",
                extra={"event": "intent_output_invalid"},
            )
            return IntentDecision(
                intent=Intent.CHAT,
                confidence=1,
                reason="Fallback to the existing chat path because classification was invalid.",
            )

        if decision.confidence < self._confidence_threshold:
            return self._clarification(
                "Yêu cầu chưa đủ rõ để xử lý an toàn; vui lòng bổ sung thông tin."
            )
        return decision

    @staticmethod
    def _clarification(reason: str) -> IntentDecision:
        return IntentDecision(intent=Intent.CLARIFICATION, confidence=0, reason=reason)
