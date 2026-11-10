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
            "Classify the request as chat, task, clarification, or unsupported. "
            "Do not plan or execute actions. For chat, provide a direct and concise "
            "user-facing answer in response. For clarification, ask one specific question. "
            "For unsupported requests, explain the limitation without claiming action. "
            "Use the language of the current request. Return only an object with intent, "
            "confidence (0 to 1), and response. Use no other fields."
        )
        system_prompt = " ".join(
            part for part in (self._chat_system_prompt, classification_prompt) if part
        )
        raw = self._gateway.chat_json(
            user_message,
            system_prompt=system_prompt,
            history=history,
        )
        if not isinstance(raw, str) or len(raw) > MAX_INTENT_OUTPUT_CHARS:
            logger.warning(
                "Intent router output exceeded its limit",
                extra={"event": "intent_output_invalid"},
            )
            return self._fallback_to_chat()
        try:
            value = json.loads(raw)
            decision = IntentDecision.model_validate(value)
        except (json.JSONDecodeError, ValidationError, TypeError, RecursionError):
            logger.warning(
                "Intent router returned malformed output",
                extra={"event": "intent_output_invalid"},
            )
            return self._fallback_to_chat()

        if decision.confidence < self._confidence_threshold:
            if not decision.response:
                return self._fallback_to_chat()
            return IntentDecision(
                intent=Intent.CLARIFICATION,
                confidence=decision.confidence,
                response=decision.response,
            )
        if decision.intent in {Intent.CHAT, Intent.CLARIFICATION, Intent.UNSUPPORTED}:
            if not decision.response:
                return self._fallback_to_chat()
        return decision

    @staticmethod
    def _fallback_to_chat() -> IntentDecision:
        return IntentDecision(intent=Intent.CHAT, confidence=1)
