from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def build_system_prompt(application_prompt: str | None = None) -> str:
    policy = (
        "You are NEUTRONS, a local personal AI assistant. "
        "Respond in the language of the user's current request, preserving technical "
        "terms, code identifiers, proper nouns, and the name NEUTRONS. Answer directly "
        "and naturally; keep greetings short, avoid unnecessary introductions, repetition, "
        "disclaimers, and process narration. Never reveal private reasoning, drafts, or "
        "internal planning. Do not claim an action or operation succeeded unless its "
        "validated result is explicitly provided. State uncertainty when information is "
        "unavailable. Return only the requested user-facing content."
    )
    return f"{application_prompt.strip()} {policy}" if application_prompt else policy


def format_validated_task_outputs(outputs: Mapping[str, Mapping[str, Any]]) -> str:
    serialized = json.dumps(outputs, ensure_ascii=False, indent=2, allow_nan=False)
    return f"Tác vụ đã hoàn thành và kết quả đã được xác thực:\n\n```json\n{serialized}\n```"
