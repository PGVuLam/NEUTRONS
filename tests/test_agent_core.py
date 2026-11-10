from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from neutrons.agent.core import AgentCore
from neutrons.agent.executor import ActionContext, ActionRegistry, ActionSpec, PermissionPolicy
from neutrons.agent.schemas import Intent, TaskStatus
from neutrons.model_gateway import ModelTimeoutError


class Gateway:
    model_name = "test-model"

    def __init__(self, *responses: str) -> None:
        self.responses = iter(responses)
        self.call_count = 0
        self.on_second_call = None

    def chat(
        self,
        user_message: str,
        system_prompt: str | None = None,
        history: Sequence[Mapping[str, str]] | None = None,
    ) -> str:
        self.call_count += 1
        if self.call_count == 2 and self.on_second_call is not None:
            self.on_second_call()
        return next(self.responses)

    def check_health(self, timeout: float | None = None) -> bool:
        return True

    def get_model_info(self, timeout: float | None = None):
        return None


class Inputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


class Outputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    result: int


class AllowAction(PermissionPolicy):
    def allows(self, action: ActionSpec, task_id: str) -> bool:
        return True


def _route(intent: str, response: str | None = None) -> str:
    return json.dumps(
        {
            "intent": intent,
            "confidence": 0.99,
            "reason": "classified",
            "response": response,
        }
    )


def test_agent_core_keeps_chat_in_the_chat_path() -> None:
    core = AgentCore(Gateway(_route("chat", "Hello!")))

    result = core.process("Hi")

    assert result.intent is Intent.CHAT
    assert result.status == "ok"
    assert result.response == "Hello!"
    assert result.task_id is None


def test_agent_core_safely_rejects_task_without_registered_actions() -> None:
    core = AgentCore(Gateway(_route("task")))

    result = core.process("Do a task")

    assert result.intent is Intent.TASK
    assert result.status == "unsupported"
    assert result.task_id is not None
    assert core.state.get_snapshot(result.task_id).status is TaskStatus.FAILED


def test_agent_core_completes_only_registered_and_validated_task() -> None:
    registry = ActionRegistry()

    async def handler(inputs: BaseModel, context: ActionContext) -> Mapping[str, Any]:
        context.check_boundary()
        return {"result": inputs.value + 1}

    registry.register(
        ActionSpec(
            name="safe.test",
            input_model=Inputs,
            output_model=Outputs,
            required_outputs=frozenset({"result"}),
            handler=handler,
        )
    )
    plan = json.dumps(
        {
            "summary": "Increment a value",
            "steps": [
                {
                    "step_id": "step_1",
                    "action": "safe.test",
                    "inputs": {"value": 4},
                    "depends_on": [],
                }
            ],
        }
    )
    core = AgentCore(Gateway(_route("task"), plan), registry, AllowAction())

    result = core.process("Increment the value")

    assert result.status == "ok"
    assert core.state.get_snapshot(result.task_id).status is TaskStatus.COMPLETED
    assert result.response == "Tác vụ đã hoàn thành và kết quả đã được xác thực."


def test_agent_core_does_not_execute_without_confirmation() -> None:
    executed = False
    registry = ActionRegistry()

    async def handler(inputs: BaseModel, context: ActionContext) -> Mapping[str, Any]:
        nonlocal executed
        executed = True
        return {"result": "done"}

    registry.register(
        ActionSpec(
            name="safe.test",
            input_model=Inputs,
            output_model=Outputs,
            required_outputs=frozenset({"result"}),
            handler=handler,
            requires_confirmation=True,
        )
    )
    plan = json.dumps(
        {
            "summary": "Confirm first",
            "steps": [
                {
                    "step_id": "step_1",
                    "action": "safe.test",
                    "inputs": {"value": 1},
                    "depends_on": [],
                }
            ],
        }
    )
    core = AgentCore(Gateway(_route("task"), plan), registry, AllowAction())

    result = core.process("Run the action")

    assert not executed
    assert result.status == "needs_confirmation"
    assert core.state.get_snapshot(result.task_id).status is TaskStatus.NEEDS_CONFIRMATION


def test_agent_core_cancellation_during_planning_stops_before_execution() -> None:
    executed = False
    registry = ActionRegistry()

    async def handler(inputs: BaseModel, context: ActionContext) -> Mapping[str, Any]:
        nonlocal executed
        executed = True
        return {"result": inputs.value}

    registry.register(
        ActionSpec(
            name="safe.test",
            input_model=Inputs,
            output_model=Outputs,
            required_outputs=frozenset({"result"}),
            handler=handler,
        )
    )
    plan = json.dumps(
        {
            "summary": "Do not execute",
            "steps": [
                {
                    "step_id": "step_1",
                    "action": "safe.test",
                    "inputs": {"value": 1},
                    "depends_on": [],
                }
            ],
        }
    )
    gateway = Gateway(_route("task"), plan)
    core = AgentCore(gateway, registry, AllowAction())
    gateway.on_second_call = lambda: core.cancel(core.active_task_ids()[0])

    result = core.process("Cancel before execution")

    assert not executed
    assert result.status == "cancelled"
    assert core.state.get_snapshot(result.task_id).status is TaskStatus.CANCELLED
    assert core.active_task_ids() == ()


def test_agent_core_marks_planning_provider_failure_and_surfaces_it() -> None:
    registry = ActionRegistry()

    async def handler(inputs: BaseModel, context: ActionContext) -> Mapping[str, Any]:
        return {"result": inputs.value}

    registry.register(
        ActionSpec(
            name="safe.test",
            input_model=Inputs,
            output_model=Outputs,
            required_outputs=frozenset({"result"}),
            handler=handler,
        )
    )
    gateway = Gateway(_route("task"))
    core = AgentCore(gateway, registry, AllowAction())
    task_ids: list[str] = []

    def fail_planning() -> None:
        task_ids.extend(core.active_task_ids())
        raise ModelTimeoutError("provider detail")

    gateway.on_second_call = fail_planning

    with pytest.raises(ModelTimeoutError):
        core.process("Run the task")

    snapshot = core.state.get_snapshot(task_ids[0])
    assert snapshot.status is TaskStatus.FAILED
    assert snapshot.error_code == "planning_model_unavailable"
    assert core.active_task_ids() == ()
