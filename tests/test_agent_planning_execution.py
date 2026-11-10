from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from neutrons.agent.executor import (
    ActionContext,
    ActionError,
    ActionRegistry,
    ActionSpec,
    Executor,
    PermissionPolicy,
)
from neutrons.agent.planner import Planner, PlanningError, validate_plan
from neutrons.agent.schemas import ExecutionPlan, StepStatus, TaskStatus
from neutrons.agent.state import TaskStateManager
from neutrons.agent.validator import ResultValidator


class Inputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


class Outputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    result: Any


class PlannerGateway:
    model_name = "test-model"

    def __init__(self, response: str) -> None:
        self.response = response

    def chat(
        self,
        user_message: str,
        system_prompt: str | None = None,
        history: Sequence[Mapping[str, str]] | None = None,
    ) -> str:
        assert user_message
        assert "registered actions" in system_prompt
        return self.response

    def check_health(self, timeout: float | None = None) -> bool:
        return True

    def get_model_info(self, timeout: float | None = None):
        return None


def _plan(*steps: dict[str, Any]) -> ExecutionPlan:
    return ExecutionPlan(summary="test", steps=list(steps))


def _active_state(plan: ExecutionPlan) -> tuple[TaskStateManager, str]:
    state = TaskStateManager()
    task_id = state.create_task()
    state.transition_task(task_id, TaskStatus.PLANNING)
    for step in plan.steps:
        state.add_step(task_id, step.step_id)
    state.transition_task(task_id, TaskStatus.READY)
    state.transition_task(task_id, TaskStatus.RUNNING)
    return state, task_id


def _action(
    handler,
    *,
    outputs: frozenset[str] = frozenset({"result"}),
    timeout: float = 1,
    confirmation: bool = False,
) -> ActionSpec:
    return ActionSpec(
        name="safe.test",
        input_model=Inputs,
        output_model=Outputs,
        required_outputs=outputs,
        handler=handler,
        timeout_seconds=timeout,
        requires_confirmation=confirmation,
    )


class AllowRegisteredActions(PermissionPolicy):
    def allows(self, action: ActionSpec, task_id: str) -> bool:
        return action.name == "safe.test" and bool(task_id)


def test_planner_validates_registered_actions_and_dependencies() -> None:
    response = (
        '{"summary":"test","steps":['
        '{"step_id":"first","action":"safe.test","inputs":{"value":1},"depends_on":[]},'
        '{"step_id":"second","action":"safe.test","inputs":{"value":2},"depends_on":["first"]}]}'
    )
    planner = Planner(
        PlannerGateway(response),
        {"safe.test": {"input": {"value": "integer"}}},
    )

    plan = planner.create_plan("Run this task")

    assert [step.step_id for step in plan.steps] == ["first", "second"]


def test_planner_refuses_to_make_up_actions_or_run_without_capabilities() -> None:
    no_actions = Planner(PlannerGateway("{}"), {})
    with pytest.raises(PlanningError, match="No task actions"):
        no_actions.create_plan("Do a task")

    unknown_action = _plan(
        {"step_id": "step", "action": "shell", "inputs": {}, "depends_on": []}
    )
    with pytest.raises(PlanningError) as error:
        validate_plan(unknown_action, ["safe.test"])
    assert error.value.code == "unsupported_action"


@pytest.mark.parametrize(
    ("raw", "error_code"),
    [
        ("not json", "malformed_plan"),
        ("x" * 20_001, "plan_too_large"),
    ],
)
def test_planner_rejects_malformed_or_oversized_model_output(
    raw: str, error_code: str
) -> None:
    planner = Planner(PlannerGateway(raw), {"safe.test": {"input": {}}})

    with pytest.raises(PlanningError) as error:
        planner.create_plan("Run this task")
    assert error.value.code == error_code


@pytest.mark.parametrize(
    ("steps", "error_code"),
    [
        (
            [
                {"step_id": "same", "action": "safe.test", "inputs": {}, "depends_on": []},
                {"step_id": "same", "action": "safe.test", "inputs": {}, "depends_on": []},
            ],
            "duplicate_step_id",
        ),
        (
            [{"step_id": "one", "action": "safe.test", "inputs": {}, "depends_on": ["missing"]}],
            "invalid_dependency",
        ),
        (
            [
                {"step_id": "one", "action": "safe.test", "inputs": {}, "depends_on": ["two"]},
                {"step_id": "two", "action": "safe.test", "inputs": {}, "depends_on": ["one"]},
            ],
            "dependency_cycle",
        ),
    ],
)
def test_planner_rejects_malformed_graphs(steps, error_code: str) -> None:
    with pytest.raises(PlanningError) as error:
        validate_plan(_plan(*steps), ["safe.test"])
    assert error.value.code == error_code


def test_planner_enforces_hard_step_bound() -> None:
    with pytest.raises(PlanningError) as error:
        validate_plan(
            _plan(
                *[
                    {
                        "step_id": f"step_{index}",
                        "action": "safe.test",
                        "inputs": {},
                        "depends_on": [],
                    }
                    for index in range(3)
                ]
            ),
            ["safe.test"],
            max_steps=2,
        )
    assert error.value.code == "step_limit"


def test_executor_runs_only_registered_permitted_actions_and_sorts_dependencies() -> None:
    calls: list[int] = []

    async def handler(inputs: BaseModel, context: ActionContext) -> Mapping[str, Any]:
        context.check_boundary()
        calls.append(inputs.value)
        return {"result": inputs.value * 2}

    registry = ActionRegistry()
    registry.register(_action(handler))
    plan = _plan(
        {"step_id": "second", "action": "safe.test", "inputs": {"value": 2}, "depends_on": ["first"]},
        {"step_id": "first", "action": "safe.test", "inputs": {"value": 1}, "depends_on": []},
    )
    state, task_id = _active_state(plan)

    results = asyncio.run(
        Executor(registry, AllowRegisteredActions()).execute(plan, task_id, state)
    )

    assert calls == [1, 2]
    assert [result.status for result in results] == [StepStatus.SUCCEEDED, StepStatus.SUCCEEDED]
    assert ResultValidator(registry).validate(plan, results).valid


def test_executor_denies_by_default_and_never_uses_unregistered_action() -> None:
    called = False

    async def handler(inputs: BaseModel, context: ActionContext) -> Mapping[str, Any]:
        nonlocal called
        called = True
        return {"result": "unsafe"}

    registry = ActionRegistry()
    registry.register(_action(handler))
    plan = _plan(
        {"step_id": "step", "action": "safe.test", "inputs": {"value": 1}, "depends_on": []}
    )
    state, task_id = _active_state(plan)

    results = asyncio.run(Executor(registry).execute(plan, task_id, state))

    assert not called
    assert results[0].error_code == "permission_denied"
    assert not ResultValidator(registry).validate(plan, results).valid


def test_executor_rejects_invalid_inputs_and_unknown_action() -> None:
    async def handler(inputs: BaseModel, context: ActionContext) -> Mapping[str, Any]:
        return {"result": "ok"}

    registry = ActionRegistry()
    registry.register(_action(handler))
    for action, inputs, expected in (
        ("safe.test", {"unexpected": True}, "invalid_inputs"),
        ("unknown", {}, "unknown_action"),
    ):
        plan = _plan(
            {"step_id": "step", "action": action, "inputs": inputs, "depends_on": []}
        )
        state, task_id = _active_state(plan)
        results = asyncio.run(
            Executor(registry, AllowRegisteredActions()).execute(plan, task_id, state)
        )
        assert results[0].error_code == expected


def test_executor_times_out_and_stops_after_failure() -> None:
    async def slow_handler(inputs: BaseModel, context: ActionContext) -> Mapping[str, Any]:
        await asyncio.sleep(0.1)
        return {"result": "late"}

    registry = ActionRegistry()
    registry.register(_action(slow_handler, timeout=0.01))
    plan = _plan(
        {"step_id": "first", "action": "safe.test", "inputs": {"value": 1}, "depends_on": []},
        {"step_id": "second", "action": "safe.test", "inputs": {"value": 2}, "depends_on": []},
    )
    state, task_id = _active_state(plan)

    results = asyncio.run(
        Executor(registry, AllowRegisteredActions()).execute(plan, task_id, state)
    )

    assert [result.status for result in results] == [StepStatus.TIMED_OUT, StepStatus.SKIPPED]
    assert state.get_snapshot(task_id).step_statuses["second"] is StepStatus.SKIPPED


def test_executor_records_registered_action_failure() -> None:
    async def failing_handler(inputs: BaseModel, context: ActionContext) -> Mapping[str, Any]:
        raise ActionError("expected_failure")

    registry = ActionRegistry()
    registry.register(_action(failing_handler))
    plan = _plan(
        {"step_id": "step", "action": "safe.test", "inputs": {"value": 1}, "depends_on": []}
    )
    state, task_id = _active_state(plan)

    results = asyncio.run(
        Executor(registry, AllowRegisteredActions()).execute(plan, task_id, state)
    )

    assert results[0].status is StepStatus.FAILED
    assert results[0].error_code == "expected_failure"
    assert state.get_snapshot(task_id).step_statuses["step"] is StepStatus.FAILED
    assert state.get_snapshot(task_id).step_errors["step"] == "expected_failure"


def test_executor_rejects_oversized_action_output() -> None:
    async def large_handler(inputs: BaseModel, context: ActionContext) -> Mapping[str, Any]:
        return {"result": "x" * 20_001}

    registry = ActionRegistry()
    registry.register(_action(large_handler))
    plan = _plan(
        {"step_id": "step", "action": "safe.test", "inputs": {"value": 1}, "depends_on": []}
    )
    state, task_id = _active_state(plan)

    result = asyncio.run(
        Executor(registry, AllowRegisteredActions()).execute(plan, task_id, state)
    )[0]

    assert result.status is StepStatus.FAILED
    assert result.error_code == "output_too_large"


def test_executor_stops_at_safe_boundary_after_cancellation() -> None:
    cancellation = threading.Event()

    async def cancel_after_first(inputs: BaseModel, context: ActionContext) -> Mapping[str, Any]:
        cancellation.set()
        return {"result": inputs.value}

    registry = ActionRegistry()
    registry.register(_action(cancel_after_first))
    plan = _plan(
        {"step_id": "first", "action": "safe.test", "inputs": {"value": 1}, "depends_on": []},
        {"step_id": "second", "action": "safe.test", "inputs": {"value": 2}, "depends_on": []},
    )
    state, task_id = _active_state(plan)

    results = asyncio.run(
        Executor(registry, AllowRegisteredActions()).execute(
            plan, task_id, state, cancellation=cancellation
        )
    )
    state.transition_task(task_id, TaskStatus.CANCELLED)

    assert [result.status for result in results] == [StepStatus.SUCCEEDED, StepStatus.SKIPPED]
    assert state.get_snapshot(task_id).status is TaskStatus.CANCELLED


def test_executor_requires_explicit_confirmation() -> None:
    called = False

    async def handler(inputs: BaseModel, context: ActionContext) -> Mapping[str, Any]:
        nonlocal called
        called = True
        return {"result": "done"}

    registry = ActionRegistry()
    registry.register(_action(handler, confirmation=True))
    plan = _plan(
        {"step_id": "step", "action": "safe.test", "inputs": {"value": 1}, "depends_on": []}
    )
    state, task_id = _active_state(plan)

    results = asyncio.run(
        Executor(registry, AllowRegisteredActions()).execute(plan, task_id, state)
    )

    assert not called
    assert results[0].status is StepStatus.NEEDS_CONFIRMATION
    assert results[0].error_code == "confirmation_required"


def test_action_registry_requires_closed_input_and_output_schemas() -> None:
    class OpenInputs(BaseModel):
        value: int

    async def handler(inputs: BaseModel, context: ActionContext) -> Mapping[str, Any]:
        return {"result": "ok"}

    with pytest.raises(ValueError, match="forbid unknown fields"):
        ActionRegistry().register(
            ActionSpec(
                name="safe.test",
                input_model=OpenInputs,
                output_model=Outputs,
                required_outputs=frozenset({"result"}),
                handler=handler,
            )
        )


def test_validator_rejects_missing_outputs_and_incomplete_results() -> None:
    async def handler(inputs: BaseModel, context: ActionContext) -> Mapping[str, Any]:
        return {}

    registry = ActionRegistry()
    registry.register(_action(handler))
    plan = _plan(
        {"step_id": "step", "action": "safe.test", "inputs": {"value": 1}, "depends_on": []}
    )

    result = ResultValidator(registry).validate(plan, [])
    assert not result.valid
    assert "incomplete_step_results" in result.errors

    from neutrons.agent.schemas import StepResult

    step_result = StepResult(step_id="step", status=StepStatus.SUCCEEDED, output={})
    result = ResultValidator(registry).validate(plan, [step_result])
    assert not result.valid
    assert "missing_required_output" in result.errors
    assert "invalid_output_schema" in result.errors
