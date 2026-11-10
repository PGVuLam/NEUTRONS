from __future__ import annotations

import asyncio
import inspect
import json
import threading
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from time import monotonic
from typing import Any

from pydantic import BaseModel, ValidationError

from neutrons.agent.planner import MAX_PLAN_STEPS
from neutrons.agent.schemas import ExecutionPlan, PlanStep, StepResult, StepStatus
from neutrons.agent.state import TaskStateManager

MAX_ACTION_TIMEOUT_SECONDS = 30.0
MAX_ACTION_OUTPUT_CHARS = 20_000


class ActionError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


@dataclass(frozen=True)
class ActionContext:
    deadline: float
    cancellation: threading.Event

    def check_boundary(self) -> None:
        if self.cancellation.is_set():
            raise ActionError("cancelled")
        if monotonic() >= self.deadline:
            raise ActionError("timed_out")


ActionHandler = Callable[[BaseModel, ActionContext], Awaitable[Mapping[str, Any]]]


@dataclass(frozen=True)
class ActionSpec:
    name: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    required_outputs: frozenset[str]
    handler: ActionHandler
    timeout_seconds: float = 10.0
    requires_confirmation: bool = False


class ActionRegistry:
    def __init__(self) -> None:
        self._actions: dict[str, ActionSpec] = {}
        self._frozen = False

    def register(self, action: ActionSpec) -> None:
        if self._frozen:
            raise RuntimeError("The action registry is frozen")
        if not action.name or len(action.name) > 64:
            raise ValueError("Action names must contain between 1 and 64 characters")
        if action.name in self._actions:
            raise ValueError(f"Action is already registered: {action.name}")
        if not inspect.iscoroutinefunction(action.handler):
            raise TypeError("Action handlers must be async functions")
        if action.input_model.model_config.get("extra") != "forbid":
            raise ValueError("Action input models must forbid unknown fields")
        if action.output_model.model_config.get("extra") != "forbid":
            raise ValueError("Action output models must forbid unknown fields")
        if not action.required_outputs <= action.output_model.model_fields.keys():
            raise ValueError("Required outputs must be defined by the output model")
        if not 0 < action.timeout_seconds <= MAX_ACTION_TIMEOUT_SECONDS:
            raise ValueError(
                f"Action timeout must be greater than 0 and at most {MAX_ACTION_TIMEOUT_SECONDS} seconds"
            )
        self._actions[action.name] = action

    def freeze(self) -> None:
        self._frozen = True

    def get(self, name: str) -> ActionSpec | None:
        return self._actions.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(self._actions)

    def schemas(self) -> dict[str, dict[str, object]]:
        return {
            name: {
                "input": action.input_model.model_json_schema(),
                "output": action.output_model.model_json_schema(),
                "required_outputs": sorted(action.required_outputs),
                "requires_confirmation": action.requires_confirmation,
            }
            for name, action in self._actions.items()
        }


class PermissionPolicy:
    """Default-deny permission boundary, overridable only by application policy."""

    def allows(self, action: ActionSpec, task_id: str) -> bool:
        return False


class Executor:
    def __init__(
        self,
        registry: ActionRegistry,
        permission_policy: PermissionPolicy | None = None,
        max_steps: int = MAX_PLAN_STEPS,
    ) -> None:
        if not 1 <= max_steps <= MAX_PLAN_STEPS:
            raise ValueError(f"max_steps must be between 1 and {MAX_PLAN_STEPS}")
        self._registry = registry
        self._permission_policy = permission_policy or PermissionPolicy()
        self._max_steps = max_steps

    async def execute(
        self,
        plan: ExecutionPlan,
        task_id: str,
        state: TaskStateManager,
        *,
        cancellation: threading.Event | None = None,
    ) -> list[StepResult]:
        if len(plan.steps) > self._max_steps:
            raise ValueError("Plan exceeds the executor step limit")

        ordered_steps = self._execution_order(plan)
        cancellation = cancellation or threading.Event()
        results: dict[str, StepResult] = {}
        for index, step in enumerate(ordered_steps):
            if cancellation.is_set():
                for remaining in ordered_steps[index:]:
                    results[remaining.step_id] = self._finish(
                        state, task_id, remaining, StepStatus.SKIPPED
                    )
                break
            if step.step_id in results:
                continue
            if any(
                results.get(dependency) is None
                or results[dependency].status is not StepStatus.SUCCEEDED
                for dependency in step.depends_on
            ):
                results[step.step_id] = self._finish(state, task_id, step, StepStatus.SKIPPED)
                continue
            result = await self._execute_step(
                step, task_id, state, cancellation
            )
            results[step.step_id] = result
            if result.status not in {StepStatus.SUCCEEDED, StepStatus.SKIPPED}:
                for later_step in ordered_steps[index + 1 :]:
                    results[later_step.step_id] = self._finish(
                        state, task_id, later_step, StepStatus.SKIPPED
                    )
                break
        return [results[step.step_id] for step in plan.steps]

    @staticmethod
    def _execution_order(plan: ExecutionPlan) -> list[PlanStep]:
        by_id = {step.step_id: step for step in plan.steps}
        if len(by_id) != len(plan.steps):
            raise ValueError("Plan contains duplicate step IDs")
        if any(set(step.depends_on) - by_id.keys() for step in plan.steps):
            raise ValueError("Plan contains an unknown dependency")

        pending = list(plan.steps)
        ordered: list[PlanStep] = []
        completed: set[str] = set()
        while pending:
            ready = [step for step in pending if set(step.depends_on) <= completed]
            if not ready:
                raise ValueError("Plan dependencies contain a cycle")
            for step in ready:
                pending.remove(step)
                ordered.append(step)
                completed.add(step.step_id)
        return ordered

    async def _execute_step(
        self,
        step: PlanStep,
        task_id: str,
        state: TaskStateManager,
        cancellation: threading.Event,
    ) -> StepResult:
        action = self._registry.get(step.action)
        state.transition_step(task_id, step.step_id, StepStatus.RUNNING)
        if action is None:
            return self._finish(
                state, task_id, step, StepStatus.FAILED, error_code="unknown_action"
            )
        if not self._permission_policy.allows(action, task_id):
            return self._finish(
                state, task_id, step, StepStatus.FAILED, error_code="permission_denied"
            )
        if action.requires_confirmation:
            return self._finish(
                state,
                task_id,
                step,
                StepStatus.NEEDS_CONFIRMATION,
                error_code="confirmation_required",
            )
        try:
            inputs = action.input_model.model_validate(step.inputs)
        except ValidationError:
            return self._finish(
                state, task_id, step, StepStatus.FAILED, error_code="invalid_inputs"
            )

        context = ActionContext(
            deadline=monotonic() + action.timeout_seconds,
            cancellation=cancellation,
        )
        try:
            output = await asyncio.wait_for(
                action.handler(inputs, context),
                timeout=action.timeout_seconds,
            )
        except asyncio.TimeoutError:
            return self._finish(
                state, task_id, step, StepStatus.TIMED_OUT, error_code="action_timeout"
            )
        except ActionError as exc:
            status = StepStatus.TIMED_OUT if exc.error_code == "timed_out" else StepStatus.FAILED
            return self._finish(
                state, task_id, step, status, error_code=exc.error_code
            )
        if not isinstance(output, Mapping):
            return self._finish(
                state, task_id, step, StepStatus.FAILED, error_code="invalid_output"
            )
        try:
            output_json = json.dumps(output, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError, RecursionError):
            return self._finish(
                state, task_id, step, StepStatus.FAILED, error_code="invalid_output"
            )
        if len(output_json) > MAX_ACTION_OUTPUT_CHARS:
            return self._finish(
                state, task_id, step, StepStatus.FAILED, error_code="output_too_large"
            )
        try:
            validated_output = action.output_model.model_validate(output).model_dump()
        except ValidationError:
            return self._finish(
                state, task_id, step, StepStatus.FAILED, error_code="invalid_output"
            )
        result = self._finish(state, task_id, step, StepStatus.SUCCEEDED)
        result.output = validated_output
        return result

    @staticmethod
    def _finish(
        state: TaskStateManager,
        task_id: str,
        step: PlanStep,
        status: StepStatus,
        *,
        error_code: str | None = None,
    ) -> StepResult:
        state.transition_step(task_id, step.step_id, status, error_code=error_code)
        return StepResult(step_id=step.step_id, status=status, error_code=error_code)
