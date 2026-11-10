from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from pydantic import ValidationError

from neutrons.agent.schemas import ExecutionPlan
from neutrons.model_gateway import ChatModelGateway

MAX_PLAN_STEPS = 8
MAX_PLAN_OUTPUT_CHARS = 20_000


class PlanningError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def validate_plan(
    plan: ExecutionPlan,
    available_actions: Sequence[str],
    max_steps: int = MAX_PLAN_STEPS,
) -> None:
    if not 1 <= max_steps <= MAX_PLAN_STEPS:
        raise ValueError(f"max_steps must be between 1 and {MAX_PLAN_STEPS}")
    if len(plan.steps) > max_steps:
        raise PlanningError("step_limit", "Plan exceeds the configured step limit.")

    step_ids = [step.step_id for step in plan.steps]
    if len(step_ids) != len(set(step_ids)):
        raise PlanningError("duplicate_step_id", "Plan contains duplicate step IDs.")
    known_ids = set(step_ids)
    action_names = set(available_actions)
    for step in plan.steps:
        if step.action not in action_names:
            raise PlanningError("unsupported_action", "Plan contains an unsupported action.")
        if len(step.depends_on) != len(set(step.depends_on)):
            raise PlanningError("duplicate_dependency", "A step contains duplicate dependencies.")
        if step.step_id in step.depends_on or not set(step.depends_on) <= known_ids:
            raise PlanningError("invalid_dependency", "Plan contains an invalid dependency.")

    dependencies = {step.step_id: set(step.depends_on) for step in plan.steps}
    resolved: set[str] = set()
    while dependencies:
        ready = {step_id for step_id, parents in dependencies.items() if parents <= resolved}
        if not ready:
            raise PlanningError("dependency_cycle", "Plan dependencies contain a cycle.")
        resolved.update(ready)
        for step_id in ready:
            del dependencies[step_id]


class Planner:
    """Produces plans but has no reference to action handlers or execution authority."""

    def __init__(
        self,
        gateway: ChatModelGateway,
        action_schemas: Mapping[str, dict[str, object]],
        max_steps: int = MAX_PLAN_STEPS,
    ) -> None:
        if not 1 <= max_steps <= MAX_PLAN_STEPS:
            raise ValueError(f"max_steps must be between 1 and {MAX_PLAN_STEPS}")
        self._gateway = gateway
        self._action_schemas = dict(action_schemas)
        self._max_steps = max_steps

    def create_plan(
        self,
        user_message: str,
        history: Sequence[Mapping[str, str]] | None = None,
    ) -> ExecutionPlan:
        if not self._action_schemas:
            raise PlanningError("no_supported_actions", "No task actions are enabled.")

        schema_text = json.dumps(self._action_schemas, ensure_ascii=False)
        raw = self._gateway.chat_json(
            user_message,
            system_prompt=(
                "Create a bounded JSON execution plan using only the supplied registered actions. "
                "Do not invent actions, commands, or code. Include summary and a steps array; "
                "each step needs step_id, action, inputs, and depends_on. "
                f"Use no more than {self._max_steps} steps. Registered action schemas: {schema_text}"
            ),
            history=history,
        )
        if not isinstance(raw, str):
            raise PlanningError("malformed_plan", "The model did not produce a valid plan.")
        if len(raw) > MAX_PLAN_OUTPUT_CHARS:
            raise PlanningError("plan_too_large", "The model plan exceeded the size limit.")
        try:
            plan = ExecutionPlan.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError, TypeError, RecursionError) as exc:
            raise PlanningError("malformed_plan", "The model did not produce a valid plan.") from exc
        validate_plan(plan, tuple(self._action_schemas), self._max_steps)
        return plan
