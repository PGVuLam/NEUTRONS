from __future__ import annotations

from collections.abc import Sequence

from pydantic import ValidationError

from neutrons.agent.executor import ActionRegistry
from neutrons.agent.schemas import ExecutionPlan, StepResult, StepStatus, ValidationResult


class ResultValidator:
    """Deterministically accepts only complete, successful, schema-compliant results."""

    def __init__(self, registry: ActionRegistry) -> None:
        self._registry = registry

    def validate(
        self,
        plan: ExecutionPlan,
        results: Sequence[StepResult],
    ) -> ValidationResult:
        errors: list[str] = []
        results_by_id: dict[str, StepResult] = {}
        for result in results:
            if result.step_id in results_by_id:
                errors.append("duplicate_step_result")
            results_by_id[result.step_id] = result

        expected_ids = {step.step_id for step in plan.steps}
        if set(results_by_id) != expected_ids:
            errors.append("incomplete_step_results")
        for step in plan.steps:
            result = results_by_id.get(step.step_id)
            if result is None:
                continue
            if result.status is not StepStatus.SUCCEEDED:
                errors.append("step_not_succeeded")
                continue
            action = self._registry.get(step.action)
            if action is None:
                errors.append("unregistered_action")
                continue
            if not action.required_outputs <= result.output.keys():
                errors.append("missing_required_output")
            try:
                action.output_model.model_validate(result.output)
            except ValidationError:
                errors.append("invalid_output_schema")
        return ValidationResult(valid=not errors, errors=errors)
