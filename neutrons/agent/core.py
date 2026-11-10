from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from neutrons.agent.executor import ActionRegistry, Executor, PermissionPolicy
from neutrons.agent.planner import Planner, PlanningError
from neutrons.agent.router import IntentRouter
from neutrons.agent.schemas import Intent, StepResult, StepStatus, TaskExecutionResult, TaskStatus
from neutrons.agent.state import TaskStateManager
from neutrons.agent.validator import ResultValidator
from neutrons.agent.response_policy import format_validated_task_outputs
from neutrons.model_gateway import ChatModelGateway, ModelGatewayError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentCoreResult:
    response: str | None
    intent: Intent
    status: str
    task_id: str | None = None
    task_result: TaskExecutionResult | None = None


class AgentCore:
    """Coordinates bounded routing, planning, execution, validation, and task state."""

    def __init__(
        self,
        gateway: ChatModelGateway,
        registry: ActionRegistry | None = None,
        permission_policy: PermissionPolicy | None = None,
        state_manager: TaskStateManager | None = None,
        chat_system_prompt: str | None = None,
    ) -> None:
        self.registry = registry if registry is not None else ActionRegistry()
        self.registry.freeze()
        self.state = state_manager or TaskStateManager()
        self._router = IntentRouter(gateway, chat_system_prompt=chat_system_prompt)
        self._planner = Planner(gateway, self.registry.schemas())
        self._executor = Executor(self.registry, permission_policy)
        self._validator = ResultValidator(self.registry)
        self._active_cancellations: dict[str, threading.Event] = {}
        self._lock = threading.RLock()

    def process(
        self,
        user_message: str,
        history: Sequence[Mapping[str, str]] | None = None,
    ) -> AgentCoreResult:
        decision = self._router.route(user_message, history)
        if decision.intent is Intent.CHAT:
            return AgentCoreResult(decision.response, Intent.CHAT, "ok")
        if decision.intent is Intent.CLARIFICATION:
            return AgentCoreResult(decision.response, decision.intent, "needs_clarification")
        if decision.intent is Intent.UNSUPPORTED:
            return AgentCoreResult(decision.response, decision.intent, "unsupported")
        return self._process_task(user_message, history)

    def cancel(self, task_id: str) -> None:
        snapshot = self.state.get_snapshot(task_id)
        if snapshot.status in {TaskStatus.PLANNING, TaskStatus.READY, TaskStatus.RUNNING}:
            with self._lock:
                cancellation = self._active_cancellations.get(task_id)
            if cancellation is None:
                raise RuntimeError("The task is not at an active execution boundary")
            cancellation.set()
            return
        self.state.transition_task(task_id, TaskStatus.CANCELLED)

    def active_task_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._active_cancellations)

    def _process_task(
        self,
        user_message: str,
        history: Sequence[Mapping[str, str]] | None,
    ) -> AgentCoreResult:
        task_id = self.state.create_task()
        self.state.transition_task(task_id, TaskStatus.PLANNING)
        cancellation = threading.Event()
        with self._lock:
            self._active_cancellations[task_id] = cancellation
        try:
            try:
                plan = self._planner.create_plan(user_message, history)
            except PlanningError as exc:
                if cancellation.is_set():
                    self.state.transition_task(task_id, TaskStatus.CANCELLED)
                    return self._cancelled_result(task_id)
                self.state.transition_task(task_id, TaskStatus.FAILED, error_code=exc.code)
                logger.info(
                    "Task planning did not produce an executable plan",
                    extra={
                        "event": "agent_task_not_planned",
                        "context": {"error_code": exc.code},
                    },
                )
                status = (
                    "unsupported"
                    if exc.code in {"no_supported_actions", "unsupported_action"}
                    else "failed"
                )
                response = (
                    "NEUTRONS hiện chưa có hành động được cấp quyền phù hợp để thực hiện tác vụ này."
                    if status == "unsupported"
                    else "Không thể tạo kế hoạch an toàn cho tác vụ này."
                )
                return AgentCoreResult(
                    response,
                    Intent.TASK,
                    status,
                    task_id,
                    self._task_result(task_id),
                )
            except ModelGatewayError:
                if cancellation.is_set():
                    self.state.transition_task(task_id, TaskStatus.CANCELLED)
                    return self._cancelled_result(task_id)
                self.state.transition_task(
                    task_id,
                    TaskStatus.FAILED,
                    error_code="planning_model_unavailable",
                )
                raise

            if cancellation.is_set():
                self.state.transition_task(task_id, TaskStatus.CANCELLED)
                return self._cancelled_result(task_id)
            for step in plan.steps:
                self.state.add_step(task_id, step.step_id)
            self.state.transition_task(task_id, TaskStatus.READY)
            if cancellation.is_set():
                self.state.transition_task(task_id, TaskStatus.CANCELLED)
                return self._cancelled_result(task_id)
            self.state.transition_task(task_id, TaskStatus.RUNNING)
            try:
                results = asyncio.run(
                    self._executor.execute(
                        plan,
                        task_id,
                        self.state,
                        cancellation=cancellation,
                    )
                )
            except Exception:
                self.state.transition_task(task_id, TaskStatus.FAILED, error_code="execution_error")
                raise

            if cancellation.is_set():
                self.state.transition_task(task_id, TaskStatus.CANCELLED)
                return self._cancelled_result(task_id)
            if any(result.status is StepStatus.NEEDS_CONFIRMATION for result in results):
                self.state.transition_task(task_id, TaskStatus.NEEDS_CONFIRMATION)
                return AgentCoreResult(
                    "Tác vụ cần xác nhận rõ ràng trước khi thực hiện hành động.",
                    Intent.TASK,
                    "needs_confirmation",
                    task_id,
                    self._task_result(task_id, results),
                )
            if any(result.status is not StepStatus.SUCCEEDED for result in results):
                self.state.transition_task(task_id, TaskStatus.FAILED, error_code="step_failed")
                return AgentCoreResult(
                    "Tác vụ không hoàn thành; không báo cáo thành công.",
                    Intent.TASK,
                    "failed",
                    task_id,
                    self._task_result(task_id, results),
                )

            self.state.transition_task(task_id, TaskStatus.VALIDATING)
            validation = self._validator.validate(plan, results)
            if not validation.valid:
                self.state.transition_task(task_id, TaskStatus.FAILED, error_code="validation_failed")
                return AgentCoreResult(
                    "Kết quả tác vụ không vượt qua kiểm tra hợp lệ.",
                    Intent.TASK,
                    "failed",
                    task_id,
                    self._task_result(task_id, results),
                )
            self.state.transition_task(task_id, TaskStatus.COMPLETED)
            validated_outputs = {result.step_id: result.output for result in results}
            return AgentCoreResult(
                format_validated_task_outputs(validated_outputs),
                Intent.TASK,
                "ok",
                task_id,
                self._task_result(task_id, results, validated_outputs),
            )
        finally:
            with self._lock:
                self._active_cancellations.pop(task_id, None)

    def _cancelled_result(self, task_id: str) -> AgentCoreResult:
        return AgentCoreResult(
            "Tác vụ đã được hủy tại ranh giới an toàn.",
            Intent.TASK,
            "cancelled",
            task_id,
            self._task_result(task_id),
        )

    def _task_result(
        self,
        task_id: str,
        step_results: Sequence[StepResult] | None = None,
        validated_outputs: dict[str, dict[str, Any]] | None = None,
    ) -> TaskExecutionResult:
        snapshot = self.state.get_snapshot(task_id)
        return TaskExecutionResult(
            task_id=task_id,
            status=snapshot.status,
            step_results=list(step_results or ()),
            step_errors=snapshot.step_errors,
            validated_outputs=validated_outputs or {},
            error_code=snapshot.error_code,
        )
