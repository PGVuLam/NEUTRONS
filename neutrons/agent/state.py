from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from uuid import uuid4

from neutrons.agent.schemas import StepStatus, TaskSnapshot, TaskStatus

_TASK_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.PENDING: frozenset(
        {TaskStatus.PLANNING, TaskStatus.CANCELLED, TaskStatus.NEEDS_CLARIFICATION}
    ),
    TaskStatus.PLANNING: frozenset(
        {
            TaskStatus.READY,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.NEEDS_CLARIFICATION,
        }
    ),
    TaskStatus.READY: frozenset({TaskStatus.RUNNING, TaskStatus.CANCELLED, TaskStatus.FAILED}),
    TaskStatus.RUNNING: frozenset(
        {
            TaskStatus.VALIDATING,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.NEEDS_CONFIRMATION,
        }
    ),
    TaskStatus.VALIDATING: frozenset({TaskStatus.COMPLETED, TaskStatus.FAILED}),
    TaskStatus.NEEDS_CONFIRMATION: frozenset({TaskStatus.CANCELLED}),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.FAILED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
    TaskStatus.NEEDS_CLARIFICATION: frozenset(),
}

_STEP_TRANSITIONS: dict[StepStatus, frozenset[StepStatus]] = {
    StepStatus.PENDING: frozenset({StepStatus.RUNNING, StepStatus.SKIPPED}),
    StepStatus.RUNNING: frozenset(
        {
            StepStatus.SUCCEEDED,
            StepStatus.FAILED,
            StepStatus.TIMED_OUT,
            StepStatus.NEEDS_CONFIRMATION,
        }
    ),
    StepStatus.SUCCEEDED: frozenset(),
    StepStatus.FAILED: frozenset(),
    StepStatus.SKIPPED: frozenset(),
    StepStatus.TIMED_OUT: frozenset(),
    StepStatus.NEEDS_CONFIRMATION: frozenset(),
}


@dataclass
class _TaskRecord:
    status: TaskStatus = TaskStatus.PENDING
    step_statuses: dict[str, StepStatus] = field(default_factory=dict)
    step_errors: dict[str, str] = field(default_factory=dict)
    error_code: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TaskStateManager:
    """Bounded, process-local task state with validated transitions."""

    def __init__(self, max_tasks: int = 100) -> None:
        if max_tasks < 1:
            raise ValueError("max_tasks must be at least 1")
        self._max_tasks = max_tasks
        self._tasks: dict[str, _TaskRecord] = {}
        self._lock = RLock()

    def create_task(self) -> str:
        with self._lock:
            if len(self._tasks) >= self._max_tasks:
                terminal_ids = [
                    task_id
                    for task_id, record in self._tasks.items()
                    if record.status in _TERMINAL_TASKS
                ]
                if not terminal_ids:
                    raise RuntimeError("Task state capacity is full")
                oldest_id = min(
                    terminal_ids, key=lambda task_id: self._tasks[task_id].created_at
                )
                del self._tasks[oldest_id]
            task_id = str(uuid4())
            self._tasks[task_id] = _TaskRecord()
            return task_id

    def transition_task(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        error_code: str | None = None,
    ) -> None:
        with self._lock:
            record = self._get(task_id)
            if status not in _TASK_TRANSITIONS[record.status]:
                raise ValueError(f"Invalid task transition: {record.status} -> {status}")
            if status is TaskStatus.COMPLETED and any(
                step_status is not StepStatus.SUCCEEDED
                for step_status in record.step_statuses.values()
            ):
                raise ValueError("A task cannot complete until all steps have succeeded")
            if status is TaskStatus.COMPLETED and not record.step_statuses:
                raise ValueError("A task without steps cannot complete")
            if status is TaskStatus.READY and not record.step_statuses:
                raise ValueError("A task without steps cannot become ready")
            if status is TaskStatus.CANCELLED and any(
                step_status is StepStatus.RUNNING
                for step_status in record.step_statuses.values()
            ):
                raise ValueError("A running step must reach a safe boundary before cancellation")
            if status is TaskStatus.CANCELLED:
                record.step_statuses = {
                    step_id: (
                        StepStatus.SKIPPED
                        if step_status is StepStatus.PENDING
                        else step_status
                    )
                    for step_id, step_status in record.step_statuses.items()
                }
            if status is TaskStatus.FAILED:
                record.step_statuses = {
                    step_id: (
                        StepStatus.FAILED
                        if step_status is StepStatus.RUNNING
                        else StepStatus.SKIPPED
                        if step_status is StepStatus.PENDING
                        else step_status
                    )
                    for step_id, step_status in record.step_statuses.items()
                }
                if error_code is not None:
                    for step_id, step_status in record.step_statuses.items():
                        if step_status is StepStatus.FAILED:
                            record.step_errors.setdefault(step_id, error_code)
            record.status = status
            record.error_code = error_code
            record.updated_at = datetime.now(timezone.utc)

    def add_step(self, task_id: str, step_id: str) -> None:
        with self._lock:
            record = self._get(task_id)
            if record.status is not TaskStatus.PLANNING:
                raise ValueError("Steps can only be added while a task is planning")
            if step_id in record.step_statuses:
                raise ValueError("Step IDs must be unique within a task")
            record.step_statuses[step_id] = StepStatus.PENDING
            record.updated_at = datetime.now(timezone.utc)

    def transition_step(
        self,
        task_id: str,
        step_id: str,
        status: StepStatus,
        *,
        error_code: str | None = None,
    ) -> None:
        with self._lock:
            record = self._get(task_id)
            if record.status is not TaskStatus.RUNNING:
                raise ValueError("Steps can only transition while a task is running")
            current = record.step_statuses.get(step_id)
            if current is None:
                raise KeyError(f"Unknown step ID: {step_id}")
            if status not in _STEP_TRANSITIONS[current]:
                raise ValueError(f"Invalid step transition: {current} -> {status}")
            record.step_statuses[step_id] = status
            if error_code is not None:
                record.step_errors[step_id] = error_code
            record.updated_at = datetime.now(timezone.utc)

    def get_snapshot(self, task_id: str) -> TaskSnapshot:
        with self._lock:
            record = self._get(task_id)
            return TaskSnapshot(
                task_id=task_id,
                status=record.status,
                step_statuses=dict(record.step_statuses),
                step_errors=dict(record.step_errors),
                error_code=record.error_code,
                created_at=record.created_at,
                updated_at=record.updated_at,
            )

    def _get(self, task_id: str) -> _TaskRecord:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise KeyError(f"Unknown task ID: {task_id}") from exc


_TERMINAL_TASKS = frozenset(
    {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.NEEDS_CLARIFICATION,
    }
)
