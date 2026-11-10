from __future__ import annotations

import pytest

from neutrons.agent.schemas import StepStatus, TaskStatus
from neutrons.agent.state import TaskStateManager


def test_task_state_manager_tracks_valid_lifecycle_and_steps() -> None:
    manager = TaskStateManager()
    task_id = manager.create_task()

    manager.transition_task(task_id, TaskStatus.PLANNING)
    manager.add_step(task_id, "step_1")
    manager.transition_task(task_id, TaskStatus.READY)
    manager.transition_task(task_id, TaskStatus.RUNNING)
    manager.transition_step(task_id, "step_1", StepStatus.RUNNING)
    manager.transition_step(task_id, "step_1", StepStatus.SUCCEEDED)
    manager.transition_task(task_id, TaskStatus.VALIDATING)
    manager.transition_task(task_id, TaskStatus.COMPLETED)

    snapshot = manager.get_snapshot(task_id)
    assert snapshot.status is TaskStatus.COMPLETED
    assert snapshot.step_statuses == {"step_1": StepStatus.SUCCEEDED}
    assert snapshot.step_errors == {}
    assert snapshot.created_at <= snapshot.updated_at


def test_task_state_manager_rejects_invalid_and_terminal_transitions() -> None:
    manager = TaskStateManager()
    task_id = manager.create_task()

    with pytest.raises(ValueError, match="Invalid task transition"):
        manager.transition_task(task_id, TaskStatus.RUNNING)

    manager.transition_task(task_id, TaskStatus.CANCELLED)
    with pytest.raises(ValueError, match="Invalid task transition"):
        manager.transition_task(task_id, TaskStatus.PLANNING)


def test_task_cannot_complete_with_pending_steps() -> None:
    manager = TaskStateManager()
    task_id = manager.create_task()
    manager.transition_task(task_id, TaskStatus.PLANNING)
    manager.add_step(task_id, "step_1")
    manager.transition_task(task_id, TaskStatus.READY)
    manager.transition_task(task_id, TaskStatus.RUNNING)
    manager.transition_task(task_id, TaskStatus.VALIDATING)

    with pytest.raises(ValueError, match="all steps have succeeded"):
        manager.transition_task(task_id, TaskStatus.COMPLETED)


def test_task_without_steps_cannot_be_ready_or_completed() -> None:
    manager = TaskStateManager()
    task_id = manager.create_task()
    manager.transition_task(task_id, TaskStatus.PLANNING)

    with pytest.raises(ValueError, match="without steps"):
        manager.transition_task(task_id, TaskStatus.READY)


def test_cancellation_waits_for_running_step_boundary() -> None:
    manager = TaskStateManager()
    task_id = manager.create_task()
    manager.transition_task(task_id, TaskStatus.PLANNING)
    manager.add_step(task_id, "step_1")
    manager.transition_task(task_id, TaskStatus.READY)
    manager.transition_task(task_id, TaskStatus.RUNNING)
    manager.transition_step(task_id, "step_1", StepStatus.RUNNING)

    with pytest.raises(ValueError, match="safe boundary"):
        manager.transition_task(task_id, TaskStatus.CANCELLED)


def test_state_manager_capacity_is_bounded() -> None:
    manager = TaskStateManager(max_tasks=1)
    first = manager.create_task()
    manager.transition_task(first, TaskStatus.CANCELLED)

    second = manager.create_task()

    assert second != first
    with pytest.raises(KeyError, match="Unknown task ID"):
        manager.get_snapshot(first)
