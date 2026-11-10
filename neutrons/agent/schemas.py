from __future__ import annotations

from enum import Enum
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Intent(str, Enum):
    CHAT = "chat"
    TASK = "task"
    CLARIFICATION = "clarification"
    UNSUPPORTED = "unsupported"


class IntentDecision(Contract):
    intent: Intent
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(max_length=500)
    response: str | None = Field(default=None, max_length=20_000)


class PlanStep(Contract):
    step_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    action: str = Field(min_length=1, max_length=64)
    inputs: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list, max_length=8)


class ExecutionPlan(Contract):
    summary: str = Field(min_length=1, max_length=500)
    steps: list[PlanStep] = Field(min_length=1, max_length=8)


class TaskStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    READY = "ready"
    RUNNING = "running"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    NEEDS_CLARIFICATION = "needs_clarification"
    NEEDS_CONFIRMATION = "needs_confirmation"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMED_OUT = "timed_out"
    NEEDS_CONFIRMATION = "needs_confirmation"


class StepResult(Contract):
    step_id: str
    status: StepStatus
    output: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None


class ValidationResult(Contract):
    valid: bool
    errors: list[str] = Field(default_factory=list)


class TaskSnapshot(Contract):
    task_id: str
    status: TaskStatus
    step_statuses: dict[str, StepStatus] = Field(default_factory=dict)
    step_errors: dict[str, str] = Field(default_factory=dict)
    error_code: str | None = None
    created_at: datetime
    updated_at: datetime
