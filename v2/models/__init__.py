"""
Models package for V2 architecture.

Exports task and context models:
- Task models: TaskBase, ResearchTask, HomeControlTask, CompositeTask
- Task enums: TaskType, TaskPriority, TaskStatus
- Context models: ContextVariables, ExecutionContext, AgentMessage, ExecutionPhase
"""

from .tasks import (
    TaskType,
    TaskPriority,
    TaskStatus,
    TaskBase,
    ResearchTask,
    HomeControlTask,
    CompositeTask,
    TaskDecompositionResult,
)

from .context import (
    ExecutionPhase,
    ContextVariables,
    ExecutionContext,
    AgentMessage,
)

__all__ = [
    # Task models
    "TaskBase",
    "ResearchTask",
    "HomeControlTask",
    "CompositeTask",
    "TaskDecompositionResult",
    # Task enums
    "TaskType",
    "TaskPriority",
    "TaskStatus",
    # Context models
    "ExecutionPhase",
    "ContextVariables",
    "ExecutionContext",
    "AgentMessage",
]
