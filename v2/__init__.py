"""
Granite Retrieval Agent V2 - AG2 Advanced Patterns Implementation

Main exports:
- GraniteOrchestrator: SocietyOfMindAgent wrapper
- TriageManager: Task decomposition and routing
- All models, agents, executors, and config utilities
"""

# Orchestrator
from .orchestrator import GraniteOrchestrator, TriageManager

# Models
from .models import (
    # Task models
    TaskBase,
    ResearchTask,
    HomeControlTask,
    CompositeTask,
    TaskDecompositionResult,
    # Task enums
    TaskType,
    TaskPriority,
    TaskStatus,
    # Context models
    ContextVariables,
    ExecutionContext,
    AgentMessage,
    ExecutionPhase,
)

# Agents
from .agents import (
    BaseAgent,
    ResearchCoordinator,
    HomeControlManager,
    AgentExecutionError,
    AgentTimeoutError,
    AgentValidationError,
)

# Executors
from .executors import HomeAssistantExecutor

# Config
from .config import (
    ProviderManager,
    get_agent_configs_for_orchestrator,
)

# Prompts
from .prompts import (
    PLANNER_MESSAGE,
    ASSISTANT_PROMPT,
    HA_PLANNER_PROMPT,
    REPORT_WRITER_PROMPT,
)

__version__ = "2.0.0"

__all__ = [
    # Orchestrator
    "GraniteOrchestrator",
    "TriageManager",
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
    "ContextVariables",
    "ExecutionContext",
    "AgentMessage",
    "ExecutionPhase",
    # Agents
    "BaseAgent",
    "ResearchCoordinator",
    "HomeControlManager",
    "AgentExecutionError",
    "AgentTimeoutError",
    "AgentValidationError",
    # Executors
    "HomeAssistantExecutor",
    # Config
    "ProviderManager",
    "get_agent_configs_for_orchestrator",
    # Prompts
    "PLANNER_MESSAGE",
    "ASSISTANT_PROMPT",
    "HA_PLANNER_PROMPT",
    "REPORT_WRITER_PROMPT",
]
