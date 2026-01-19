"""
Agents package for V2 architecture.

Exports all agent classes:
- BaseAgent: Base class for all agents
- ResearchCoordinator: Parallel research orchestration
- HomeControlManager: Home automation with approval workflow
"""

from .base_agent import (
    BaseAgent,
    AgentExecutionError,
    AgentTimeoutError,
    AgentValidationError,
)
from .research_coordinator import ResearchCoordinator
from .home_control_manager import HomeControlManager

__all__ = [
    "BaseAgent",
    "AgentExecutionError",
    "AgentTimeoutError",
    "AgentValidationError",
    "ResearchCoordinator",
    "HomeControlManager",
]