"""
Configuration package for V2 architecture.

Exports provider management and configuration utilities:
- ProviderManager: LLM provider configuration manager
- get_agent_configs_for_orchestrator: Generate AG2 agent configs from valves
"""

from .providers import (
    ProviderConfig,
    OllamaConfig,
    OpenRouterConfig,
    ProviderManager,
    create_structured_output_config,
    get_agent_configs_for_orchestrator,
)

__all__ = [
    "ProviderConfig",
    "OllamaConfig",
    "OpenRouterConfig",
    "ProviderManager",
    "create_structured_output_config",
    "get_agent_configs_for_orchestrator",
]
