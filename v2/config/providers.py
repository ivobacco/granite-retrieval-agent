"""
Provider configuration utilities for V2 architecture.

Handles configuration for both Ollama (local) and OpenRouter (cloud) providers.
Extracted from V1 Pipe class for better separation of concerns.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import logging


class ProviderConfig(BaseModel):
    """Base provider configuration"""
    provider_type: str = Field(..., description="Provider type: ollama or openrouter")
    model_name: str = Field(..., description="Model name/ID")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    base_url: Optional[str] = None
    api_key: Optional[str] = None


class OllamaConfig(ProviderConfig):
    """Ollama (local) provider configuration"""
    provider_type: str = "ollama"
    client_host: str = Field(default="http://localhost:11434")
    num_ctx: int = Field(default=131072, description="Context window size")


class OpenRouterConfig(ProviderConfig):
    """OpenRouter (cloud) provider configuration"""
    provider_type: str = "openrouter"
    base_url: str = Field(default="https://openrouter.ai/api/v1")
    api_key: str = Field(..., description="OpenRouter API key (sk-or-v1-...)")


class ProviderManager:
    """
    Manager for LLM provider configurations.

    Handles configuration for different agents (planner, assistant, etc.)
    with support for both Ollama and OpenRouter providers.
    """

    def __init__(self, valves: Any):
        """
        Initialize provider manager from Valves configuration.

        Args:
            valves: Valves object from Pipe class
        """
        self.valves = valves
        self.use_openrouter = valves.USE_OPENROUTER
        self.logger = logging.getLogger(__name__)

    def get_base_config(self) -> ProviderConfig:
        """Get base provider configuration"""
        if self.use_openrouter:
            if not self.valves.OPENROUTER_API_KEY:
                raise ValueError("OPENROUTER_API_KEY must be set when USE_OPENROUTER is enabled")

            return OpenRouterConfig(
                model_name=self.valves.OPENROUTER_TASK_MODEL,
                base_url=self.valves.OPENROUTER_BASE_URL,
                api_key=self.valves.OPENROUTER_API_KEY,
                temperature=self.valves.MODEL_TEMPERATURE
            )
        else:
            return OllamaConfig(
                model_name=self.valves.TASK_MODEL_ID,
                client_host=self.valves.OPENAI_API_URL,
                temperature=self.valves.MODEL_TEMPERATURE
            )

    def get_quick_model_config(self) -> Dict[str, Any]:
        """
        Get configuration for quick/lightweight operations.

        Used for: assistant, planner, reflection, search query generation, HA planner
        """
        if self.use_openrouter:
            return {
                "model": "nvidia/nemotron-3-nano-30b-a3b:free",
                "base_url": self.valves.OPENROUTER_BASE_URL,
                "api_type": "openai",
                "api_key": self.valves.OPENROUTER_API_KEY,
                "temperature": self.valves.MODEL_TEMPERATURE,
            }
        else:
            return {
                "model": self.valves.TASK_MODEL_ID,
                "client_host": self.valves.OPENAI_API_URL,
                "api_type": "openai",
                "temperature": self.valves.MODEL_TEMPERATURE,
                "num_ctx": 131072,
            }

    def get_vision_config(self) -> Dict[str, Any]:
        """Get configuration for vision/multimodal models"""
        if self.use_openrouter:
            return {
                "model": self.valves.OPENROUTER_VISION_MODEL,
                "base_url": self.valves.OPENROUTER_BASE_URL,
                "api_type": "openai",
                "api_key": self.valves.OPENROUTER_API_KEY,
            }
        else:
            return {
                "model": self.valves.VISION_MODEL_ID,
                "base_url": self.valves.VISION_API_URL,
                "api_type": "openai",
                "api_key": self.valves.OPENAI_API_KEY,
            }

    def build_llm_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Build all LLM configurations for different agent types.

        Returns:
            Dictionary mapping agent types to their LLM configurations
        """
        base_config = self.get_base_config()
        quick_config = self.get_quick_model_config()
        vision_config = self.get_vision_config()

        # Convert to AG2-compatible format
        if self.use_openrouter:
            base_llm = {
                "model": base_config.model_name,
                "base_url": base_config.base_url,
                "api_type": "openai",
                "api_key": base_config.api_key,
                "temperature": base_config.temperature,
            }
        else:
            base_llm = {
                "model": base_config.model_name,
                "client_host": base_config.base_url,
                "api_type": "openai",
                "temperature": base_config.temperature,
                "num_ctx": 131072,
            }

        return {
            "ollama_llm_config": {
                **base_llm,
                "config_list": [base_llm]
            },
            "assistant_llm_config": {
                **base_llm,
                "config_list": [quick_config]
            },
            "planner_llm_config": {
                **base_llm,
                "config_list": [quick_config]
            },
            "critic_llm_config": {
                **base_llm,
                "config_list": [base_llm]
            },
            "reflection_llm_config": {
                **base_llm,
                "config_list": [quick_config]
            },
            "search_query_llm_config": {
                **base_llm,
                "config_list": [quick_config]
            },
            "ha_planner_llm_config": {
                **base_llm,
                "config_list": [quick_config]
            },
            "vision_llm_config": {
                "config_list": [vision_config]
            },
        }

    def log_configuration(self) -> None:
        """Log current provider configuration"""
        provider = "OpenRouter (Cloud)" if self.use_openrouter else "Ollama (Local)"
        base_config = self.get_base_config()

        self.logger.info(f"Provider: {provider}")
        self.logger.info(f"Base Model: {base_config.model_name}")
        self.logger.info(f"Temperature: {base_config.temperature}")

        if self.use_openrouter:
            self.logger.info(f"Base URL: {base_config.base_url}")
        else:
            self.logger.info(f"Client Host: {base_config.base_url}")


def create_structured_output_config(
    base_config: Dict[str, Any],
    response_format: Any
) -> Dict[str, Any]:
    """
    Create LLM config with structured output (Pydantic model).

    Args:
        base_config: Base LLM configuration
        response_format: Pydantic model for structured output

    Returns:
        Updated configuration with response_format
    """
    config = base_config.copy()
    config["config_list"] = [
        {**config["config_list"][0], "response_format": response_format}
    ]
    return config


# Example usage for V2 agents
def get_agent_configs_for_orchestrator(valves: Any) -> Dict[str, Any]:
    """
    Get all agent configurations for orchestrator initialization.

    This is the main entry point for V2 architecture to get
    all necessary LLM configurations.

    Args:
        valves: Valves configuration from Pipe

    Returns:
        Dictionary with all agent configurations
    """
    manager = ProviderManager(valves)
    manager.log_configuration()
    return manager.build_llm_configs()
