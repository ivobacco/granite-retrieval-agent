"""
Base agent class for V2 architecture.

Provides common functionality for all specialized agents:
- Initialization with LLM config
- Event emission for status updates
- Error handling and logging
- Context access
"""

from typing import Optional, Dict, Any, Callable, Awaitable
from autogen import ConversableAgent
import logging


class BaseAgent:
    """
    Base class for all V2 agents.

    Provides common initialization, logging, and utility methods.
    All specialized agents (WebSearchAgent, KnowledgeAgent, etc.) inherit from this.

    Attributes:
        name: Agent name
        agent: AutoGen ConversableAgent instance
        logger: Logger instance
        emit_event: Optional event emitter for status updates
    """

    def __init__(
        self,
        name: str,
        system_message: str,
        llm_config: Dict[str, Any],
        emit_event: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        human_input_mode: str = "NEVER",
        **kwargs
    ):
        """
        Initialize base agent.

        Args:
            name: Agent name
            system_message: System prompt for the agent
            llm_config: LLM configuration dictionary
            emit_event: Optional event emitter for Open WebUI status updates
            human_input_mode: Human input mode (NEVER, TERMINATE, ALWAYS)
            **kwargs: Additional arguments passed to ConversableAgent
        """
        self.name = name
        self.emit_event = emit_event
        self.logger = logging.getLogger(f"{__name__}.{name}")

        # Create AutoGen agent
        self.agent = ConversableAgent(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            human_input_mode=human_input_mode,
            **kwargs
        )

        self.logger.info(f"Initialized {name} agent")

    async def emit_status(self, message: str, done: bool = False) -> None:
        """
        Emit status event to Open WebUI.

        Args:
            message: Status message to display
            done: Whether this status indicates completion
        """
        if self.emit_event:
            event = {
                "type": "status",
                "data": {
                    "description": f"[{self.name}] {message}",
                    "done": done
                }
            }
            try:
                await self.emit_event(event)
            except Exception as e:
                self.logger.error(f"Error emitting status: {e}")

    async def emit_message(self, content: str) -> None:
        """
        Emit message event to Open WebUI.

        Args:
            content: Message content to display
        """
        if self.emit_event:
            event = {
                "type": "message",
                "data": {"content": content}
            }
            try:
                await self.emit_event(event)
            except Exception as e:
                self.logger.error(f"Error emitting message: {e}")

    def log_info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(f"[{self.name}] {message}")

    def log_error(self, message: str, exception: Optional[Exception] = None) -> None:
        """Log error message"""
        if exception:
            self.logger.error(f"[{self.name}] {message}: {exception}")
        else:
            self.logger.error(f"[{self.name}] {message}")

    def log_debug(self, message: str) -> None:
        """Log debug message"""
        self.logger.debug(f"[{self.name}] {message}")

    async def handle_error(
        self,
        error: Exception,
        context: str = "",
        emit_to_user: bool = True
    ) -> Dict[str, Any]:
        """
        Handle errors with logging and optional user notification.

        Args:
            error: The exception that occurred
            context: Context string describing where the error occurred
            emit_to_user: Whether to emit error to user via status

        Returns:
            Error dictionary with type and message
        """
        error_msg = f"{context}: {str(error)}" if context else str(error)
        self.log_error(error_msg, error)

        if emit_to_user:
            await self.emit_status(f"Error: {error_msg}", done=True)

        return {
            "error": True,
            "error_type": type(error).__name__,
            "error_message": error_msg
        }


class AgentExecutionError(Exception):
    """Custom exception for agent execution errors"""
    pass


class AgentTimeoutError(Exception):
    """Custom exception for agent timeout errors"""
    pass


class AgentValidationError(Exception):
    """Custom exception for agent validation errors"""
    pass
