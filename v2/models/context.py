"""
Context models for tracking execution state across agent interactions.

These models implement the Context-Aware Routing pattern by maintaining
shared context that agents can read and update during task execution.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum
import uuid


class ExecutionPhase(str, Enum):
    """Phases of execution in the orchestration workflow"""
    INITIALIZATION = "initialization"
    TRIAGE = "triage"
    RESEARCH = "research"
    HOME_CONTROL = "home_control"
    AGGREGATION = "aggregation"
    COMPLETION = "completion"
    ERROR = "error"


class ContextVariables(BaseModel):
    """
    Context tracking across agent interactions.

    This class implements shared context for the Context-Aware Routing pattern,
    allowing agents to coordinate by reading and updating shared state.

    Attributes:
        user_query: Original user query
        query_timestamp: When the query was received
        conversation_id: Optional conversation ID for multi-turn interactions

        # Research context
        research_results: Aggregated research results from web/knowledge searches
        knowledge_retrieved: List of document titles/IDs retrieved from knowledge base
        web_sources: List of URLs from web search results

        # Home automation context
        home_state: Current state of relevant Home Assistant entities
        pending_approvals: List of task IDs waiting for approval
        approved_operations: List of task IDs that were approved
        denied_operations: List of task IDs that were denied

        # Execution tracking
        completed_tasks: List of completed task IDs
        failed_tasks: List of failed task IDs with error messages
        current_task_id: Task currently being executed

        # Provider context
        model_provider: LLM provider (ollama or openrouter)
        model_name: Model name being used
        temperature: Model temperature setting

        # Performance metrics
        total_tool_calls: Number of tool calls made
        execution_time: Total execution time in seconds
    """
    # Query context
    user_query: str
    query_timestamp: datetime = Field(default_factory=datetime.now)
    conversation_id: Optional[str] = None

    # Research context
    research_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Aggregated research results: {web: [...], knowledge: [...]}"
    )
    knowledge_retrieved: List[str] = Field(
        default_factory=list,
        description="List of document titles/IDs from knowledge base"
    )
    web_sources: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of web sources: [{title, url, snippet}]"
    )

    # Home automation context
    home_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current state of Home Assistant entities"
    )
    pending_approvals: List[str] = Field(
        default_factory=list,
        description="Task IDs waiting for user approval"
    )
    approved_operations: List[str] = Field(
        default_factory=list,
        description="Task IDs that were approved by user"
    )
    denied_operations: List[str] = Field(
        default_factory=list,
        description="Task IDs that were denied by user"
    )

    # Execution tracking
    completed_tasks: List[str] = Field(
        default_factory=list,
        description="Task IDs that completed successfully"
    )
    failed_tasks: Dict[str, str] = Field(
        default_factory=dict,
        description="Failed task IDs mapped to error messages"
    )
    current_task_id: Optional[str] = None

    # Provider context
    model_provider: str = "ollama"
    model_name: str = ""
    temperature: float = 0.0

    # Performance metrics
    total_tool_calls: int = 0
    execution_time: float = 0.0

    # Image context (for multimodal queries)
    image_descriptions: List[str] = Field(
        default_factory=list,
        description="Descriptions of images provided by user"
    )

    class Config:
        """Pydantic configuration"""
        arbitrary_types_allowed = True

    def mark_task_completed(self, task_id: str) -> None:
        """Mark a task as completed"""
        if task_id not in self.completed_tasks:
            self.completed_tasks.append(task_id)
        if task_id in self.failed_tasks:
            del self.failed_tasks[task_id]

    def mark_task_failed(self, task_id: str, error_message: str) -> None:
        """Mark a task as failed with error message"""
        self.failed_tasks[task_id] = error_message
        if task_id in self.completed_tasks:
            self.completed_tasks.remove(task_id)

    def request_approval(self, task_id: str) -> None:
        """Add task to pending approvals"""
        if task_id not in self.pending_approvals:
            self.pending_approvals.append(task_id)

    def approve_operation(self, task_id: str) -> None:
        """Mark operation as approved"""
        if task_id in self.pending_approvals:
            self.pending_approvals.remove(task_id)
        if task_id not in self.approved_operations:
            self.approved_operations.append(task_id)

    def deny_operation(self, task_id: str) -> None:
        """Mark operation as denied"""
        if task_id in self.pending_approvals:
            self.pending_approvals.remove(task_id)
        if task_id not in self.denied_operations:
            self.denied_operations.append(task_id)

    def increment_tool_calls(self, count: int = 1) -> None:
        """Increment tool call counter"""
        self.total_tool_calls += count

    def get_success_rate(self) -> float:
        """Calculate task success rate"""
        total = len(self.completed_tasks) + len(self.failed_tasks)
        if total == 0:
            return 0.0
        return len(self.completed_tasks) / total


class ExecutionContext(BaseModel):
    """
    Full execution context for orchestrator and agents.

    This is the master context object that gets passed through the entire
    orchestration workflow, from Triage Manager through agents to executors.

    Attributes:
        context_id: Unique identifier for this execution context
        variables: ContextVariables with all shared state
        tasks: List of all tasks (from decomposition)
        current_phase: Current execution phase
        phase_history: History of phase transitions

        # Event handling
        pending_events: Events waiting to be processed
        event_handlers: Registered event handlers

        # Streaming support
        stream_enabled: Whether streaming is enabled
        stream_buffer: Buffer for streaming chunks

        # Error handling
        errors: List of errors encountered during execution
        max_retries: Maximum number of retries for failed operations
        retry_count: Current retry count
    """
    context_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    variables: ContextVariables
    tasks: List[Any] = Field(
        default_factory=list,
        description="List of TaskBase objects from decomposition"
    )
    current_phase: ExecutionPhase = ExecutionPhase.INITIALIZATION
    phase_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of phase transitions: [{phase, timestamp, duration}]"
    )

    # Event handling
    pending_events: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Events waiting to be processed"
    )
    event_handlers: Dict[str, Any] = Field(
        default_factory=dict,
        description="Registered event handlers (not serialized)"
    )

    # Streaming support
    stream_enabled: bool = True
    stream_buffer: List[str] = Field(
        default_factory=list,
        description="Buffer for streaming chunks to Open WebUI"
    )

    # Error handling
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Errors encountered: [{phase, task_id, error_message, timestamp}]"
    )
    max_retries: int = 3
    retry_count: int = 0

    # Valves (passed from Pipe)
    valves: Optional[Any] = Field(
        default=None,
        description="Valves configuration from Pipe class"
    )

    # Event emitter (for Open WebUI status updates)
    event_emitter: Optional[Any] = Field(
        default=None,
        description="Event emitter function for status updates"
    )

    class Config:
        """Pydantic configuration"""
        arbitrary_types_allowed = True

    def transition_phase(self, new_phase: ExecutionPhase) -> None:
        """
        Transition to a new execution phase and record history.

        Args:
            new_phase: The phase to transition to
        """
        now = datetime.now()

        # Calculate duration of previous phase
        if self.phase_history:
            last_phase = self.phase_history[-1]
            last_timestamp = last_phase["timestamp"]
            duration = (now - last_timestamp).total_seconds()
            last_phase["duration"] = duration

        # Record new phase
        self.phase_history.append({
            "phase": new_phase,
            "timestamp": now,
            "duration": None  # Will be calculated on next transition
        })

        self.current_phase = new_phase

    def add_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Add an event to the pending events queue"""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now(),
            "processed": False
        }
        self.pending_events.append(event)

    def get_unprocessed_events(self) -> List[Dict[str, Any]]:
        """Get all unprocessed events"""
        return [e for e in self.pending_events if not e.get("processed")]

    def mark_event_processed(self, event: Dict[str, Any]) -> None:
        """Mark an event as processed"""
        event["processed"] = True

    def register_event_handler(
        self,
        event_type: str,
        handler: Callable
    ) -> None:
        """Register a handler for a specific event type"""
        self.event_handlers[event_type] = handler

    def add_error(
        self,
        phase: ExecutionPhase,
        error_message: str,
        task_id: Optional[str] = None
    ) -> None:
        """Record an error"""
        error = {
            "phase": phase,
            "task_id": task_id,
            "error_message": error_message,
            "timestamp": datetime.now()
        }
        self.errors.append(error)

    def stream_chunk(self, chunk: str) -> None:
        """Add a chunk to the stream buffer"""
        if self.stream_enabled:
            self.stream_buffer.append(chunk)

    def get_phase_duration(self, phase: ExecutionPhase) -> Optional[float]:
        """Get the duration of a specific phase"""
        for record in self.phase_history:
            if record["phase"] == phase and record["duration"] is not None:
                return record["duration"]
        return None

    def get_total_execution_time(self) -> float:
        """Calculate total execution time across all phases"""
        if not self.phase_history:
            return 0.0

        total = sum(
            record["duration"]
            for record in self.phase_history
            if record["duration"] is not None
        )
        return total

    def to_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the execution context for logging/debugging.

        Returns:
            Dictionary with key metrics and state
        """
        return {
            "context_id": self.context_id,
            "current_phase": self.current_phase,
            "total_tasks": len(self.tasks),
            "completed_tasks": len(self.variables.completed_tasks),
            "failed_tasks": len(self.variables.failed_tasks),
            "success_rate": self.variables.get_success_rate(),
            "total_tool_calls": self.variables.total_tool_calls,
            "execution_time": self.get_total_execution_time(),
            "pending_approvals": len(self.variables.pending_approvals),
            "errors": len(self.errors),
            "retry_count": self.retry_count
        }


class AgentMessage(BaseModel):
    """
    Message format for agent-to-agent communication.

    Used for structured communication between agents in the orchestration workflow.
    """
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = Field(..., description="Agent name that sent the message")
    recipient: str = Field(..., description="Agent name that should receive the message")
    message_type: str = Field(
        ...,
        description="Message type: request, response, notification, error"
    )
    content: Dict[str, Any] = Field(
        default_factory=dict,
        description="Message content"
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    in_reply_to: Optional[str] = Field(
        default=None,
        description="Message ID this is replying to"
    )

    class Config:
        arbitrary_types_allowed = True
