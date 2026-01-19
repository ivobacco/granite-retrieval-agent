"""
Task models for V2 architecture.

Defines Pydantic models for different task types used in the Triage with Tasks pattern:
- TaskBase: Base task model with common fields
- ResearchTask: Tasks that involve gathering information (web search, knowledge search)
- HomeControlTask: Tasks that control or query Home Assistant devices
- CompositeTask: Container for multiple subtasks with execution strategy
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import uuid


class TaskType(str, Enum):
    """Types of tasks that can be created"""
    RESEARCH = "research"
    HOME_CONTROL = "home_control"
    COMPOSITE = "composite"


class TaskPriority(str, Enum):
    """Priority levels for task execution"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(str, Enum):
    """Lifecycle states for tasks"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskBase(BaseModel):
    """
    Base task model with common fields for all task types.

    Attributes:
        task_id: Unique identifier for the task
        task_type: Type of task (research, home_control, composite)
        priority: Priority level for execution ordering
        status: Current lifecycle status
        context: Additional context data for task execution
        created_at: Timestamp when task was created
        updated_at: Timestamp when task was last updated
        error_message: Error message if task failed
    """
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: TaskType
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    context: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    error_message: Optional[str] = None

    class Config:
        """Pydantic configuration"""
        use_enum_values = True
        arbitrary_types_allowed = True


class ResearchTask(TaskBase):
    """
    Research task for information gathering from web and knowledge bases.

    This task type is used when the system needs to gather information from:
    - Web search (current events, statistics, online data)
    - Personal knowledge bases (user documents, notes)
    - Both sources (comprehensive research)

    Attributes:
        query: The research query or question
        sources: Which sources to query (web, knowledge, or both)
        max_results: Maximum number of results to retrieve per source
        synthesize: Whether to synthesize results from multiple sources
        results: Populated after execution with gathered information
    """
    task_type: TaskType = TaskType.RESEARCH
    query: str = Field(..., description="Research query or question to answer")
    sources: List[Literal["web", "knowledge"]] = Field(
        default=["web", "knowledge"],
        description="Research sources to query: 'web' for internet, 'knowledge' for personal documents"
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results per source"
    )
    synthesize: bool = Field(
        default=True,
        description="Whether to synthesize results from multiple sources into unified answer"
    )

    # Results (populated after execution)
    results: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Research results: {web: [...], knowledge: [...], synthesis: '...'}"
    )
    sources_used: List[str] = Field(
        default_factory=list,
        description="List of actual sources that returned results"
    )


class HomeControlTask(TaskBase):
    """
    Home automation control task for Home Assistant operations.

    This task type handles all Home Assistant interactions including:
    - Device discovery (finding lights, sensors, etc.)
    - State queries (getting current temperature, light status)
    - Control operations (turning devices on/off, adjusting settings)
    - Automation management (triggering scenes, managing automations)
    - History queries (retrieving historical data)

    Attributes:
        instruction: Natural language instruction for the operation
        operation: Structured operation from HA Planner (e.g., 'lights_control', 'search_entities')
        parameters: Operation parameters from HA Planner
        requires_approval: Whether user approval is required before execution
        approval_timeout: How long to wait for user approval (seconds)
        approval_status: Current approval status (pending/approved/denied)
        approval_timestamp: When approval was granted/denied
        execution_result: Result from MCP execution
    """
    task_type: TaskType = TaskType.HOME_CONTROL
    instruction: str = Field(
        ...,
        description="Natural language instruction (e.g., 'turn off bedroom lights', 'what is the temperature')"
    )
    operation: Optional[str] = Field(
        default=None,
        description="Structured operation from HA Planner (e.g., 'lights_control', 'search_entities')"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured parameters for the operation from HA Planner"
    )
    requires_approval: bool = Field(
        default=True,
        description="Whether this operation requires user approval before execution"
    )
    approval_timeout: int = Field(
        default=300,
        ge=10,
        le=3600,
        description="Approval timeout in seconds (10-3600)"
    )

    # Approval tracking
    approval_status: Optional[Literal["pending", "approved", "denied", "timeout"]] = None
    approval_timestamp: Optional[datetime] = None
    approval_message: Optional[str] = Field(
        default=None,
        description="Message to display to user when requesting approval"
    )

    # Execution results
    execution_result: Optional[Dict[str, Any]] = None
    mcp_response: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Raw MCP response for debugging"
    )


class CompositeTask(TaskBase):
    """
    Composite task containing multiple subtasks with execution strategy.

    Used for complex operations that require multiple steps, such as:
    - "Research thermostat best practices and set mine to optimal temperature"
      (ResearchTask → HomeControlTask)
    - "Find all motion sensors and check their battery levels"
      (HomeControlTask → HomeControlTask)

    Attributes:
        subtasks: List of subtasks (ResearchTask, HomeControlTask, or nested CompositeTask)
        execution_strategy: How to execute subtasks (sequential or parallel)
        aggregated_results: Combined results from all subtasks
        failed_subtasks: List of subtask IDs that failed
        success_rate: Percentage of subtasks that completed successfully
    """
    task_type: TaskType = TaskType.COMPOSITE
    subtasks: List[TaskBase] = Field(
        default_factory=list,
        description="List of subtasks to execute"
    )
    execution_strategy: Literal["sequential", "parallel"] = Field(
        default="sequential",
        description="Execution strategy: 'sequential' waits for each task, 'parallel' runs concurrently"
    )

    # Aggregated results
    aggregated_results: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Combined results from all subtasks"
    )
    failed_subtasks: List[str] = Field(
        default_factory=list,
        description="List of task IDs that failed during execution"
    )
    success_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Percentage of subtasks that completed successfully (0.0-1.0)"
    )

    def add_subtask(self, task: TaskBase) -> None:
        """Add a subtask to this composite task"""
        self.subtasks.append(task)
        self.updated_at = datetime.now()

    def get_subtasks_by_status(self, status: TaskStatus) -> List[TaskBase]:
        """Get all subtasks with a specific status"""
        return [task for task in self.subtasks if task.status == status]

    def calculate_success_rate(self) -> float:
        """Calculate success rate based on completed vs total subtasks"""
        if not self.subtasks:
            return 0.0

        completed = len(self.get_subtasks_by_status(TaskStatus.COMPLETED))
        total = len(self.subtasks)
        self.success_rate = completed / total
        return self.success_rate


class TaskDecompositionResult(BaseModel):
    """
    Result from task decomposition analysis by Triage Manager.

    The Triage Manager analyzes user queries and decomposes them into typed tasks.
    This model represents the output of that decomposition process.

    Attributes:
        original_query: The original user query
        intent_analysis: Analysis of user intent (information gathering, home control, mixed)
        tasks: List of decomposed tasks
        execution_order: Suggested execution order (usually research before control)
        estimated_duration: Estimated time to complete all tasks (seconds)
    """
    original_query: str = Field(..., description="Original user query")
    intent_analysis: str = Field(
        ...,
        description="Analysis of user intent: information_gathering, home_control, mixed"
    )
    tasks: List[TaskBase] = Field(
        default_factory=list,
        description="Decomposed tasks from the query"
    )
    execution_order: List[str] = Field(
        default_factory=list,
        description="Task IDs in suggested execution order"
    )
    estimated_duration: Optional[int] = Field(
        default=None,
        description="Estimated duration to complete all tasks (seconds)"
    )

    class Config:
        arbitrary_types_allowed = True
