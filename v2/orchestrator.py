"""
Orchestrator for V2 architecture.

Implements the main orchestration logic:
- GraniteOrchestrator: SocietyOfMindAgent wrapper
- TriageManager: Task decomposition and routing
- Context tracking and phase management
"""

from typing import AsyncGenerator, List, Dict, Any, Optional
from autogen import ConversableAgent
import json
import logging

from .models import (
    ExecutionContext,
    ContextVariables,
    ExecutionPhase,
    ResearchTask,
    HomeControlTask,
    TaskType,
    TaskPriority,
    TaskDecompositionResult,
)
from .prompts import PLANNER_MESSAGE, ASSISTANT_PROMPT, HA_PLANNER_PROMPT
from .agents.research_coordinator import ResearchCoordinator
from .agents.home_control_manager import HomeControlManager
from .agents.base_agent import BaseAgent
from .executors.homeassistant_executor import HomeAssistantExecutor


class TriageManager:
    """
    Task decomposition and routing manager.

    Implements the Triage with Tasks pattern:
    - Analyzes user query
    - Decomposes into typed tasks (ResearchTask, HomeControlTask)
    - Routes to appropriate coordinators
    - Enforces sequential execution (research → home control)
    """

    def __init__(
        self,
        planner_agent: ConversableAgent,
        user_proxy: ConversableAgent,
        research_coordinator: ResearchCoordinator,
        home_control_manager: HomeControlManager,
    ):
        """
        Initialize triage manager.

        Args:
            planner_agent: AG2 agent for creating initial plan
            user_proxy: AG2 user proxy for tool execution
            research_coordinator: Research coordinator instance
            home_control_manager: Home control manager instance
        """
        self.planner = planner_agent
        self.user_proxy = user_proxy
        self.research_coordinator = research_coordinator
        self.home_control_manager = home_control_manager
        self.logger = logging.getLogger(__name__)

    async def decompose(
        self,
        query: str,
        context: ExecutionContext
    ) -> TaskDecompositionResult:
        """
        Decompose user query into typed tasks.

        Args:
            query: User query
            context: Execution context

        Returns:
            TaskDecompositionResult with decomposed tasks
        """
        context.transition_phase(ExecutionPhase.TRIAGE)
        self.logger.info(f"Decomposing query: {query}")

        try:
            # Get plan from planner agent
            planner_response = await self.user_proxy.a_initiate_chat(
                recipient=self.planner,
                message=f"Gather enough data to accomplish the goal: {query}",
                max_turns=1
            )

            plan_content = planner_response.chat_history[-1]["content"]
            plan_dict = json.loads(plan_content)
            steps = plan_dict.get("steps", [])

            # Analyze intent and create typed tasks
            intent = self._analyze_intent(steps)
            tasks = self._create_tasks_from_steps(steps, query)

            # Determine execution order (research before control)
            execution_order = self._determine_execution_order(tasks)

            result = TaskDecompositionResult(
                original_query=query,
                intent_analysis=intent,
                tasks=tasks,
                execution_order=execution_order
            )

            # Store tasks in context
            context.tasks = tasks

            self.logger.info(
                f"Decomposed into {len(tasks)} tasks: {len([t for t in tasks if isinstance(t, ResearchTask)])} research, "
                f"{len([t for t in tasks if isinstance(t, HomeControlTask)])} control"
            )

            return result

        except Exception as e:
            self.logger.error(f"Task decomposition failed: {e}")
            context.add_error(ExecutionPhase.TRIAGE, str(e))
            raise

    def _analyze_intent(self, steps: List[str]) -> str:
        """Analyze intent from plan steps"""
        has_research = any(
            keyword in step.lower()
            for step in steps
            for keyword in ["search", "find", "gather", "fetch", "retrieve"]
        )

        has_control = any(
            keyword in step.lower()
            for step in steps
            for keyword in ["turn", "set", "adjust", "activate", "control", "start", "stop"]
        )

        if has_research and has_control:
            return "mixed"
        elif has_research:
            return "information_gathering"
        elif has_control:
            return "home_control"
        else:
            return "unknown"

    def _create_tasks_from_steps(
        self,
        steps: List[str],
        original_query: str
    ) -> List:
        """Create typed tasks from plan steps"""
        tasks = []

        for step in steps:
            step_lower = step.lower()

            # Determine if this is a research or control step
            if any(kw in step_lower for kw in ["search", "find", "gather", "fetch", "retrieve"]):
                # Research task
                sources = []
                if "internet" in step_lower or "web" in step_lower or "online" in step_lower:
                    sources.append("web")
                if "document" in step_lower or "knowledge" in step_lower or "local" in step_lower:
                    sources.append("knowledge")

                # Default to both if not specified
                if not sources:
                    sources = ["web", "knowledge"]

                task = ResearchTask(
                    query=step,
                    sources=sources,
                    priority=TaskPriority.HIGH
                )
                tasks.append(task)

            elif any(kw in step_lower for kw in ["turn", "set", "adjust", "activate", "control", "start", "stop", "temperature", "light", "thermostat"]):
                # Home control task
                task = HomeControlTask(
                    instruction=step,
                    requires_approval=True,  # Default to requiring approval
                    priority=TaskPriority.HIGH
                )
                tasks.append(task)

        return tasks

    def _determine_execution_order(self, tasks: List) -> List[str]:
        """Determine execution order: research BEFORE control"""
        research_ids = [t.task_id for t in tasks if isinstance(t, ResearchTask)]
        control_ids = [t.task_id for t in tasks if isinstance(t, HomeControlTask)]

        # Research tasks first, then control tasks
        return research_ids + control_ids

    async def execute_sequential(
        self,
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """
        Execute tasks in sequential phases.

        Phase 1: Research (parallel within phase)
        Phase 2: Home Control (sequential with approval)

        Args:
            context: Execution context with tasks

        Returns:
            Aggregated results
        """
        results = {}

        # Separate tasks by type
        research_tasks = [t for t in context.tasks if isinstance(t, ResearchTask)]
        control_tasks = [t for t in context.tasks if isinstance(t, HomeControlTask)]

        # Phase 1: Research (parallel execution)
        if research_tasks:
            context.transition_phase(ExecutionPhase.RESEARCH)
            self.logger.info(f"Executing {len(research_tasks)} research tasks")

            research_results = await self.research_coordinator.execute_parallel(
                research_tasks,
                context
            )
            results["research"] = research_results

        # Phase 2: Home Control (sequential execution with approval)
        if control_tasks:
            context.transition_phase(ExecutionPhase.HOME_CONTROL)
            self.logger.info(f"Executing {len(control_tasks)} control tasks")

            control_results = await self.home_control_manager.execute_with_approval(
                control_tasks,
                context
            )
            results["home_control"] = control_results

        context.transition_phase(ExecutionPhase.AGGREGATION)
        return results


class GraniteOrchestrator:
    """
    Main orchestrator for V2 architecture.

    Implements SocietyOfMindAgent pattern:
    - Clean external interface
    - Internal multi-agent coordination
    - Streaming support for Open WebUI
    """

    def __init__(
        self,
        valves: Any,
        llm_configs: Dict[str, Any],
        user_proxy: ConversableAgent,
        assistant: ConversableAgent,
        planner: ConversableAgent,
        homeassistant_planner: ConversableAgent,
        emit_event: Optional[Any] = None
    ):
        """
        Initialize orchestrator.

        Args:
            valves: Valves configuration from Pipe
            llm_configs: LLM configurations for agents
            user_proxy: User proxy agent with tools registered
            assistant: Research assistant agent
            planner: Planner agent for task decomposition
            homeassistant_planner: HA planner for MCP operations
            emit_event: Event emitter for Open WebUI
        """
        self.valves = valves
        self.llm_configs = llm_configs
        self.user_proxy = user_proxy
        self.assistant = assistant
        self.planner = planner
        self.homeassistant_planner = homeassistant_planner
        self.emit_event = emit_event
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize coordinators and managers"""
        # Create base agents for research
        web_agent = BaseAgent(
            name="WebSearchAgent",
            system_message="Web search specialist",
            llm_config=self.llm_configs["assistant_llm_config"],
            emit_event=self.emit_event
        )

        knowledge_agent = BaseAgent(
            name="KnowledgeAgent",
            system_message="Personal knowledge specialist",
            llm_config=self.llm_configs["assistant_llm_config"],
            emit_event=self.emit_event
        )

        # Research coordinator
        self.research_coordinator = ResearchCoordinator(
            web_agent=web_agent,
            knowledge_agent=knowledge_agent,
            emit_event=self.emit_event
        )

        # Home Assistant executor
        self.ha_executor = HomeAssistantExecutor(
            mcp_url=self.valves.HOMEASSISTANT_MCP_URL,
            mcp_enabled=self.valves.HOMEASSISTANT_MCP_ENABLED
        )

        # Home control manager
        self.home_control_manager = HomeControlManager(
            homeassistant_executor=self.ha_executor,
            emit_event=self.emit_event
        )

        # Triage manager
        self.triage_manager = TriageManager(
            planner_agent=self.planner,
            user_proxy=self.user_proxy,
            research_coordinator=self.research_coordinator,
            home_control_manager=self.home_control_manager
        )

    async def process(
        self,
        query: str,
        context: ExecutionContext,
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Process user query through V2 orchestration.

        Args:
            query: User query
            context: Execution context
            stream: Whether to stream results

        Yields:
            Response chunks for Open WebUI
        """
        try:
            context.transition_phase(ExecutionPhase.INITIALIZATION)
            self.logger.info(f"Processing query: {query}")

            # Step 1: Decompose into tasks
            await self._emit_status("Analyzing request and creating plan...")
            decomposition = await self.triage_manager.decompose(query, context)

            # Step 2: Execute tasks in phases
            await self._emit_status("Executing tasks...")
            results = await self.triage_manager.execute_sequential(context)

            # Step 3: Format response
            context.transition_phase(ExecutionPhase.COMPLETION)
            await self._emit_status("Preparing response...")
            response = await self._format_response(query, results, context)

            # Step 4: Stream or return
            if stream:
                async for chunk in self._stream_response(response):
                    yield chunk
            else:
                yield response

            self.logger.info(f"Processing complete. Summary: {context.to_summary()}")

        except Exception as e:
            self.logger.error(f"Orchestration error: {e}")
            context.transition_phase(ExecutionPhase.ERROR)
            context.add_error(ExecutionPhase.ERROR, str(e))

            error_message = f"Error processing request: {str(e)}"
            if stream:
                async for chunk in self._stream_response(error_message):
                    yield chunk
            else:
                yield error_message

    async def _format_response(
        self,
        query: str,
        results: Dict[str, Any],
        context: ExecutionContext
    ) -> str:
        """Format final response from results"""
        # Extract research results
        research_info = results.get("research", {})
        control_info = results.get("home_control", {})

        # Build response
        response_parts = []

        if research_info:
            research_data = research_info.get("results", {})
            if research_data:
                response_parts.append("## Research Findings\n")
                for task_id, task_result in research_data.items():
                    if "web" in task_result:
                        response_parts.append(f"- Web search: {task_result['web']}\n")
                    if "knowledge" in task_result:
                        response_parts.append(f"- Knowledge base: {task_result['knowledge']}\n")

        if control_info:
            control_results = control_info.get("results", {})
            if control_results:
                response_parts.append("\n## Home Automation Actions\n")
                for task_id, task_result in control_results.items():
                    if task_result.get("success"):
                        response_parts.append(f"- ✓ Operation completed successfully\n")
                    elif task_result.get("error"):
                        response_parts.append(f"- ✗ Operation failed: {task_result.get('error_message')}\n")

        if not response_parts:
            response_parts.append("Task completed. No results to display.")

        return "".join(response_parts)

    async def _stream_response(self, response: str) -> AsyncGenerator[str, None]:
        """Stream response in chunks"""
        chunk_size = 50
        for i in range(0, len(response), chunk_size):
            chunk = response[i:i + chunk_size]
            yield chunk

    async def _emit_status(self, message: str, done: bool = False):
        """Emit status to Open WebUI"""
        if self.emit_event:
            try:
                await self.emit_event({
                    "type": "status",
                    "data": {"description": message, "done": done}
                })
            except Exception as e:
                self.logger.error(f"Error emitting status: {e}")
