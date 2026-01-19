"""
Research Coordinator agent for V2 architecture.

Orchestrates parallel research execution across multiple sources:
- Web search agent
- Personal knowledge agent
- Result synthesis and aggregation
"""

from typing import List, Dict, Any, Optional, Callable, Awaitable
import asyncio
from ..models import ResearchTask, TaskStatus, ExecutionContext
from .base_agent import BaseAgent
import logging


class ResearchCoordinator:
    """
    Coordinates parallel research tasks across multiple agents.

    This coordinator implements the parallel execution pattern for research phase:
    - Routes tasks to appropriate agents (web, knowledge)
    - Executes searches in parallel when possible
    - Aggregates and synthesizes results
    - Updates execution context with findings
    """

    def __init__(
        self,
        web_agent: BaseAgent,
        knowledge_agent: BaseAgent,
        emit_event: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
    ):
        """
        Initialize research coordinator.

        Args:
            web_agent: Web search agent instance
            knowledge_agent: Personal knowledge agent instance
            emit_event: Optional event emitter for status updates
        """
        self.web_agent = web_agent
        self.knowledge_agent = knowledge_agent
        self.emit_event = emit_event
        self.logger = logging.getLogger(__name__)

    async def emit_status(self, message: str, done: bool = False) -> None:
        """Emit status to Open WebUI"""
        if self.emit_event:
            try:
                await self.emit_event({
                    "type": "status",
                    "data": {"description": f"[Research] {message}", "done": done}
                })
            except Exception as e:
                self.logger.error(f"Error emitting status: {e}")

    async def execute_parallel(
        self,
        tasks: List[ResearchTask],
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """
        Execute research tasks in parallel.

        Args:
            tasks: List of ResearchTask objects
            context: Execution context for state tracking

        Returns:
            Aggregated research results: {
                "tasks_completed": int,
                "tasks_failed": int,
                "results": {task_id: {...}},
                "synthesis": str (if multiple tasks with synthesize=True)
            }
        """
        if not tasks:
            return {"tasks_completed": 0, "tasks_failed": 0, "results": {}}

        await self.emit_status(f"Starting parallel research for {len(tasks)} tasks")

        # Execute all tasks in parallel
        task_coroutines = [
            self._execute_single_task(task, context)
            for task in tasks
        ]

        results = await asyncio.gather(*task_coroutines, return_exceptions=True)

        # Process results
        completed = 0
        failed = 0
        aggregated_results = {}

        for task, result in zip(tasks, results):
            if isinstance(result, Exception):
                self.logger.error(f"Task {task.task_id} failed: {result}")
                task.status = TaskStatus.FAILED
                task.error_message = str(result)
                context.variables.mark_task_failed(task.task_id, str(result))
                failed += 1
            else:
                task.results = result
                task.status = TaskStatus.COMPLETED
                context.variables.mark_task_completed(task.task_id)
                aggregated_results[task.task_id] = result
                completed += 1

        # Synthesize results if requested
        synthesis = None
        if any(task.synthesize for task in tasks if task.status == TaskStatus.COMPLETED):
            synthesis = await self._synthesize_results(
                [task for task in tasks if task.status == TaskStatus.COMPLETED],
                aggregated_results
            )

        # Update context with research findings
        context.variables.research_results = aggregated_results
        if synthesis:
            context.variables.research_results["synthesis"] = synthesis

        await self.emit_status(
            f"Research complete: {completed} succeeded, {failed} failed",
            done=True
        )

        return {
            "tasks_completed": completed,
            "tasks_failed": failed,
            "results": aggregated_results,
            "synthesis": synthesis
        }

    async def _execute_single_task(
        self,
        task: ResearchTask,
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """
        Execute a single research task by routing to appropriate agents.

        Args:
            task: ResearchTask to execute
            context: Execution context

        Returns:
            Task results: {"web": [...], "knowledge": [...]}
        """
        task.status = TaskStatus.IN_PROGRESS
        context.variables.current_task_id = task.task_id

        results = {}

        # Execute searches based on requested sources
        search_coroutines = []

        if "web" in task.sources:
            search_coroutines.append(
                self._search_web(task.query, task.max_results)
            )
        else:
            search_coroutines.append(
                asyncio.sleep(0)  # Placeholder for parallel execution
            )

        if "knowledge" in task.sources:
            search_coroutines.append(
                self._search_knowledge(task.query, task.max_results)
            )
        else:
            search_coroutines.append(
                asyncio.sleep(0)  # Placeholder
            )

        # Execute in parallel
        web_result, knowledge_result = await asyncio.gather(*search_coroutines)

        if "web" in task.sources and web_result:
            results["web"] = web_result
            task.sources_used.append("web")

        if "knowledge" in task.sources and knowledge_result:
            results["knowledge"] = knowledge_result
            task.sources_used.append("knowledge")

        return results

    async def _search_web(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Execute web search via web agent.

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of search results
        """
        await self.emit_status(f"Searching web: {query}")

        try:
            # Web agent will use the registered web_search tool
            # The tool is registered on the user_proxy, so we need to trigger it
            # through agent communication
            self.logger.info(f"Web search: {query}")

            # Return placeholder - actual implementation will use registered tools
            # This will be filled in during orchestrator integration
            return {
                "query": query,
                "max_results": max_results,
                "source": "web"
            }

        except Exception as e:
            self.logger.error(f"Web search failed: {e}")
            raise

    async def _search_knowledge(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Execute knowledge base search via knowledge agent.

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of knowledge base results
        """
        await self.emit_status(f"Searching knowledge base: {query}")

        try:
            # Knowledge agent will use the registered personal_knowledge_search tool
            self.logger.info(f"Knowledge search: {query}")

            # Return placeholder - actual implementation will use registered tools
            return {
                "query": query,
                "max_results": max_results,
                "source": "knowledge"
            }

        except Exception as e:
            self.logger.error(f"Knowledge search failed: {e}")
            raise

    async def _synthesize_results(
        self,
        tasks: List[ResearchTask],
        results: Dict[str, Any]
    ) -> str:
        """
        Synthesize results from multiple research tasks.

        Args:
            tasks: List of completed research tasks
            results: Aggregated results dictionary

        Returns:
            Synthesized summary
        """
        await self.emit_status("Synthesizing research findings")

        # Extract all results
        all_findings = []
        for task_id, task_results in results.items():
            if "web" in task_results:
                all_findings.append(f"Web search findings: {task_results['web']}")
            if "knowledge" in task_results:
                all_findings.append(f"Knowledge base findings: {task_results['knowledge']}")

        # Simple synthesis (can be enhanced with LLM later)
        synthesis = "\n\n".join(all_findings)

        self.logger.info("Research synthesis complete")
        return synthesis
