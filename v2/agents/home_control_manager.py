"""
Home Control Manager for V2 architecture.

Manages Home Assistant operations with approval workflow:
- Interprets home control tasks
- Requests user approval for operations
- Delegates to HomeAssistant executor
- Tracks operation results
"""

from typing import List, Dict, Any, Optional, Callable, Awaitable
from ..models import HomeControlTask, TaskStatus, ExecutionContext
from .base_agent import BaseAgent
import logging
import asyncio


class HomeControlManager:
    """
    Manages Home Assistant control operations with approval workflow.

    This manager implements the sequential execution pattern for home control phase:
    - Processes HomeControlTask objects
    - Requests user approval for operations (if required)
    - Delegates to HomeAssistant executor
    - Updates execution context with results
    """

    def __init__(
        self,
        homeassistant_executor: Any,  # HomeAssistantExecutor
        emit_event: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
    ):
        """
        Initialize home control manager.

        Args:
            homeassistant_executor: HomeAssistantExecutor instance
            emit_event: Optional event emitter for status updates
        """
        self.executor = homeassistant_executor
        self.emit_event = emit_event
        self.logger = logging.getLogger(__name__)

    async def emit_status(self, message: str, done: bool = False) -> None:
        """Emit status to Open WebUI"""
        if self.emit_event:
            try:
                await self.emit_event({
                    "type": "status",
                    "data": {"description": f"[Home Control] {message}", "done": done}
                })
            except Exception as e:
                self.logger.error(f"Error emitting status: {e}")

    async def emit_approval_request(self, task: HomeControlTask) -> None:
        """Emit approval request to user"""
        if self.emit_event:
            try:
                await self.emit_event({
                    "type": "approval_request",
                    "data": {
                        "task_id": task.task_id,
                        "instruction": task.instruction,
                        "operation": task.operation,
                        "parameters": task.parameters,
                        "message": task.approval_message or f"Approve: {task.instruction}?",
                        "timeout": task.approval_timeout
                    }
                })
            except Exception as e:
                self.logger.error(f"Error emitting approval request: {e}")

    async def execute_with_approval(
        self,
        tasks: List[HomeControlTask],
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """
        Execute home control tasks with approval workflow.

        Tasks are executed SEQUENTIALLY (one at a time) to ensure:
        - User can review each operation
        - Operations happen in correct order
        - Errors don't cascade

        Args:
            tasks: List of HomeControlTask objects
            context: Execution context

        Returns:
            Execution results: {
                "tasks_completed": int,
                "tasks_failed": int,
                "tasks_denied": int,
                "results": {task_id: {...}}
            }
        """
        if not tasks:
            return {
                "tasks_completed": 0,
                "tasks_failed": 0,
                "tasks_denied": 0,
                "results": {}
            }

        await self.emit_status(f"Processing {len(tasks)} home control operations")

        completed = 0
        failed = 0
        denied = 0
        results = {}

        # Execute tasks SEQUENTIALLY (not parallel)
        for task in tasks:
            try:
                result = await self._execute_single_task(task, context)
                results[task.task_id] = result

                if task.status == TaskStatus.COMPLETED:
                    completed += 1
                elif task.approval_status == "denied":
                    denied += 1
                else:
                    failed += 1

            except Exception as e:
                self.logger.error(f"Task {task.task_id} failed: {e}")
                task.status = TaskStatus.FAILED
                task.error_message = str(e)
                context.variables.mark_task_failed(task.task_id, str(e))
                failed += 1

        await self.emit_status(
            f"Home control complete: {completed} succeeded, {failed} failed, {denied} denied",
            done=True
        )

        return {
            "tasks_completed": completed,
            "tasks_failed": failed,
            "tasks_denied": denied,
            "results": results
        }

    async def _execute_single_task(
        self,
        task: HomeControlTask,
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """
        Execute a single home control task with approval workflow.

        Args:
            task: HomeControlTask to execute
            context: Execution context

        Returns:
            Task result dictionary
        """
        task.status = TaskStatus.IN_PROGRESS
        context.variables.current_task_id = task.task_id

        # Step 1: Request approval if required
        if task.requires_approval:
            await self.emit_status(f"Requesting approval: {task.instruction}")
            await self.emit_approval_request(task)

            task.status = TaskStatus.WAITING_APPROVAL
            context.variables.request_approval(task.task_id)

            # Wait for approval (this will be handled by approval handler)
            approved = await self._wait_for_approval(task, context)

            if not approved:
                task.status = TaskStatus.FAILED
                task.approval_status = "denied" if task.approval_status == "denied" else "timeout"
                context.variables.deny_operation(task.task_id)

                return {
                    "status": "denied",
                    "reason": "User denied or timeout",
                    "task_id": task.task_id
                }

            # Mark as approved
            task.approval_status = "approved"
            context.variables.approve_operation(task.task_id)

        # Step 2: Execute via Home Assistant executor
        await self.emit_status(f"Executing: {task.instruction}")
        result = await self.executor.execute(task, context)

        # Step 3: Update task status
        if result.get("error"):
            task.status = TaskStatus.FAILED
            task.error_message = result.get("error_message", "Unknown error")
            context.variables.mark_task_failed(task.task_id, task.error_message)
        else:
            task.status = TaskStatus.COMPLETED
            task.execution_result = result
            context.variables.mark_task_completed(task.task_id)

        return result

    async def _wait_for_approval(
        self,
        task: HomeControlTask,
        context: ExecutionContext
    ) -> bool:
        """
        Wait for user approval.

        This is a placeholder - actual implementation will use
        the approval handler in the executor.

        Args:
            task: Task waiting for approval
            context: Execution context

        Returns:
            True if approved, False if denied/timeout
        """
        # In actual implementation, this will wait for approval event
        # For now, auto-approve in V2 (can be enhanced later)
        self.logger.warning(
            f"Auto-approving task {task.task_id} - approval workflow not fully implemented"
        )

        # Simulate short wait
        await asyncio.sleep(0.5)

        # Auto-approve for now (will be replaced with actual approval handler)
        return True
