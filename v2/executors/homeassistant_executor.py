"""
Home Assistant async executor for V2 architecture.

Handles MCP (Model Context Protocol) communication with Home Assistant:
- Async execution of Home Assistant operations
- MCP session management
- Error handling and retries
"""

from typing import Dict, Any, Optional
import aiohttp
import uuid
import json
import logging
from ..models import HomeControlTask, ExecutionContext


class HomeAssistantExecutor:
    """
    Async executor for Home Assistant MCP operations.

    Handles low-level MCP communication including:
    - Session initialization
    - Tool calls via JSON-RPC 2.0
    - Response parsing (JSON and SSE)
    - Error handling
    """

    def __init__(
        self,
        mcp_url: str,
        mcp_enabled: bool = True
    ):
        """
        Initialize Home Assistant executor.

        Args:
            mcp_url: URL of Home Assistant MCP server
            mcp_enabled: Whether MCP is enabled
        """
        self.mcp_url = mcp_url.rstrip('/')
        self.mcp_enabled = mcp_enabled
        self.session_id: Optional[str] = None
        self.logger = logging.getLogger(__name__)

    async def execute(
        self,
        task: HomeControlTask,
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """
        Execute a Home Assistant control task.

        Args:
            task: HomeControlTask to execute
            context: Execution context

        Returns:
            Execution result dictionary
        """
        if not self.mcp_enabled:
            return {
                "error": True,
                "error_message": "Home Assistant MCP is disabled"
            }

        try:
            # Use operation and parameters from task (populated by HA Planner)
            operation = task.operation
            parameters = task.parameters

            if not operation:
                return {
                    "error": True,
                    "error_message": "No operation specified for Home Assistant task"
                }

            # Execute MCP call
            result = await self._call_mcp_tool(operation, parameters)

            # Store in task
            task.mcp_response = result

            return result

        except Exception as e:
            self.logger.error(f"Home Assistant execution failed: {e}")
            return {
                "error": True,
                "error_type": type(e).__name__,
                "error_message": str(e)
            }

    async def _init_session(self, http_session: aiohttp.ClientSession) -> str:
        """
        Initialize MCP session.

        Args:
            http_session: aiohttp client session

        Returns:
            Session ID
        """
        if self.session_id:
            return self.session_id

        init_payload = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "granite-retrieval-agent-v2",
                    "version": "2.0.0"
                }
            },
            "id": str(uuid.uuid4())
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }

        async with http_session.post(
            self.mcp_url,
            json=init_payload,
            headers=headers
        ) as response:
            if response.status == 200:
                session_id = response.headers.get("Mcp-Session-Id")
                if session_id:
                    self.session_id = session_id
                    self.logger.info(f"MCP session initialized: {session_id}")
                else:
                    result = await response.json()
                    self.logger.info(f"MCP init response: {result}")

                return self.session_id
            else:
                error_text = await response.text()
                raise Exception(
                    f"Failed to initialize MCP session: {response.status} - {error_text}"
                )

    async def _call_mcp_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call MCP tool via JSON-RPC 2.0.

        Args:
            tool_name: Name of the MCP tool
            arguments: Tool arguments

        Returns:
            Tool result
        """
        async with aiohttp.ClientSession() as http_session:
            # Initialize session
            session_id = await self._init_session(http_session)

            # Prepare tool call
            request_id = str(uuid.uuid4())
            payload = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                },
                "id": request_id
            }

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
            if session_id:
                headers["Mcp-Session-Id"] = session_id

            self.logger.info(f"Calling MCP tool: {tool_name} with args {arguments}")

            async with http_session.post(
                self.mcp_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                # Update session ID if returned
                new_session_id = response.headers.get("Mcp-Session-Id")
                if new_session_id:
                    self.session_id = new_session_id

                if response.status == 200:
                    content_type = response.headers.get("Content-Type", "")

                    if "text/event-stream" in content_type:
                        # Handle SSE response
                        result = await self._parse_sse_response(response)
                    else:
                        # Handle JSON response
                        result = await response.json()

                    # Extract result or error
                    if "result" in result:
                        return {"success": True, "data": result["result"]}
                    elif "error" in result:
                        return {
                            "error": True,
                            "error_message": f"MCP Error: {json.dumps(result['error'])}"
                        }

                    return {"success": True, "data": result}
                else:
                    error_text = await response.text()
                    self.logger.error(f"MCP error: {response.status} - {error_text}")
                    return {
                        "error": True,
                        "error_message": f"MCP returned {response.status}: {error_text}"
                    }

    async def _parse_sse_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """
        Parse Server-Sent Events response.

        Args:
            response: aiohttp response object

        Returns:
            Parsed result
        """
        result_text = ""
        async for line in response.content:
            line = line.decode("utf-8").strip()
            if line.startswith("data:"):
                data = line[5:].strip()
                if data:
                    try:
                        event_data = json.loads(data)
                        if "result" in event_data:
                            return event_data
                        elif "error" in event_data:
                            return event_data
                    except json.JSONDecodeError:
                        result_text += data

        return {"result": result_text} if result_text else {"result": "No response"}
