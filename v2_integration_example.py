"""
V2 Integration Example for granite_autogen_rag.py

This file shows how to integrate V2 orchestrator with the existing Pipe class.
Add this code to your Pipe class in granite_autogen_rag.py.
"""

# =============================================================================
# STEP 1: Add V2 feature flag to Valves class
# =============================================================================

class Valves(BaseModel):
    # ... existing valves ...

    # NEW: V2 Orchestration Feature Flag
    USE_V2_ORCHESTRATION: bool = Field(
        default=False,  # Set to True to enable V2
        description="Enable V2 orchestration with Triage + Context-Aware Routing patterns"
    )


# =============================================================================
# STEP 2: Import V2 at top of file
# =============================================================================

# Add this import after existing imports
from v2 import (
    GraniteOrchestrator,
    ContextVariables,
    ExecutionContext,
    get_agent_configs_for_orchestrator,
)


# =============================================================================
# STEP 3: Modify pipe() method to use V2
# =============================================================================

async def pipe(
    self,
    body,
    __user__: Optional[dict],
    __request__: Request,
    __event_emitter__: Callable[[dict], Awaitable[None]] = None,
) -> str:
    # ... existing setup code ...

    self.event_emitter = __event_emitter__
    self.owui_request = __request__
    self.user = __user__

    # Extract user message
    messages = body.get("messages", [])
    last_message = messages[-1] if messages else {}
    content = last_message.get("content", "")

    # Handle string or multipart content
    if isinstance(content, list):
        user_message = ""
        for part in content:
            if part.get("type") == "text":
                user_message += part.get("text", "")
    else:
        user_message = content

    # =================================================================
    # V2 INTEGRATION POINT
    # =================================================================

    if self.valves.USE_V2_ORCHESTRATION:
        # V2 Path: Use new orchestration
        return await self._pipe_v2(body, user_message)
    else:
        # V1 Path: Existing code (unchanged)
        return await self._pipe_v1(body, user_message)


async def _pipe_v2(self, body, user_message: str) -> str:
    """
    V2 orchestration pipeline.

    Uses:
    - Triage with Tasks pattern
    - Context-Aware Routing
    - Sequential execution (research → control)
    - Async approval workflow
    """
    try:
        # Step 1: Get LLM configs
        llm_configs = get_agent_configs_for_orchestrator(self.valves)

        # Step 2: Create AG2 agents (same as V1)
        from autogen import ConversableAgent
        from pydantic import BaseModel, Field

        # Structured output models (same as V1)
        class Plan(BaseModel):
            steps: list[str]

        class HAOperation(BaseModel):
            operation: str
            parameters: dict
            reasoning: str

        # Create agents
        planner = ConversableAgent(
            name="Planner",
            system_message=PLANNER_MESSAGE,  # From v2.prompts
            llm_config={
                **llm_configs["planner_llm_config"],
                "config_list": [{
                    **llm_configs["planner_llm_config"]["config_list"][0],
                    "response_format": Plan
                }]
            },
            human_input_mode="NEVER",
        )

        assistant = ConversableAgent(
            name="Research_Assistant",
            system_message=ASSISTANT_PROMPT,  # From v2.prompts
            llm_config=llm_configs["assistant_llm_config"],
            human_input_mode="NEVER",
        )

        homeassistant_planner = ConversableAgent(
            name="HomeAssistant_Planner",
            system_message=HA_PLANNER_PROMPT,  # From v2.prompts
            llm_config={
                **llm_configs["ha_planner_llm_config"],
                "config_list": [{
                    **llm_configs["ha_planner_llm_config"]["config_list"][0],
                    "response_format": HAOperation
                }]
            },
            human_input_mode="NEVER"
        )

        user_proxy = ConversableAgent(
            name="User",
            human_input_mode="NEVER",
        )

        # Step 3: Register tools on user_proxy (same as V1)
        self._register_tools_v2(user_proxy, assistant)

        # Step 4: Create execution context
        context = ExecutionContext(
            variables=ContextVariables(
                user_query=user_message,
                model_provider="openrouter" if self.valves.USE_OPENROUTER else "ollama",
                model_name=self.valves.OPENROUTER_TASK_MODEL if self.valves.USE_OPENROUTER else self.valves.TASK_MODEL_ID
            ),
            valves=self.valves,
            event_emitter=self.emit_event_safe
        )

        # Step 5: Create orchestrator
        orchestrator = GraniteOrchestrator(
            valves=self.valves,
            llm_configs=llm_configs,
            user_proxy=user_proxy,
            assistant=assistant,
            planner=planner,
            homeassistant_planner=homeassistant_planner,
            emit_event=self.emit_event_safe
        )

        # Step 6: Process query
        response_chunks = []
        async for chunk in orchestrator.process(user_message, context, stream=True):
            response_chunks.append(chunk)

        return "".join(response_chunks)

    except Exception as e:
        logging.error(f"V2 orchestration error: {e}")
        return f"Error processing request with V2: {str(e)}"


def _register_tools_v2(self, user_proxy, assistant):
    """Register tools for V2 (same as V1)"""

    # Web search tool (copy from V1)
    @assistant.register_for_llm(name="web_search", description="...")
    @user_proxy.register_for_execution(name="web_search")
    def do_web_search(search_instruction: str) -> str:
        # ... V1 implementation ...
        pass

    # Knowledge search tool (copy from V1)
    @assistant.register_for_llm(name="personal_knowledge_search", description="...")
    @user_proxy.register_for_execution(name="personal_knowledge_search")
    def do_knowledge_search(search_instruction: str) -> str:
        # ... V1 implementation ...
        pass

    # Home Assistant tool (copy from V1)
    @assistant.register_for_llm(name="homeassistant", description="...")
    @user_proxy.register_for_execution(name="homeassistant")
    async def do_homeassistant(instruction: str) -> str:
        # ... V1 implementation ...
        pass


async def _pipe_v1(self, body, user_message: str) -> str:
    """
    V1 pipeline (existing code).

    All existing V1 code goes here unchanged.
    """
    # ... ALL EXISTING PIPE CODE FROM LINE 462 ONWARDS ...
    pass


# =============================================================================
# USAGE IN OPEN WEBUI
# =============================================================================

"""
To use V2 in Open WebUI:

1. In granite_autogen_rag.py, add the code above to your Pipe class

2. In Open WebUI:
   - Go to the Functions menu
   - Find "Granite Retrieval Agent"
   - Click the Settings (gear) icon
   - Find "USE_V2_ORCHESTRATION"
   - Toggle it to True
   - Click Save

3. Start a new conversation and test:
   - Simple research: "What are the best smart thermostats?"
   - Home control: "Turn off bedroom lights"
   - Mixed: "Research energy saving and set my thermostat to 68°F"

4. V2 Features:
   - Tasks are decomposed and typed (ResearchTask, HomeControlTask)
   - Research runs in parallel
   - Home control requires approval (placeholder for now)
   - Better error handling and logging
   - Context tracking across phases

5. Rollback:
   - If issues occur, toggle USE_V2_ORCHESTRATION back to False
   - V1 code remains unchanged and available
"""
