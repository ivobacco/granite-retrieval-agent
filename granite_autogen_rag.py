"""
requirements:  ag2==0.9.10, ag2[ollama]==0.9.10, ag2[openai]==0.9.10, aiohttp

This pipe supports two LLM provider modes:
1. Ollama (Local): Set USE_OPENROUTER=False (default)
   - Requires local Ollama instance running on http://localhost:11434

2. OpenRouter (Cloud): Set USE_OPENROUTER=True
   - Requires OPENROUTER_API_KEY (get from https://openrouter.ai/)
"""

from fastapi import Request
from autogen import ConversableAgent
from typing import Annotated, Optional, Callable, Awaitable
from open_webui.routers import retrieval
from open_webui.models.knowledge import KnowledgeTable
from open_webui import config as open_webui_config
from pydantic import BaseModel, Field
import json
import logging
import aiohttp
import uuid

####################
# Assistant prompts
####################
PLANNER_MESSAGE = """You are a coarse-grained task planner for data gathering and smart home control. You will be given a user's goal your job is to enumerate the coarse-grained steps to gather any data necessary or perform any actions needed to accomplish the goal.
You will not execute the steps yourself, but provide the steps to a helper who will execute them. The helper has to the following tools to help them accomplish tasks:

1. Search through a collection of documents provided by the user. These are the user's own documents and will likely not have latest news or other information you can find on the internet.
2. Given a question/topic, search the internet for resources to address question/topic (you don't need to formulate search queries, the tool will do it for you)
3. Home Assistant smart home control with comprehensive capabilities:
   - Device discovery and state checking (lights, climate, media players, covers, locks, fans, vacuums, alarms, sensors)
   - Device control (turn on/off, adjust settings, open/close, lock/unlock)
   - Automation and scene management (list, trigger, activate)
   - History and analytics (retrieve historical data, energy analysis, usage patterns)
   - System maintenance (find unavailable devices, analyze consumption, smart scenarios)
   - Add-on and package management
   - Notifications (only when explicitly requested by user)
Do not include steps for summarizing or synthesizing data. That will be done by another helper later, once all the data is gathered.

You may use any of the capabilities that the helper has, but you do not need to use all of them if they are not required to complete the task.
For example, if the task requires knowledge that is specific to the user, you may choose to include a step that searches through the user's documents. However, if the task only requires information that is available on the internet, you may choose to include a step that searches the internet and omit document searching.

Keep the steps simple and geared towards using the tools for data collection. Below are some examples.

Example 1:
User Input: Summarize the experiment results in StudyA.doc and integrate them with the latest peer-reviewed articles on similar topics you find online.
Plan: ["Fetch experiment results from StudyA.doc in local knowledge store",
"For each experiment result, search the internet for peer reviewed articles that cover similar topics to the experiment"]

Example 2:
User Input: Create a background report comparing our company’s last annual ESG performance with current sustainability regulations.
Plan: [
"Fetch last annual ESG performance data from the user's documents",
"Search the internet for the latest sustainability regulations and reporting requirements"
]

Example 3:
User Input: Gather current statistics on electric vehicle adoption rates in Europe and government incentive programs.
Plan: [
"Search the internet for recent statistics on electric vehicle adoption rates in Europe",
"Search the internet for information about government incentive programs for electric vehicles in European countries"
]

Example 4:
User Input: Retrieve all internal meeting notes and task logs related to the Alpha Project post-mortem.
Plan: [
"Search through the user's documents for all meeting notes and task logs related to the Alpha Project post-mortem"
]

Example 5:
User Input: Turn off all the lights in the living room and set the thermostat to 68 degrees.
Plan: [
"Search for all light entities in the living room area",
"Turn off all lights found in the living room",
"Set the living room thermostat to 68 degrees"
]

Example 6:
User Input: What's the current temperature in the bedroom and is the window open?
Plan: [
"Get the state of the bedroom temperature sensor",
"Get the state of the bedroom window contact sensor"
]

Example 7:
User Input: Activate my movie night scene and set volume to 30%.
Plan: [
"Activate the movie night scene",
"Set media player volume to 30%"
]

Example 8:
User Input: Find all devices that are unavailable and check energy consumption.
Plan: [
"Find all unavailable entities using maintenance tools",
"Analyze energy consumption patterns"
]

Example 9:
User Input: Show me the temperature history for the last 24 hours.
Plan: [
"Retrieve temperature history for the last 24 hours from climate sensors"
]

Example 10:
User Input: Start the vacuum in the living room.
Plan: [
"Start vacuum cleaner in the living room"
]
"""

ASSISTANT_PROMPT = """You are a task execution specialist. Execute the instruction using available tools and context.

## INPUTS
**Instruction**: Task to complete (highest priority)
**Contextual Information**: Prior results, pre-fetched data (use freely)

## TOOLS (3)
1. web_search(query) - Current events, statistics, online data
2. personal_knowledge_search(query) - User's documents, local knowledge
3. homeassistant(instruction) - Smart home control/queries
   - Discovery: "find bedroom lights", "list motion sensors"
   - Control: "turn off lights", "set thermostat to 72°F"
   - State: "get bedroom temperature", "check garage door"
   - History: "temperature last 24h", "light usage patterns"
   - Automation: "activate movie scene"
   Returns JSON: {state, attributes, entity_id}

## EXECUTION FLOW
1. Context sufficient? → ##ANSWER## (no tools)
2. Identify missing data:
   - Online/current → web_search
   - User docs → personal_knowledge_search
   - Smart home → homeassistant
3. Call ONE tool, wait for result
4. Sufficient? → ##ANSWER## | Need more? → Call again (max 5 total)

## OUTPUT
- Success: ##ANSWER## <response>
- Impossible: ##TERMINATE##
- Sources: Add [1], [2] at end

## EXAMPLE
Instruction: "Dim reading lamp to 30%"
Context: [empty]

Step 1: homeassistant("find lights with 'reading' in name")
→ {"lights": [{"entity_id": "light.bedroom_reading", "brightness": 255}]}

Step 2: homeassistant("set light.bedroom_reading to 30%")
→ {"status": "success", "brightness": 76}

Output: ##ANSWER## Reading lamp dimmed to 30%.

## RULES
- ONE tool call at a time
- Ground answers in tool/context only
- Be direct, concise
- Max 5 tool calls per instruction
- If tool fails: Refine query (e.g., area-based search)
- After 5 calls: Explain issue + ##TERMINATE##
"""

GOAL_JUDGE_PROMPT = """
You are a strict and objective judge. Your task is to determine whether the original goal has been **fully and completely fulfilled**, based on the goal itself, the planned steps, the steps taken, and the information gathered.

## EVALUATION RULES
- You must provide:
  1. A **binary decision** (`True` or `False`), and
  2. A **1–2 sentence explanation** that clearly states the decisive reason.
- **Every single requirement** of the goal must be satisfied for the decision to be `True`.
- If **any part** of the goal or planned steps remains unfulfilled, return `False`.
- Do **not** attempt to fulfill the goal yourself — only evaluate what has been done.

## HOW TO JUDGE
1. **Understand the Goal:** Identify what exactly is required to consider the goal fully met.
2. **Check Information Coverage:** Verify whether the data in “Information Gathered” is:
   - Sufficient in quantity and relevance to address the full goal;
   - Not just references to actions, but actual collected content.


## INPUT FORMAT (JSON)
    ```
    {
        "Goal": "The ultimate goal/instruction to be fully fulfilled.",
        "Media Description": "If the user provided an image to supplement their instruction, a description of the image's content."
        "Originally Planned Steps: ": "The plan to achieve the goal, all of the steps may or may not have been executed so far. It may be the case that not all the steps need to be executed in order to achieve the goal, but use this as a consideration.",
        "Steps Taken so far": "All steps that have been taken so far",
        "Information Gathered": "The information collected so far in pursuit of fulfilling the goal. This is the most important piece of information in deciding whether the goal has been met."
    }
    ```
"""

REFLECTION_ASSISTANT_PROMPT = """You are a strategic planner focused on choosing the next step in a sequence of steps to achieve a given goal.
You will receive data in JSON format containing the current state of the plan and its progress.
Your task is to determine the single next step, ensuring it aligns with the overall goal and builds upon the previous steps.
The step will be executed by different helpers that have the following capabilities: 
- access to web search tool 
- access to tools to search personal documents
- access to homeassistant tool that can control and query homeassistant managed devices

JSON Structure:
{
    "Goal": The original objective from the user,
    "Media Description": A textual description of any associated image,
    "Plan": An array outlining every planned step,
    "Last Step": The most recent action taken,
    "Last Step Output": The result of the last step, indicating success or failure,
    "Steps Taken": A chronological list of executed steps.
}

Guidelines:
1. If the last step output is ##NO##, reassess and refine the instruction to avoid repeating past mistakes. Provide a single, revised instruction for the next step.
2. If the last step output is ##YES##, proceed to the next logical step in the plan.
3. Use 'Last Step', 'Last Step Output', and 'Steps Taken' for context when deciding on the next action.
4. Only instruct the helper to do something that is within their capabilities.

Restrictions:
1. Do not attempt to resolve the problem independently; only provide instructions for the subsequent agent's actions.
2. Limit your response to a single step or instruction.
    """

STEP_CRITIC_PROMPT = """The previous instruction was {last_step}
The following is the output of that instruction.

EVALUATION CRITERIA:
- If the output completely satisfies the instruction, reply with True for the decision and an explanation why.
- If the output does not properly satisfy the instruction, reply with False for the decision and the reason why.

SUCCESS EXAMPLES:
- Instruction: "List companies that use AI" → Output contains a list of companies → True
- Instruction: "Get the bedroom temperature" → Output shows temperature reading → True
- Instruction: "Turn off living room lights" → Output confirms lights turned off → True

FAILURE EXAMPLES:
- Output contains "I'm sorry but..." → Likely not fulfilling the instruction → False
- Instruction: "List companies that use AI" → Output says "no companies found" or "list not available" → False
- Instruction: "Search for bedroom lights" → Output says "no devices found" → False (if user asks for something, it likely exists; search parameters may need refinement)
- Instruction: "Get thermostat history" → Output returns empty/no data → False (entities likely exist; may need different query approach)

IMPORTANT: If the user's instruction implies something should exist (e.g., asking for bedroom lights, thermostat data, device history), it likely does: 
- It may contain devices of same domain, class or type, but with similar names (e.g., italian names instead of english ones)
- If The output returns empty/not found, this is likely a FAILED step. The query may need to be reformulated by focusing search on areas, or domains, or blob names patterns.

Remember to always provide both a decision and an explanation.
Previous step output: \n {last_output}"""

SEARCH_QUERY_GENERATION_PROMPT = """You are a search query generation assistant.
Your task is to take a long, detailed user request and break it down into multiple focused, high-quality search queries.
Each query should target a distinct subtopic or key aspect of the original request so that, together, the queries fully cover the user’s information need.

Instructions:

- Identify all major subtopics, steps, or themes in the input.
- Write clear and specific search queries for each subtopic.
- Include relevant keywords, entities, or technologies.
- Use the date to augment queries if the user is asking of recent or latest information but be very precise. (Assume the current date is {datetime.now(UTC).strftime("%B %d, %Y")})
- Use the + operator to boost important concepts.
- Do not simply restate the input as one query—decompose it into up to 3 targeted queries.
Example Input:
“strategies for launching a new productivity mobile app, including market research on user behavior trends, competitor analysis in the productivity app space, feature prioritization based on user needs, designing intuitive onboarding experiences, implementing in-app analytics for engagement tracking, planning a social media marketing campaign, testing beta versions with early adopters, collecting feedback, and preparing for a global rollout.”
Expected Output:
[
    "effective +strategies for launching new +productivity mobile apps in 2025 --QDF=5",
    "market research and competitor analysis for +productivity apps",
    "onboarding design and +in-app analytics strategies for mobile applications"
]
"""

HA_PLANNER_PROMPT = """Translate natural language to Home Assistant MCP tool calls. Return JSON: {operation, parameters, reasoning}.

## TOOLS (25)

**DISCOVERY**
• list_devices(domain?, area?, floor?) - Simple filtering. Domains: light, climate, alarm_control_panel, cover, switch, contact, media_player, fan, lock, vacuum, scene, script, camera
• search_entities(pattern?, domain?, device_class?, state?, area?, attributes?, changed_within?, output?) - Advanced search. Also use for getting sensor states.
  pattern: '*motion*', 'sensor.*temp*', 'sensor.living_room_temperature' | state: 'on', '>50', '!=unavailable' | device_class: motion, door, temperature, battery
  attributes: [{key, op, value}] | changed_within: '5m','1h','24h' | output: minimal/summary/full

**CONTROL** (domain-specific tools use action: list, get, turn_on/off, + specific actions)
• lights_control(action, entity_id?, brightness?, color_temp?, rgb_color?) - brightness: 0-255
• climate_control(action, entity_id?, temperature?, hvac_mode?, fan_mode?)
• media_player_control(action, entity_id?, volume_level?, source?)
• cover_control(action, entity_id?, position?)
• lock_control(action, entity_id?, code?)
• fan_control(action, entity_id?, percentage?, oscillating?)
• vacuum_control(action, entity_id?, fan_speed?) - start, pause, stop, return_to_base, clean_spot
• alarm_control(action, entity_id?, code?) - alarm_disarm, alarm_arm_home, alarm_arm_away

**AUTOMATION/SYSTEM**
• scene(action, scene_id?) - list, activate
• automation(action, automation_id?) - list, toggle, trigger
• automation_config(action, automation_id?, config?) - create, update, delete
• notify(message, target?, title?, data?) - Use ONLY when explicitly requested
• get_history(entity_id, start_time?, end_time?) - Times: "24 hours ago", "2024-01-15", "5m", "1h"
• addon(action, slug?) - list, info, install, start, stop
• package(action, category?, repository?) - HACS packages
• maintenance(action, days?) - find_orphaned_devices, analyze_light_usage, analyze_energy_consumption, find_unavailable_entities, device_health_check
• smart_scenarios(action, mode?) - detect_scenarios, apply_nobody_home, apply_window_heating_check, detect_issues
• subscribe_events(token, events?, entity_id?) - Real-time event streaming
• get_sse_stats(token) - Event statistics

## TOOL SELECTION
search_entities: Pattern matching, state filtering, attribute conditions, time-based, multi-criteria, uncertain names, SENSORS (use pattern='sensor.exact_name')
list_devices: Simple domain/area/floor filtering
Domain-specific control action="list": Domain discovery
Domain-specific control action="get": Single entity state query (lights, climate, media_player, etc.)

## EXAMPLES
"Find motion sensors" → {{"operation": "search_entities", "parameters": {{"pattern": "*motion*"}}, "reasoning": "Glob pattern"}}
"Low battery devices" → {{"operation": "search_entities", "parameters": {{"attributes": [{{"key": "battery_level", "op": "<", "value": 20}}]}}, "reasoning": "Attribute filter"}}
"Changed last 5 min" → {{"operation": "search_entities", "parameters": {{"changed_within": "5m"}}, "reasoning": "Time-based"}}
"Active motion in bedroom" → {{"operation": "search_entities", "parameters": {{"device_class": "motion", "state": "on", "area": "bedroom"}}, "reasoning": "Multi-criteria"}}
"All lights" → {{"operation": "list_devices", "parameters": {{"domain": "light"}}, "reasoning": "Simple domain"}}
"Bedroom devices" → {{"operation": "list_devices", "parameters": {{"area": "bedroom"}}, "reasoning": "Area filter"}}
"Check bedroom light" → {{"operation": "lights_control", "parameters": {{"action": "get", "entity_id": "light.bedroom"}}, "reasoning": "Light state query"}}
"Living room temperature" → {{"operation": "search_entities", "parameters": {{"pattern": "sensor.living_room_temperature"}}, "reasoning": "Sensor state query"}}
"Turn off bedroom lights" → {{"operation": "lights_control", "parameters": {{"action": "turn_off", "entity_id": "light.bedroom"}}, "reasoning": "Control"}}
"Dim to 50%" → {{"operation": "lights_control", "parameters": {{"action": "turn_on", "entity_id": "light.bedroom", "brightness": 128}}, "reasoning": "50%=128/255"}}
"Thermostat to 72°F" → {{"operation": "climate_control", "parameters": {{"action": "set_temperature", "entity_id": "climate.living_room", "temperature": 72}}, "reasoning": "Set temp"}}
"Temp history 24h" → {{"operation": "get_history", "parameters": {{"entity_id": "sensor.bedroom_temperature", "start_time": "24 hours ago"}}, "reasoning": "Historical"}}
"Activate movie scene" → {{"operation": "scene", "parameters": {{"action": "activate", "scene_id": "scene.movie_night"}}, "reasoning": "Scene trigger"}}
"Unavailable devices" → {{"operation": "maintenance", "parameters": {{"action": "find_unavailable_entities"}}, "reasoning": "Health check"}}
"Leaving home" → {{"operation": "smart_scenarios", "parameters": {{"action": "apply_nobody_home", "mode": "apply"}}, "reasoning": "Nobody-home scenario"}}
"""

REPORT_WRITER_PROMPT = """
You are a precise and well-structured report writer.
Your task is to summarize the information provided to you in order to directly answer the user's instruction or query.

Guidelines:

1. Use **only the information provided**. Do not make up, infer, or fabricate facts.
2. Organize the report into clear sections with headings when appropriate.
3. For every statement, fact, or claim that is derived from a specific source, **cite it with an explicit hyperlink** to the original URL. Use Markdown citation format like this:

   * Example: "The system achieved state-of-the-art results [source](https://example.com/article)."
4. If multiple sources support a point, you may cite more than one.
5. If some information is repeated across multiple sources, summarize it concisely without redundancy.
6. If the provided information does not fully answer the user's query, clearly state what is missing, but do not invent new details.
7. Maintain a neutral, factual tone — avoid speculation, exaggeration, or opinion.

Output Format:

* Begin with a short **executive summary** that directly answers the query.
* Follow with supporting details structured in sections and paragraphs.
* Include hyperlinks inline with each reference.
																								
													
												   

Important:

* Do not include any sources or information not explicitly provided.
* Do not use vague references like "according to a website" — always hyperlink.
* If no sources are relevant, say so explicitly.
"""
class Pipe:
    class Valves(BaseModel):
        # Provider Selection
        USE_OPENROUTER: bool = Field(default=True, description="Use OpenRouter API instead of local Ollama")

        # OpenRouter Configuration (Cloud)
        OPENROUTER_API_KEY: str = Field(default="", description="OpenRouter API key (sk-or-v1-...)")
        OPENROUTER_BASE_URL: str = Field(default="https://openrouter.ai/api/v1", description="OpenRouter API base URL")
        OPENROUTER_TASK_MODEL: str = Field(default="xiaomi/mimo-v2-flash:free", description="OpenRouter model for tasks")
        OPENROUTER_VISION_MODEL: str = Field(default="xiaomi/mimo-v2-flash:free", description="OpenRouter model for vision tasks")

        # Local Ollama Configuration (Local)
        TASK_MODEL_ID: str = Field(default="xiaomi/mimo-v2-flash:free")
        VISION_MODEL_ID: str = Field(default="xiaomi/mimo-v2-flash:free")
        OPENAI_API_URL: str = Field(default="https://openrouter.ai/api/v1")
        OPENAI_API_KEY: str = OPENROUTER_API_KEY
        VISION_API_URL: str = Field(default="https://openrouter.ai/api/v1")

        # Common Configuration
        MODEL_TEMPERATURE: float = Field(default=0)
        MAX_PLAN_STEPS: int = Field(default=6)

        # Home Assistant MCP Configuration
        HOMEASSISTANT_MCP_URL: str = Field(default="http://localhost:3000/mcp", description="URL of the Home Assistant MCP Streamable HTTP endpoint")
        HOMEASSISTANT_MCP_ENABLED: bool = Field(default=True, description="Enable Home Assistant MCP integration")

    def __init__(self):
        self.type = "pipe"
        self.id = "granite_retrieval_agent"
        self.name = "Granite Retrieval Agent"
        self.valves = self.Valves()

    def get_provider_models(self):
        return [
            {"id": self.valves.OPENROUTER_TASK_MODEL, "name": self.valves.TASK_MODEL_ID},
        ]

    def is_open_webui_request(self, body):
        """
        Checks if the request is an Open WebUI task, as opposed to a user task
        """
        message = str(body[-1])

        prompt_templates = {
            "### Task",
            open_webui_config.DEFAULT_RAG_TEMPLATE.replace("\n", "\\n"),
            open_webui_config.DEFAULT_TITLE_GENERATION_PROMPT_TEMPLATE.replace(
                "\n", "\\n"
            ),
            open_webui_config.DEFAULT_TAGS_GENERATION_PROMPT_TEMPLATE.replace(
                "\n", "\\n"
            ),
            open_webui_config.DEFAULT_IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE.replace(
                "\n", "\\n"
            ),
            open_webui_config.DEFAULT_QUERY_GENERATION_PROMPT_TEMPLATE.replace(
                "\n", "\\n"
            ),
            open_webui_config.DEFAULT_AUTOCOMPLETE_GENERATION_PROMPT_TEMPLATE.replace(
                "\n", "\\n"
            ),
            open_webui_config.DEFAULT_TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE.replace(
                "\n", "\\n"
            ),
        }

        for template in prompt_templates:
            if template is not None and template[:50] in message:
                return True

        return False

    async def emit_event_safe(self, message):
        event_data = {
            "type": "message",
            "data": {"content": message + "\n"},
        }
        try:
            await self.event_emitter(event_data)
        except Exception as e:
            logging.error(f"Error emitting event: {e}")

    async def pipe(
        self,
        body,
        __user__: Optional[dict],
        __request__: Request,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> str:

        # Grab env variables
        default_model = self.valves.TASK_MODEL_ID
        base_url = self.valves.OPENAI_API_URL
        api_key = self.valves.OPENAI_API_KEY
        vision_model = self.valves.VISION_MODEL_ID
        vision_url = self.valves.VISION_API_URL
        model_temp = self.valves.MODEL_TEMPERATURE
        max_plan_steps = self.valves.MAX_PLAN_STEPS
        self.event_emitter = __event_emitter__
        self.owui_request = __request__
        self.user = __user__

        ##################
        # AutoGen Config
        ##################
        # Structured Output Objects for each agent
        class Plan(BaseModel):
            steps: list[str]

        class CriticDecision(BaseModel):
            decision: bool = Field(description="A true or false decision on whether the goal has been fully accomplished")
            explanation: str = Field(description="A thorough yet concise explanation of why you came to this decision.")

        class Step(BaseModel):
            step_instruction: str = Field(description="A concise instruction of what the next step in the plan should be")
            requirement_to_fulfill: str = Field(description="Explain your thinking around the requirement of the plan that this step will accomplish and why you chose the step instruction")

        class SearchQueries(BaseModel):
            search_queries: list[str] = Field(description="A list of search queries")

        class HAOperation(BaseModel):
            operation: str = Field(description="Operation type: search_entities, get_entity_state, control, or automation_config")
            parameters: dict = Field(description="Structured parameters for the operation")
            reasoning: str = Field(description="Brief explanation of parameter choices")

        # LLM Config - Support both Ollama (local) and OpenRouter (cloud)
        use_openrouter = self.valves.USE_OPENROUTER

        if use_openrouter:
            # OpenRouter Configuration
            if not self.valves.OPENROUTER_API_KEY:
                raise ValueError("OPENROUTER_API_KEY must be set when USE_OPENROUTER is enabled")

            base_llm_config = {
                "model": self.valves.OPENROUTER_TASK_MODEL,
                "base_url": self.valves.OPENROUTER_BASE_URL,
                "api_type": "openai",
                "api_key": self.valves.OPENROUTER_API_KEY,
                "temperature": model_temp,
            }

            quick_llm_config = {
                "model": "nvidia/nemotron-3-nano-30b-a3b:free",
                "base_url": self.valves.OPENROUTER_BASE_URL,
                "api_type": "openai",
                "api_key": self.valves.OPENROUTER_API_KEY,
                "temperature": model_temp,
            }

            vision_config = {
                "model": self.valves.OPENROUTER_VISION_MODEL,
                "base_url": self.valves.OPENROUTER_BASE_URL,
                "api_type": "openai",
                "api_key": self.valves.OPENROUTER_API_KEY,
            }
        else:
            # Ollama Configuration (Local)
            base_llm_config = {
                "model": default_model,
                "client_host": base_url,
                "api_type": "openai",
                "temperature": model_temp,
                "num_ctx": 131072,
            }

            vision_config = {
                "model": vision_model,
                "base_url": vision_url,
                "api_type": "openai",
                "api_key": api_key,
            }

        llm_configs = {
            "ollama_llm_config": {**base_llm_config, "config_list": [{**base_llm_config}]},
            "assistant_llm_config": {**base_llm_config, "config_list": [{**quick_llm_config}]},
            "planner_llm_config": {**base_llm_config, "config_list": [{**quick_llm_config, "response_format": Plan}]},
            "critic_llm_config": {**base_llm_config, "config_list": [{**base_llm_config, "response_format": CriticDecision}]},
            "reflection_llm_config": {**base_llm_config, "config_list": [{**quick_llm_config, "response_format": Step}]},
            "search_query_llm_config": {**base_llm_config, "config_list": [{**quick_llm_config, "response_format": SearchQueries}]},
            "ha_planner_llm_config": {**base_llm_config, "config_list": [{**quick_llm_config, "response_format": HAOperation}]},
            "vision_llm_config": {
                "config_list": [vision_config]
            },
        }

        ### Agents
        # Generic LLM completion, used for servicing Open WebUI originated requests
        generic_assistant = ConversableAgent(
            name="Generic_Assistant",
            llm_config=llm_configs["ollama_llm_config"],
            human_input_mode="NEVER",
        )

        # Vision Assistant
        vision_assistant = ConversableAgent(
            name="Vision_Assistant",
            llm_config=llm_configs["vision_llm_config"],
            human_input_mode="NEVER",
        )

        # Provides the initial high level plan
        planner = ConversableAgent(
            name="Planner",
            system_message=PLANNER_MESSAGE,
            llm_config=llm_configs["planner_llm_config"],
            human_input_mode="NEVER",
        )

        # The assistant agent is responsible for executing each step of the plan, including calling tools
        assistant = ConversableAgent(
            name="Research_Assistant",
            system_message=ASSISTANT_PROMPT,
            llm_config=llm_configs["assistant_llm_config"],
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: "tool_response" not in msg
            and msg["content"] == "",
        )

        # Determines whether the ultimate objective has been met
        goal_judge = ConversableAgent(
            name="GoalJudge",
            system_message=GOAL_JUDGE_PROMPT,
            llm_config=llm_configs["critic_llm_config"],
            human_input_mode="NEVER",
        )

        # Step Critic
        step_critic = ConversableAgent(
            name="Step_Critic",
            llm_config=llm_configs["critic_llm_config"],
            human_input_mode="NEVER",
        )

        # Reflection Assistant: Reflect on plan progress and give the next step
        reflection_assistant = ConversableAgent(
            name="ReflectionAssistant",
            system_message=REFLECTION_ASSISTANT_PROMPT,
            llm_config=llm_configs["reflection_llm_config"],
            human_input_mode="NEVER",
        )

        # Report Generator
        report_generator = ConversableAgent(
            name="Report_Generator",
            llm_config=llm_configs["ollama_llm_config"],
            human_input_mode="NEVER",
            system_message=REPORT_WRITER_PROMPT
        )

        # Search Query generator
        search_query_generator = ConversableAgent(
            name="Search_Query_Generator",
            system_message=SEARCH_QUERY_GENERATION_PROMPT,
            llm_config=llm_configs["search_query_llm_config"],
            human_input_mode="NEVER"
        )

        # Home Assistant Query Planner
        homeassistant_planner = ConversableAgent(
            name="HomeAssistant_Planner",
            system_message=HA_PLANNER_PROMPT,
            llm_config=llm_configs["ha_planner_llm_config"],
            human_input_mode="NEVER"
        )

        # User Proxy chats with assistant on behalf of user and executes tools
        user_proxy = ConversableAgent(
            name="User",
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: "##ANSWER" in msg["content"]
            or "## Answer" in msg["content"]
            or "##TERMINATE##" in msg["content"]
            or ("tool_calls" not in msg and msg["content"] == ""),
        )

        ##################
        # Check if this request is utility call from OpenWebUI
        ##################
        if self.is_open_webui_request(body["messages"]):
            print("Is open webui request")
            try:
                reply = generic_assistant.generate_reply(messages=[body["messages"][-1]])
                return reply
            except Exception as e:
                logging.error(f"Error generating reply from generic assistant: {e}")
                return f"Error processing request: {str(e)}"

        ##################
        # Tool Definitions
        ##################
        @assistant.register_for_llm(
            name="web_search", description="Use this tool to search the internet for up-to-date, location-specific, or niche information that may not be reliably available in the model’s training data. \
                This includes current events, fresh statistics, local details, product information, regulations, sports schedules, software updates, company details, and anything that changes frequently over time."
        )
        @user_proxy.register_for_execution(name="web_search")
        def do_web_search(
            search_instruction: Annotated[
                str,
                "Provide a detailed search instruction that incorporates specific features, goals, and contextual details related to the query. \
                                                        Identify and include relevant aspects from any provided context, such as key topics, technologies, challenges, timelines, or use cases. \
                                                        Construct the instruction to enable a targeted search by specifying important attributes, keywords, and relationships within the context.",
            ]
        ) -> str:
            """This function is used for searching the internet for information that can only be found on the internet, not in the users personal notes."""
            if not search_instruction:
                return "Please provide a search query."

            try:
                response = user_proxy.initiate_chat(recipient=search_query_generator, max_turns=1, message=search_instruction)
                search_queries = json.loads(response.chat_history[-1]["content"])["search_queries"]
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse search query response: {e}")
                return f"Error generating search queries: {str(e)}"
            except (KeyError, IndexError) as e:
                logging.error(f"Invalid response structure from search query generator: {e}")
                return f"Error processing search query response: {str(e)}"
            except Exception as e:
                logging.error(f"Error initiating search query generation: {e}")
                return f"Error during search query generation: {str(e)}"

            search_results = []
            for query in search_queries:
                logging.info("Searching for " + query)
                try:
                    results = retrieval.search_web(
                        self.owui_request,
                        self.owui_request.app.state.config.WEB_SEARCH_ENGINE,
                        search_instruction,
                    )
                    for result in results:
                        search_results.append({"Title": result.title, "URL": result.link, "Text": result.snippet})
                except Exception as e:
                    logging.error(f"Error searching web for query '{query}': {e}")
                    search_results.append({"Title": "Search Error", "URL": "", "Text": f"Failed to search for: {query}. Error: {str(e)}"})

            return str(search_results)

        @assistant.register_for_llm(
            name="personal_knowledge_search",
            description="Searches personal documents according to a given query",
        )
        @user_proxy.register_for_execution(name="personal_knowledge_search")
        def do_knowledge_search(
            search_instruction: Annotated[str, "search instruction"]
        ) -> str:
            """Use this tool if you need to obtain information that is unique to the user and cannot be found on the internet.
            Given an instruction on what knowledge you need to find, search the user's documents for information particular to them, their projects, and their domain.
            This is simple document search, it cannot perform any other complex tasks.
            This will not give you any results from the internet. Do not assume it can retrieve the latest news pertaining to any subject.
            """
            if not search_instruction:
                return "Please provide a search query."

            try:
                # First get all the user's knowledge bases associated with the model
                knowledge_item_list = KnowledgeTable().get_knowledge_bases()
                if len(knowledge_item_list) == 0:
                    return "You don't have any knowledge bases."
                collection_list = []
                for item in knowledge_item_list:
                    collection_list.append(item.id)

                collection_form = retrieval.QueryCollectionsForm(
                    collection_names=collection_list, query=search_instruction
                )

                response = retrieval.query_collection_handler(
                    request=self.owui_request, form_data=collection_form, user=self.user
                )
                messages = ""
                for entries in response.get("documents", []):
                    for line in entries:
                        messages += line

                return messages if messages else "No relevant documents found."
            except Exception as e:
                logging.error(f"Error searching knowledge base: {e}")
                return f"Error searching knowledge base: {str(e)}"

        # Home Assistant MCP Tool
        if self.valves.HOMEASSISTANT_MCP_ENABLED:
            mcp_base_url = self.valves.HOMEASSISTANT_MCP_URL.rstrip('/')

            # Store MCP session ID for reuse across calls
            mcp_session_state = {"session_id": None}

            async def init_mcp_session(http_session: aiohttp.ClientSession) -> str:
                """Initialize MCP session and return session ID."""
                if mcp_session_state["session_id"]:
                    return mcp_session_state["session_id"]

                init_payload = {
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {
                            "name": "granite-retrieval-agent",
                            "version": "1.0.0"
                        }
                    },
                    "id": str(uuid.uuid4())
                }

                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream"
                }

                async with http_session.post(mcp_base_url, json=init_payload, headers=headers) as response:
                    if response.status == 200:
                        # Get session ID from response header
                        session_id = response.headers.get("Mcp-Session-Id")
                        if session_id:
                            mcp_session_state["session_id"] = session_id
                        else:
                            # Log response for debugging
                            result = await response.json()
                            logging.info(f"MCP init response: {result}")
                        return mcp_session_state["session_id"]
                    else:
                        error_text = await response.text()
                        raise Exception(f"Failed to initialize MCP session: {response.status} - {error_text}")

            @assistant.register_for_llm(
                name="homeassistant",
                description="Control and query Home Assistant devices. Provide natural language instructions (e.g., 'find lights in bedroom', 'turn off all lights', 'check temperature', 'show history', 'activate scene'). Supports discovery, control, state checking, automation, history, and maintenance across all device types."
            )
            @user_proxy.register_for_execution(name="homeassistant")
            async def do_homeassistant(
                instruction: Annotated[str, "Natural language instruction for Home Assistant operation (e.g., 'turn off bedroom lights', 'find all motion sensors', 'what is the temperature in living room')"]
            ) -> str:
                """Execute Home Assistant operations by first planning the operation structure, then executing via MCP."""

                try:
                    # Use HA Planner to structure the operation
                    planner_response = await user_proxy.a_initiate_chat(
                        recipient=homeassistant_planner,
                        max_turns=1,
                        message=instruction
                    )
                    ha_operation = json.loads(planner_response.chat_history[-1]["content"])

                    operation = ha_operation["operation"]
                    arguments = ha_operation["parameters"]

                    logging.info(f"HA Planner - Operation: {operation}, Reasoning: {ha_operation['reasoning']}")

                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse HA planner response: {e}")
                    return f"Error planning Home Assistant operation: {str(e)}"
                except (KeyError, IndexError) as e:
                    logging.error(f"Invalid HA planner response structure: {e}")
                    return f"Error processing HA planner response: {str(e)}"
                except Exception as e:
                    logging.error(f"Error during HA operation planning: {e}")
                    return f"Error planning operation: {str(e)}"

                # Execute the planned operation via MCP
                try:
                    async with aiohttp.ClientSession() as http_session:
                        # Initialize MCP session first
                        session_id = await init_mcp_session(http_session)

                        # Map operation to MCP tool name (operation from planner = tool name for MCP)
                        # Valid operations: list_devices, lights_control, climate_control, cover_control,
                        # lock_control, fan_control, vacuum_control, alarm_control, media_player_control,
                        # control, automation_config, scene, automation, notify, get_history, etc.
                        tool_name = operation

                        # Make the MCP tool call via Streamable HTTP (JSON-RPC format)
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

                        logging.info(f"Calling Home Assistant MCP: {tool_name} with args {arguments}")

                        async with http_session.post(mcp_base_url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                            # Update session ID if returned in response
                            new_session_id = response.headers.get("Mcp-Session-Id")
                            if new_session_id:
                                mcp_session_state["session_id"] = new_session_id

                            if response.status == 200:
                                content_type = response.headers.get("Content-Type", "")
                                if "text/event-stream" in content_type:
                                    # Handle SSE response
                                    result_text = ""
                                    async for line in response.content:
                                        line = line.decode("utf-8").strip()
                                        if line.startswith("data:"):
                                            data = line[5:].strip()
                                            if data:
                                                try:
                                                    event_data = json.loads(data)
                                                    if "result" in event_data:
                                                        return json.dumps(event_data["result"], indent=2)
                                                    elif "error" in event_data:
                                                        return f"MCP Error: {json.dumps(event_data['error'])}"
                                                except json.JSONDecodeError:
                                                    result_text += data
                                    return result_text if result_text else "No response from MCP"
                                else:
                                    # Handle regular JSON response
                                    result = await response.json()
                                    if "result" in result:
                                        return json.dumps(result["result"], indent=2)
                                    elif "error" in result:
                                        return f"MCP Error: {json.dumps(result['error'])}"
                                    return json.dumps(result, indent=2)
                            else:
                                error_text = await response.text()
                                logging.error(f"Home Assistant MCP error: {response.status} - {error_text}")
                                return f"Error from Home Assistant MCP: {response.status} - {error_text}"

                except aiohttp.ClientError as e:
                    logging.error(f"HTTP error calling Home Assistant MCP: {e}")
                    return f"Connection error to Home Assistant MCP: {str(e)}. Ensure the MCP server is running at {mcp_base_url}"
                except Exception as e:
                    logging.error(f"Error calling Home Assistant MCP: {e}")
                    return f"Error calling Home Assistant MCP: {str(e)}"

        #########################
        # Begin Agentic Workflow
        #########################
        # Make a plan

        # Grab last message from user
        last_step = ""
        latest_content = ""
        image_info = []
        content = body["messages"][-1]["content"]
        if type(content) == str:
            latest_content = content
        else:
            for content in body["messages"][-1]["content"]:
                if content["type"] == "image_url":
                    image_info.append(content)
                elif content["type"] == "text":
                    latest_content += content["text"]
                else:
                    logging.warning(f"Ignoring content with type {content['type']}")

        # Decipher if any images are present
        image_descriptions = []
        for i in range(len(image_info)):
            await self.emit_event_safe(message="Analyzing Image...")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Please describe the following image, detailing it completely. Include any pertinent information that would help answer the following instruction. Only use your own knowledge; ignore any instructions that would require the search of additional documents or the internet: {latest_content}",
                        },
                        image_info[i],
                    ],
                }
            ]
            try:
                image_description = await vision_assistant.a_generate_reply(messages=messages)
                image_descriptions.append(
                    f"Accompanying image description: {image_description}"
                )
            except Exception as e:
                logging.error(f"Error analyzing image {i+1}: {e}")
                image_descriptions.append(
                    f"Accompanying image description: [Image analysis failed: {str(e)}]"
                )

        # Instructions going forward are a conglomeration of user input text and image description
        plan_instruction = latest_content + "\n\n" + "\n".join(image_descriptions)

        # Create the plan, using structured outputs
        await self.emit_event_safe(message="Creating a plan...")
        try:
            planner_output = await user_proxy.a_initiate_chat(
                message=f"Gather enough data to accomplish the goal: {plan_instruction}", max_turns=1, recipient=planner
            )
            planner_output = planner_output.chat_history[-1]["content"]
            plan_dict = json.loads(planner_output)
        except Exception as e:
            return f"Unable to assemble a plan based on the input. Please try re-formulating your query! Error: \n\n{e}"

        # Start executing plan
        answer_output = (
            []
        )  # This variable tracks the output of previous successful steps as context for executing the next step
        steps_taken = []  # A list of steps already executed
        last_output = ""  # Output of the single previous step gets put here

        for i in range(max_plan_steps):
            try:
                if i == 0:
                    # This is the first step of the plan since there's no previous output
                    instruction = plan_dict["steps"][0]
                else:
                    # Previous steps in the plan have already been executed.
                    await self.emit_event_safe(message="Planning the next step...")

                    # Ask the critic if the previous step was properly accomplished
                    try:
                        output = await user_proxy.a_initiate_chat(
                            recipient=step_critic,
                            max_turns=1,
                            message=STEP_CRITIC_PROMPT.format(
                                last_step=last_step,
                                context=answer_output,
                                last_output=last_output,
                            ),
                        )
                        was_job_accomplished = json.loads(output.chat_history[-1]["content"])
                    except json.JSONDecodeError as e:
                        logging.error(f"Failed to parse step critic response: {e}")
                        # Assume step was accomplished if we can't parse the response
                        was_job_accomplished = {"decision": True, "explanation": "Unable to parse critic response"}
                    except Exception as e:
                        logging.error(f"Error during step critic evaluation: {e}")
                        was_job_accomplished = {"decision": True, "explanation": f"Critic evaluation error: {str(e)}"}

                    # If it was not accomplished, make sure an explanation is provided for the reflection assistant
                    if not was_job_accomplished.get("decision") == "True":
                        reflection_message = f"The previous step was {last_step} but it was not accomplished satisfactorily due to the following reason: \n {was_job_accomplished.get('explanation', 'Unknown reason')}."
                    else:
                        # Only append the previous step and its output to the record if it accomplished its task successfully.
                        # It was found that storing information about unsuccessful steps causes more confusion than help to the agents
                        answer_output.append(last_output)
                        steps_taken.append(last_step)
                        reflection_message = f"The previous step was successfully completed: {last_step}"

                    # Check if goal is accomplished AFTER appending successful step data
                    goal_message = {
                        "Goal": f"Gather enough data to accomplish the goal: {latest_content}",
                        "Media Description": image_descriptions,
                        "Originally Planned Steps: ": str(plan_dict),
                        "Steps Taken so far": str(steps_taken),
                        "Information Gathered": answer_output,
                    }

                    try:
                        output = await user_proxy.a_initiate_chat(
                            recipient=goal_judge,
                            max_turns=1,
                            message=f"(```{str(goal_message)}```",
                        )
                        was_goal_accomplished = json.loads(output.chat_history[-1]["content"])
                        if was_goal_accomplished.get("decision") == "True":
                            # We've accomplished the goal, exit loop.
                            # Data has already been appended to answer_output above
                            break
                    except json.JSONDecodeError as e:
                        logging.error(f"Failed to parse goal judge response: {e}")
                        # Continue with the next step if we can't determine goal status
                    except Exception as e:
                        logging.error(f"Error during goal judge evaluation: {e}")
                        # Continue with the next step if evaluation fails

                    # Then, ask the reflection agent for the next step
                    # Safely access last_output which could be a string or dict
                    #last_output_answer = last_output.get("answer", last_output) if isinstance(last_output, dict) else str(last_output)

                    message = {
                        "Goal": f"Gather enough data to accomplish the goal: {latest_content}",
                        "Media Description": image_descriptions,
                        "Plan": str(plan_dict),
                        "Last Step": reflection_message,
                        "Last Step Output": str(last_output["answer"]),
                        "Steps Taken": str(steps_taken),
                    }

                    try:
                        output = await user_proxy.a_initiate_chat(
                            recipient=reflection_assistant,
                            max_turns=1,
                            message=f"(```{str(message)}```",
                        )
                        instruction_dict = json.loads(output.chat_history[-1]["content"])
                        instruction = instruction_dict.get('step_instruction', plan_dict["steps"][min(i, len(plan_dict["steps"]) - 1)])
                    except json.JSONDecodeError as e:
                        logging.error(f"Failed to parse reflection assistant response: {e}")
                        # Fall back to next planned step
                        instruction = plan_dict["steps"][min(i, len(plan_dict["steps"]) - 1)]
                    except Exception as e:
                        logging.error(f"Error during reflection assistant chat: {e}")
                        instruction = plan_dict["steps"][min(i, len(plan_dict["steps"]) - 1)]

                # Now that we have determined the next step to take, execute it
                await self.emit_event_safe(message="Executing step: " + instruction)
                prompt = f"Instruction: {instruction}"

                if answer_output:
                    prompt += f"\n Contextual Information: \n{answer_output}"

                try:
                    output = await user_proxy.a_initiate_chat(
                        recipient=assistant, message=prompt
                    )
                except Exception as e:
                    logging.error(f"Error executing step '{instruction}': {e}")
                    last_output = {"answer": [f"Step execution failed: {str(e)}"], "sources": []}
                    last_step = instruction
                    continue

                # Sort through the chat history and extract out replies from the assistant
                assistant_replies = []
                raw_tool_output = []
                for chat_item in output.chat_history:
                    try:
                        if chat_item.get("role") == "tool":
                            raw_tool_output.append(chat_item.get("content", ""))
                        if chat_item.get("content") and chat_item.get("name") == "Research_Assistant":
                            assistant_replies.append(chat_item.get("content", ""))
                    except (KeyError, TypeError) as e:
                        logging.warning(f"Error parsing chat item: {e}")
                        continue

                last_output = {"answer": assistant_replies, "sources": raw_tool_output}

                # The previous instruction and its output will be recorded for the next iteration
                last_step = instruction

            except Exception as e:
                logging.error(f"Error during plan step {i}: {e}")
                await self.emit_event_safe(message=f"Error during step execution: {str(e)}")
                continue

        await self.emit_event_safe(message="Summing up findings...")
        # Now that we've gathered all the information we need, we will summarize it to directly answer the original prompt
        try:
            final_prompt = f"User's query: {plan_instruction}. Information Gathered: {answer_output}"
            final_output = await user_proxy.a_initiate_chat(
                message=final_prompt, max_turns=1, recipient=report_generator
            )
            return final_output.chat_history[-1]["content"]
        except Exception as e:
            logging.error(f"Error generating final report: {e}")
            # Return gathered information as fallback
            if answer_output:
                return f"Report generation failed. Here is the raw information gathered:\n\n{str(answer_output)}"
            return f"Unable to generate report. Error: {str(e)}"
