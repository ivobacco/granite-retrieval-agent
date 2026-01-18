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
3. Control and query Home Assistant smart home devices - including lights, switches, climate/thermostats, covers/blinds, and more. Can also search for entities, get device states, and manage automations.
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
"""

ASSISTANT_PROMPT = """
You are an AI assistant that must complete a single user task.

INPUTS
- "Instruction:" — the task to complete. This has the highest priority.
- "Contextual Information:" — background that may include data, excerpts, or pre-fetched search results. Treat this as allowed evidence you may quote/summarize. It can be used even if you do not call any tools.

GENERAL POLICY
1) Follow "Instruction" over any conflicting context.
2) If the task can be done with the provided inputs (Instruction + Contextual Information), DO NOT call tools.
3) If essential info is missing and the task requires external facts, call exactly one tool at a time. Prefer a single decisive call over many speculative ones.
4) When you use tools, ground your answer ONLY in tool or provided-context outputs. Do not add unsupported facts.
5) If you still cannot complete the task after the allowed attempts, explain why and terminate.

STRUCTURE & OUTPUT
- Always produce one of:
  a) ##ANSWER## <your final answer>   (no headers before it)
  b) ##TERMINATE##   (only if truly impossible to complete)
- If using tools or provided excerpts as sources, include a brief "Sources:" line with identifiers (e.g., [1], [2]).

DECISION CHECKLIST (run mentally before answering)
- Q1: Can I answer directly from Instruction + Contextual Information? If yes → answer now (no tools).
- Q2: Is a tool REQUIRED to fetch missing facts? If yes → make one focused tool call that will likely resolve the task.
- Q3: After a tool call, do I have enough to answer? If yes → answer now. If not → at most 10 more targeted calls. Then either answer or terminate with a clear reason.

ERROR & MISSING-INFO HANDLING
- Scrutinize the returned `state` and the `attributes` dictionary. The answer is often a pre-calculated value here.
- If inputs are vague but still permit a reasonable interpretation, make the best good-faith assumption and proceed (state assumptions briefly in the answer).

STYLE
- Be direct, specific, and avoid boilerplate.

TOOL USE RULES
- Use only the tools provided here. Only one tool at a time.
- Cite from tool outputs or provided context; do not mix in outside knowledge.
- Use the homeassistant tool with operations: search_entities (find devices by domains or rooms), get_entity_state (check current state), control (operate devices like lights, climate, media players, covers, locks, fans, vacuums, alarms, scenes), automation_config (manage automations).
- Stop after a maximum of 5 total tool calls.

EXAMPLE CHAIN OF THOUGHT FOR HOME ASSISTANT TOOL USE
*User: "Dim the reading lamp."*
1. **Thought:** I don't know the ID for "reading lamp".
2. **Tool:** `search_entities(name_filter="reading")` -> Returns `[]` (Empty).
3. **Thought:** Search failed. User didn't specify a room, but "lamp" implies a light. I will widen scope to all lights.
4. **Tool:** `search_entities(domain="light")` -> Returns `['light.bedroom_reading', 'light.office_desk']`.
5. **Answer:** "I found two reading lights: one in the Bedroom and one in the Office. Which one would you like to dim?"

TERMINATION RULE
- If after following the above you cannot satisfy the Instruction, output only:
  ##TERMINATE##
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
                description="""Control and query Home Assistant smart home devices via MCP.
                Supports multiple operations:
                1. search_entities - Find devices by domain, area, state, or pattern (e.g., domain='light', area='bedroom', pattern='*motion*')
				  You can use filters, COMBINE multiple filters for better results: 
				  - domain: 'light', 'climate', 'binary_sensor', etc.
                  - area: 'bathroom', 'bedroom', etc.
                  - pattern: glob pattern like '*motion*', '*bedroom*'
                  - state: exact or comparison like 'on', '>50', '!=unavailable'
                  - changed_within: '5m', '1h', '24h'
                  - changed_after: ISO timestamp
                  - output: 'minimal', 'summary', 'full'
                  - sort_by: 'last_changed', 'last_updated', 'entity_id', 'state'
                  - sort_order: 'asc', 'desc'
                  - limit: max results number
                2. get_entity_state - Get state of specific entity:
                  - entity_id: required (e.g., 'sensor.bedroom_temperature')
                  - include_attributes: true/false (default true)
                3. control - Control devices:
                   - command: required (turn_on, turn_off, toggle, open, close, stop, set_position, set_tilt_position, set_temperature, set_hvac_mode, set_fan_mode, set_humidity)
                   - entity_id OR area_id: target device or area (at least one required)
                   - state: desired state
                   - Light: brightness (0-255), color_temp, rgb_color [r,g,b]
                   - Cover: position (0-100), tilt_position (0-100)
                   - Climate: temperature, target_temp_high, target_temp_low, hvac_mode (off|heat|cool|heat_cool|auto|dry|fan_only), fan_mode (auto|low|medium|high), humidity (0-100)
                4. automation_config - Manage automations:
                   - automation_action: required (create, update, delete, duplicate)
                   - automation_id: for update/delete/duplicate
                   - automation_config: dict for create/update
                """)
            @user_proxy.register_for_execution(name="homeassistant")
            async def do_homeassistant(
                operation: Annotated[str, "The MCP tool operation: 'search_entities', 'get_entity_state', 'control', or 'automation_config'"],
                # Common parameters
                entity_id: Annotated[Optional[str], "Entity ID for get_entity_state or control operations (e.g., 'light.living_room', 'climate.thermostat')"] = None,
                # search_entities parameters
                domain: Annotated[Optional[str], "Domain filter for search_entities (e.g., 'light', 'switch', 'climate', 'binary_sensor')"] = None,
                area: Annotated[Optional[str], "Area filter for search_entities (e.g., 'living_room', 'bedroom')"] = None,
                pattern: Annotated[Optional[str], "Glob pattern for search_entities (e.g., '*motion*', '*temperature*')"] = None,
                state: Annotated[Optional[str], "State filter for search_entities (e.g., 'on', 'off', '>50', '!=unavailable')"] = None,
                changed_within: Annotated[Optional[str], "Duration filter for search_entities (e.g., '5m', '1h', '24h')"] = None,
                changed_after: Annotated[Optional[str], "ISO timestamp filter for search_entities"] = None,
                output: Annotated[Optional[str], "Output format for search_entities: 'minimal', 'summary', or 'full'"] = "summary",
                sort_by: Annotated[Optional[str], "Sort field for search_entities: 'last_changed', 'last_updated', 'entity_id', or 'state'"] = None,
                sort_order: Annotated[Optional[str], "Sort order for search_entities: 'asc' or 'desc'"] = None,
                limit: Annotated[Optional[int], "Maximum results for search_entities"] = None,
                # get_entity_state parameters
                include_attributes: Annotated[Optional[bool], "Include entity attributes in get_entity_state response"] = True,
                # control parameters
                command: Annotated[Optional[str], "Control command: turn_on, turn_off, toggle, open, close, stop, set_position, set_tilt_position, set_temperature, set_hvac_mode, set_fan_mode, set_humidity"] = None,
                area_id: Annotated[Optional[str], "Area ID for control operations to control all devices in area"] = None,
                brightness: Annotated[Optional[int], "Brightness level 0-255 for light control"] = None,
                color_temp: Annotated[Optional[int], "Color temperature for light control"] = None,
                rgb_color: Annotated[Optional[list], "RGB color as [r, g, b] for light control"] = None,
                position: Annotated[Optional[int], "Position 0-100 for cover control"] = None,
                tilt_position: Annotated[Optional[int], "Tilt position 0-100 for cover control"] = None,
                temperature: Annotated[Optional[float], "Target temperature for climate control"] = None,
                target_temp_high: Annotated[Optional[float], "Target high temperature for climate control"] = None,
                target_temp_low: Annotated[Optional[float], "Target low temperature for climate control"] = None,
                hvac_mode: Annotated[Optional[str], "HVAC mode: off, heat, cool, heat_cool, auto, dry, fan_only"] = None,
                fan_mode: Annotated[Optional[str], "Fan mode: auto, low, medium, high"] = None,
                humidity: Annotated[Optional[int], "Target humidity 0-100 for climate control"] = None,
                # automation_config parameters
                automation_action: Annotated[Optional[str], "Automation action: create, update, delete, duplicate"] = None,
                automation_id: Annotated[Optional[str], "Automation ID for update/delete/duplicate operations"] = None,
                automation_config: Annotated[Optional[dict], "Automation configuration for create/update operations"] = None,
            ) -> str:
                """Execute Home Assistant MCP operations for smart home control and queries."""

                try:
                    async with aiohttp.ClientSession() as http_session:
                        # Initialize MCP session first
                        session_id = await init_mcp_session(http_session)

                        # Build the request based on operation type
                        if operation == "search_entities":
                            tool_name = "search_entities"
                            arguments = {}
                            if domain:
                                arguments["domain"] = domain
                            if area:
                                arguments["area"] = area
                            if pattern:
                                arguments["pattern"] = pattern
                            if state:
                                arguments["state"] = state
                            if changed_within:
                                arguments["changed_within"] = changed_within
                            if changed_after:
                                arguments["changed_after"] = changed_after
                            if output:
                                arguments["output"] = output
                            if sort_by:
                                arguments["sort_by"] = sort_by
                            if sort_order:
                                arguments["sort_order"] = sort_order
                            if limit:
                                arguments["limit"] = limit

                        elif operation == "get_entity_state":
                            if not entity_id:
                                return "Error: entity_id is required for get_entity_state operation"
                            tool_name = "get_entity_state"
                            arguments = {
                                "entity_id": entity_id,
                                "include_attributes": include_attributes
                            }

                        elif operation == "control":
                            if not command:
                                return "Error: command is required for control operation"
                            tool_name = "control"
                            arguments = {"command": command}
                            # Target selection (at least one required)
                            if entity_id:
                                arguments["entity_id"] = entity_id
                            if area_id:
                                arguments["area_id"] = area_id
                            # Common parameters
                            if state:
                                arguments["state"] = state
                            # Light control parameters
                            if brightness is not None:
                                arguments["brightness"] = brightness
                            if color_temp is not None:
                                arguments["color_temp"] = color_temp
                            if rgb_color:
                                arguments["rgb_color"] = rgb_color
                            # Cover control parameters
                            if position is not None:
                                arguments["position"] = position
                            if tilt_position is not None:
                                arguments["tilt_position"] = tilt_position
                            # Climate control parameters
                            if temperature is not None:
                                arguments["temperature"] = temperature
                            if target_temp_high is not None:
                                arguments["target_temp_high"] = target_temp_high
                            if target_temp_low is not None:
                                arguments["target_temp_low"] = target_temp_low
                            if hvac_mode:
                                arguments["hvac_mode"] = hvac_mode
                            if fan_mode:
                                arguments["fan_mode"] = fan_mode
                            if humidity is not None:
                                arguments["humidity"] = humidity

                        elif operation == "automation_config":
                            if not automation_action:
                                return "Error: automation_action is required for automation_config operation"
                            tool_name = "automation_config"
                            arguments = {"action": automation_action}
                            if automation_id:
                                arguments["automation_id"] = automation_id
                            if automation_config:
                                arguments["config"] = automation_config
                        else:
                            return f"Error: Unknown operation '{operation}'. Use 'search_entities', 'get_entity_state', 'control', or 'automation_config'."

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
