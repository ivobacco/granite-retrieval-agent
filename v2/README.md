

# Granite Retrieval Agent V2 - Implementation Progress

## 🎯 Architecture Overview

V2 transforms the agent execution flow using AG2 advanced patterns:
- **Triage with Tasks**: Intelligent task decomposition with typed Pydantic models
- **Context-Aware Routing**: Dynamic distribution to specialized agents
- **Sequential Safety**: Research ALWAYS executes before home control
- **Async Approvals**: User approval workflow for Home Assistant operations
- **SocietyOfMindAgent**: Clean external interface wrapping multi-agent system

## 📁 Current Implementation Status

### ✅ COMPLETED (Phase 1)

#### 1. Directory Structure
```
v2/
├── models/          ✅ Task and context models
├── prompts/         ✅ Centralized system prompts
├── config/          ✅ Provider configuration utilities
├── agents/          ⏳ Agent implementations (next)
├── executors/       ⏳ Async executors (next)
├── tools/           ⏳ Tool wrappers (next)
└── tests/           ⏳ Test suite (next)
```

#### 2. Pydantic Models (`models/`)

**`models/tasks.py`** ✅
- `TaskBase`: Base model with common fields (task_id, status, priority, timestamps, error handling)
- `ResearchTask`: Information gathering from web/knowledge bases
  - Supports multiple sources (web, knowledge, both)
  - Result synthesis capability
  - Max results configuration
- `HomeControlTask`: Home Assistant operations
  - Natural language instruction support
  - Structured operation/parameters from HA Planner
  - Approval workflow fields (status, timeout, timestamps)
  - MCP response tracking
- `CompositeTask`: Container for multiple subtasks
  - Sequential or parallel execution strategies
  - Success rate tracking
  - Helper methods for subtask management
- `TaskDecompositionResult`: Output from Triage Manager
  - Intent analysis
  - Execution ordering
  - Duration estimation

**Enums:**
- `TaskType`: RESEARCH, HOME_CONTROL, COMPOSITE
- `TaskPriority`: LOW, MEDIUM, HIGH, CRITICAL
- `TaskStatus`: PENDING, IN_PROGRESS, WAITING_APPROVAL, COMPLETED, FAILED, CANCELLED

**`models/context.py`** ✅
- `ContextVariables`: Shared state across agents
  - Research context (results, sources, knowledge retrieved)
  - Home automation context (state, approvals, operations)
  - Execution tracking (completed/failed tasks, current task)
  - Provider context (model, temperature)
  - Performance metrics (tool calls, execution time)
  - Helper methods for task management and approval workflow
- `ExecutionContext`: Master context object
  - Context variables
  - Task list
  - Phase tracking (with history and duration)
  - Event handling system
  - Streaming support
  - Error tracking with retry logic
  - Valves and event emitter from Pipe
  - Helper methods for phase transitions, events, errors
- `AgentMessage`: Structured agent-to-agent communication
- `ExecutionPhase` enum: INITIALIZATION, TRIAGE, RESEARCH, HOME_CONTROL, AGGREGATION, COMPLETION, ERROR

#### 3. System Prompts (`prompts/`)

**`prompts/system_prompts.py`** ✅

All V1 prompts extracted and centralized:
- `PLANNER_MESSAGE`: Initial task decomposition with examples
- `ASSISTANT_PROMPT`: Tool execution specialist (web_search, personal_knowledge_search, homeassistant)
- `GOAL_JUDGE_PROMPT`: Objective evaluation of goal completion
- `REFLECTION_ASSISTANT_PROMPT`: Strategic planning for next steps
- `STEP_CRITIC_PROMPT`: Individual step evaluation
- `SEARCH_QUERY_GENERATION_PROMPT`: Web search query decomposition
- `HA_PLANNER_PROMPT`: Natural language to MCP tool translation (25 tools documented)
- `REPORT_WRITER_PROMPT`: Final synthesis with source citations

#### 4. Provider Configuration (`config/`)

**`config/providers.py`** ✅
- `ProviderConfig`: Base configuration model
- `OllamaConfig`: Local Ollama configuration (with num_ctx)
- `OpenRouterConfig`: Cloud OpenRouter configuration
- `ProviderManager`:
  - Builds all LLM configs from Valves
  - Supports both Ollama and OpenRouter
  - Separate configs for base, quick, and vision models
  - AG2-compatible config format
  - Logging support
- `create_structured_output_config()`: Helper for Pydantic response formats
- `get_agent_configs_for_orchestrator()`: Main entry point for V2

## 🚧 NEXT STEPS (Phase 2)

### 1. Base Agent Classes (`agents/`)
- [ ] `base_agent.py`: Abstract base class for all agents
- [ ] `research_coordinator.py`: Orchestrates parallel research
- [ ] `web_search_agent.py`: Web search specialist
- [ ] `knowledge_agent.py`: Personal knowledge specialist
- [ ] `home_control_manager.py`: Home automation orchestration

### 2. Orchestration Layer (`orchestrator.py`)
- [ ] `GraniteOrchestrator`: SocietyOfMindAgent wrapper
- [ ] `TriageManager`: Task decomposition and routing
- [ ] `ContextTracker`: Context management utilities

### 3. Async Executors (`executors/`)
- [ ] `homeassistant_executor.py`: Async MCP execution
- [ ] `ApprovalHandler`: User approval workflow

### 4. Tool Wrappers (`tools/`)
- [ ] Wrap existing V1 tools with clean interfaces
- [ ] `web_search.py`, `knowledge_search.py`, `homeassistant_mcp.py`

### 5. Pipe Integration
- [ ] Modify `granite_autogen_rag.py` to use V2 orchestrator
- [ ] Add feature flag `USE_V2_ORCHESTRATION`
- [ ] Maintain backward compatibility

## 📊 Implementation Metrics

| Component | Status | Files | Lines of Code | Test Coverage |
|-----------|--------|-------|---------------|---------------|
| Models | ✅ Complete | 3 | ~750 | 0% (pending) |
| Prompts | ✅ Complete | 2 | ~350 | N/A |
| Config | ✅ Complete | 2 | ~250 | 0% (pending) |
| Agents | ⏳ Pending | 0 | 0 | 0% |
| Executors | ⏳ Pending | 0 | 0 | 0% |
| Tools | ⏳ Pending | 0 | 0 | 0% |
| Orchestrator | ⏳ Pending | 0 | 0 | 0% |
| Tests | ⏳ Pending | 0 | 0 | 0% |

## 🎨 Key Design Decisions

### 1. Pydantic Models for Type Safety
- All tasks are typed (ResearchTask, HomeControlTask, CompositeTask)
- Context variables fully structured
- Validation at runtime prevents errors
- Easy serialization for logging/debugging

### 2. Phase-Based Execution
- Explicit phase tracking (TRIAGE → RESEARCH → HOME_CONTROL → AGGREGATION)
- Phase history with duration metrics
- Error handling per phase
- Easy to add new phases

### 3. Approval Workflow
- Tasks can require user approval (HomeControlTask)
- Timeout support (default 300s)
- Approval status tracking (pending/approved/denied/timeout)
- Event-driven architecture

### 4. Context-Aware Routing
- ContextVariables shared across all agents
- Agents read and update context
- Sequential execution enforced by phase transitions
- Research results flow into home control decisions

### 5. Provider Flexibility
- Supports Ollama (local) and OpenRouter (cloud)
- Different models for different agents (base, quick, vision)
- Structured output support via Pydantic
- Configuration centralized in ProviderManager

## 📝 Usage Examples

### Creating Tasks

```python
from v2.models import ResearchTask, HomeControlTask, TaskPriority

# Research task
research = ResearchTask(
    query="smart thermostat best practices",
    sources=["web", "knowledge"],
    priority=TaskPriority.HIGH
)

# Home control task with approval
control = HomeControlTask(
    instruction="set thermostat to 70°F",
    requires_approval=True,
    approval_timeout=300,
    priority=TaskPriority.HIGH
)
```

### Building Execution Context

```python
from v2.models import ContextVariables, ExecutionContext

# Create context
context = ExecutionContext(
    variables=ContextVariables(
        user_query="Research thermostats and set mine to optimal",
        model_provider="openrouter",
        model_name="xiaomi/mimo-v2-flash:free"
    ),
    valves=pipe.valves,
    event_emitter=emit_event_safe
)

# Track task completion
context.variables.mark_task_completed(task_id)

# Request approval
context.variables.request_approval(task_id)
```

### Provider Configuration

```python
from v2.config import get_agent_configs_for_orchestrator

# Get all LLM configurations
configs = get_agent_configs_for_orchestrator(valves)

# Use in agents
planner = ConversableAgent(
    name="Planner",
    llm_config=configs["planner_llm_config"],
    system_message=PLANNER_MESSAGE
)
```

## 🔄 Migration from V1

V1 code is preserved. V2 introduces:
1. **Structured tasks** instead of string-based plan steps
2. **Phase-based execution** instead of linear loops
3. **Context-aware agents** instead of isolated tool calls
4. **Async approval workflow** instead of direct execution
5. **Type safety** via Pydantic models

Feature flag will allow gradual rollout:
```python
if valves.USE_V2_ORCHESTRATION:
    # V2 path
    orchestrator = GraniteOrchestrator(...)
else:
    # V1 path (existing code)
    ...
```

## 🧪 Testing Strategy

Next phase will include:
- Unit tests for all Pydantic models
- Integration tests for agent coordination
- End-to-end tests for full workflows
- Mock MCP server for Home Assistant tests
- Performance benchmarks (V1 vs V2)

## 📚 Dependencies

All V1 dependencies preserved:
- `ag2==0.9.10`
- `ag2[ollama]==0.9.10`
- `ag2[openai]==0.9.10`
- `aiohttp`
- `pydantic` (already used in V1)

## 🎯 Success Criteria

V2 will be considered complete when:
- ✅ All Phase 1 models and config implemented
- ⏳ All agents implemented and tested
- ⏳ Orchestrator functional with async approvals
- ⏳ Feature flag integration in Pipe
- ⏳ End-to-end tests passing
- ⏳ Performance equal or better than V1
- ⏳ Documentation complete

## 📞 Next Session Tasks

1. Implement `base_agent.py` with common agent functionality
2. Implement `research_coordinator.py` with parallel execution
3. Implement `web_search_agent.py` and `knowledge_agent.py`
4. Begin `orchestrator.py` with GraniteOrchestrator and TriageManager
5. Write unit tests for models

---

**Implementation Date**: January 19, 2026
**Status**: Phase 1 Complete (Models, Prompts, Config)
**Next Milestone**: Phase 2 (Agents and Orchestration)
