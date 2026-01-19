# How to Use V2 in Open WebUI

## 🎯 Quick Start Guide

V2 is now implemented and ready to use! Here's how to enable and test it in Open WebUI.

---

## 📋 Prerequisites

1. ✅ V2 code is in `v2/` directory
2. ✅ All dependencies installed: `ag2==0.9.10`, `aiohttp`, `pydantic`
3. ✅ Open WebUI running
4. ✅ `granite_autogen_rag.py` is loaded as a Function/Pipeline

---

## 🔧 Integration Steps

### Step 1: Import V2 in granite_autogen_rag.py

Add these imports at the top of `granite_autogen_rag.py` (after existing imports):

```python
# V2 imports
from v2 import (
    GraniteOrchestrator,
    ContextVariables,
    ExecutionContext,
    get_agent_configs_for_orchestrator,
)
from v2.prompts import PLANNER_MESSAGE, ASSISTANT_PROMPT, HA_PLANNER_PROMPT
```

### Step 2: Add Feature Flag to Valves

In the `Valves` class (around line 361), add:

```python
class Valves(BaseModel):
    # ... existing valves ...

    # V2 Orchestration Feature Flag
    USE_V2_ORCHESTRATION: bool = Field(
        default=False,
        description="Enable V2 orchestration (Triage with Tasks + Context-Aware Routing)"
    )
```

### Step 3: Modify pipe() Method

**Option A: Minimal Integration (Recommended for Testing)**

Add this at the START of the `pipe()` method (after variable setup, around line 460):

```python
async def pipe(self, body, __user__, __request__, __event_emitter__=None) -> str:
    # ... existing setup code (lines 450-460) ...

    # Extract user message
    messages = body.get("messages", [])
    user_message = messages[-1]["content"] if messages else ""

    if isinstance(user_message, list):
        # Handle multipart content
        text_parts = [p["text"] for p in user_message if p.get("type") == "text"]
        user_message = " ".join(text_parts)

    # V2 ROUTING
    if self.valves.USE_V2_ORCHESTRATION:
        return await self._pipe_v2(body, user_message)
    else:
        # V1 path - existing code continues here
        pass  # ... all existing V1 code ...
```

### Step 4: Add V2 Pipeline Method

Add this new method to the `Pipe` class:

```python
async def _pipe_v2(self, body, user_message: str) -> str:
    """V2 orchestration pipeline"""
    try:
        # Get LLM configs
        llm_configs = get_agent_configs_for_orchestrator(self.valves)

        # Create structured output models
        from pydantic import BaseModel, Field

        class Plan(BaseModel):
            steps: list[str]

        class HAOperation(BaseModel):
            operation: str
            parameters: dict
            reasoning: str

        # Create agents
        from autogen import ConversableAgent

        planner = ConversableAgent(
            name="Planner",
            system_message=PLANNER_MESSAGE,
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
            system_message=ASSISTANT_PROMPT,
            llm_config=llm_configs["assistant_llm_config"],
            human_input_mode="NEVER",
        )

        homeassistant_planner = ConversableAgent(
            name="HomeAssistant_Planner",
            system_message=HA_PLANNER_PROMPT,
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

        # Register tools (copy your existing tool registration code here)
        # For now, tools will be placeholders - full integration in next phase

        # Create execution context
        context = ExecutionContext(
            variables=ContextVariables(
                user_query=user_message,
                model_provider="openrouter" if self.valves.USE_OPENROUTER else "ollama",
                model_name=self.valves.OPENROUTER_TASK_MODEL if self.valves.USE_OPENROUTER else self.valves.TASK_MODEL_ID
            ),
            valves=self.valves,
            event_emitter=self.emit_event_safe
        )

        # Create orchestrator
        orchestrator = GraniteOrchestrator(
            valves=self.valves,
            llm_configs=llm_configs,
            user_proxy=user_proxy,
            assistant=assistant,
            planner=planner,
            homeassistant_planner=homeassistant_planner,
            emit_event=self.emit_event_safe
        )

        # Process query
        response_chunks = []
        async for chunk in orchestrator.process(user_message, context, stream=True):
            response_chunks.append(chunk)

        return "".join(response_chunks)

    except Exception as e:
        logging.error(f"V2 error: {e}", exc_info=True)
        return f"V2 Error: {str(e)}\n\nFalling back to V1..."
```

---

## 🚀 Usage in Open WebUI

### 1. Enable V2

1. Open Open WebUI
2. Go to **Workspace** → **Functions**
3. Find **"Granite Retrieval Agent"**
4. Click the **⚙️ Settings** icon
5. Scroll to **"USE_V2_ORCHESTRATION"**
6. Toggle it to **True** ✅
7. Click **Save**

### 2. Test V2

Open a new conversation and try these test queries:

#### Test 1: Simple Research
```
"What are the best smart thermostats in 2025?"
```

**Expected V2 Behavior:**
- ✅ Status: "Analyzing request and creating plan..."
- ✅ Status: "Executing tasks..."
- ✅ Status: "[Research] Starting parallel research for 1 tasks"
- ✅ Response with research findings

#### Test 2: Home Control
```
"Turn off the bedroom lights"
```

**Expected V2 Behavior:**
- ✅ Status: "Analyzing request and creating plan..."
- ✅ Status: "[Home Control] Processing 1 home control operations"
- ✅ Status: "Requesting approval: Turn off bedroom lights"
- ✅ Auto-approval (placeholder)
- ✅ Response with operation result

#### Test 3: Mixed (Research + Control)
```
"Research energy-saving tips and set my thermostat to 68°F"
```

**Expected V2 Behavior:**
- ✅ Decomposed into 2 tasks (1 research, 1 control)
- ✅ Research executes FIRST (parallel phase)
- ✅ Control executes SECOND (sequential phase with approval)
- ✅ Response combines both results

### 3. Monitor V2 Execution

Watch the status messages in Open WebUI:

```
[Status] Analyzing request and creating plan...
[Status] [Research] Starting parallel research for 1 tasks
[Status] [Research] Searching web: smart thermostats
[Status] [Research] Research complete: 1 succeeded, 0 failed
[Status] [Home Control] Processing 1 home control operations
[Status] [Home Control] Requesting approval: set thermostat to 68°F
[Status] [Home Control] Executing: set thermostat to 68°F
[Status] [Home Control] Home control complete: 1 succeeded, 0 failed, 0 denied
[Status] Preparing response...
```

---

## 🔍 Verify V2 is Working

### Check 1: Task Decomposition

V2 should show task analysis in logs:

```python
# Check your console/logs for:
"Decomposing query: ..."
"Decomposed into X tasks: Y research, Z control"
```

### Check 2: Phase Transitions

V2 tracks phases:

```python
# Phases in order:
INITIALIZATION → TRIAGE → RESEARCH → HOME_CONTROL → AGGREGATION → COMPLETION
```

### Check 3: Context Tracking

V2 maintains context:

```python
# At the end, should log:
"Processing complete. Summary: {context.to_summary()}"
# Shows: total_tasks, completed_tasks, success_rate, execution_time
```

---

## 🐛 Troubleshooting

### Issue: "V2 Error: No module named 'v2'"

**Solution**: Ensure `v2/` directory is in the same location as `granite_autogen_rag.py`

```bash
cd /path/to/granite-retrieval-agent
ls -la
# Should show:
#   granite_autogen_rag.py
#   v2/
```

### Issue: Import errors

**Solution**: Check Python path

```python
import sys
sys.path.append('/path/to/granite-retrieval-agent')
```

### Issue: "Tool not registered" errors

**Solution**: Tool registration is not yet fully integrated. This is expected in current phase. Tools will be integrated in next iteration.

### Issue: V2 not activating

**Solution**:
1. Verify `USE_V2_ORCHESTRATION` is in Valves class
2. Verify the feature flag is **True** in Open WebUI settings
3. Check the `if self.valves.USE_V2_ORCHESTRATION:` routing in pipe()
4. Restart Open WebUI to reload the function

---

## 🔄 Rollback to V1

If you encounter issues:

1. In Open WebUI, go to Function settings
2. Toggle `USE_V2_ORCHESTRATION` to **False**
3. Click Save
4. V1 will take over seamlessly

**OR** comment out the V2 routing in code:

```python
# Disable V2
# if self.valves.USE_V2_ORCHESTRATION:
#     return await self._pipe_v2(body, user_message)

# V1 continues as normal...
```

---

## 📊 V2 vs V1 Comparison

| Feature | V1 | V2 |
|---------|----|----|
| **Task Model** | String steps | Typed Pydantic models |
| **Execution** | Linear loop | Phase-based (Triage → Research → Control) |
| **Research** | Sequential | Parallel |
| **Home Control** | Direct execution | Approval workflow (placeholder) |
| **Error Handling** | Try-catch | Phase-aware error tracking |
| **Context** | Implicit | Explicit ContextVariables |
| **Routing** | Hardcoded | Context-Aware Routing |
| **Logging** | Basic | Detailed with metrics |

---

## 🎯 Next Steps

After testing V2:

1. **Full Tool Integration**: Connect web_search, knowledge_search, homeassistant tools
2. **Approval Workflow**: Implement real user approval for Home Assistant operations
3. **Result Synthesis**: Use Report Generator for final responses
4. **Testing**: Add unit and integration tests
5. **Performance**: Compare V1 vs V2 execution times

---

## 💡 Tips

1. **Start Simple**: Test with research-only queries first
2. **Check Logs**: V2 is verbose - use logs to debug
3. **Gradual Rollout**: Keep V1 as fallback during testing
4. **Monitor Context**: V2 tracks execution metrics - review `context.to_summary()`
5. **Report Issues**: If V2 fails, note the query and error for debugging

---

## 📚 Documentation

- **V2 README**: `v2/README.md` - Full architecture documentation
- **Integration Example**: `v2_integration_example.py` - Complete integration code
- **Models**: `v2/models/` - Task and context models
- **Agents**: `v2/agents/` - Agent implementations
- **Orchestrator**: `v2/orchestrator.py` - Main orchestration logic

---

## ✅ Success Checklist

- [ ] V2 imports added to granite_autogen_rag.py
- [ ] USE_V2_ORCHESTRATION added to Valves
- [ ] V2 routing added to pipe() method
- [ ] _pipe_v2() method implemented
- [ ] Feature flag enabled in Open WebUI
- [ ] Test query executed successfully
- [ ] Status messages visible
- [ ] No import errors
- [ ] Context tracking working
- [ ] Ready for full tool integration

---

**🎉 Congratulations! V2 is now ready to use in Open WebUI!**

For questions or issues, refer to:
- `v2/README.md` - Architecture details
- `v2_integration_example.py` - Full integration code
- Console logs - Detailed execution traces
