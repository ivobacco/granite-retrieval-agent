"""
Prompts package for V2 architecture.

Exports system prompts for all agents:
- PLANNER_MESSAGE: Task decomposition and planning
- ASSISTANT_PROMPT: Task execution specialist
- HA_PLANNER_PROMPT: Home Assistant operation planning
- REPORT_WRITER_PROMPT: Final report generation
- GOAL_JUDGE_PROMPT: Goal completion evaluation
- REFLECTION_ASSISTANT_PROMPT: Next step planning
- STEP_CRITIC_PROMPT: Step validation
- SEARCH_QUERY_GENERATION_PROMPT: Web search query generation
"""

from .system_prompts import (
    PLANNER_MESSAGE,
    ASSISTANT_PROMPT,
    GOAL_JUDGE_PROMPT,
    REFLECTION_ASSISTANT_PROMPT,
    STEP_CRITIC_PROMPT,
    SEARCH_QUERY_GENERATION_PROMPT,
    HA_PLANNER_PROMPT,
    REPORT_WRITER_PROMPT,
)

__all__ = [
    "PLANNER_MESSAGE",
    "ASSISTANT_PROMPT",
    "GOAL_JUDGE_PROMPT",
    "REFLECTION_ASSISTANT_PROMPT",
    "STEP_CRITIC_PROMPT",
    "SEARCH_QUERY_GENERATION_PROMPT",
    "HA_PLANNER_PROMPT",
    "REPORT_WRITER_PROMPT",
]
