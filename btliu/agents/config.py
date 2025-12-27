"""Agent configuration presets and data structures."""

from dataclasses import dataclass
from typing import Any


@dataclass
class AgentConfig:
    """Agent configuration.

    Attributes:
        name: Agent name
        prompt_fn: Dynamic prompt function
        tools: List of tool names to use
        middleware: List of middleware names to apply
        store: Store instance
    """

    name: str
    prompt_fn: Any  # Dynamic prompt function
    tools: list[str] | None = None
    middleware: list[str] | None = None
    store: Any | None = None
    checkpointer: Any | None = None

    def __post_init__(self) -> None:
        if self.tools is None:
            self.tools = []
        if self.middleware is None:
            self.middleware = []


# Import all prompt functions
from btliu.agents.dynamic_prompts import (
    analysis_prompt_with_context,
    decision_prompt_with_context,
    execution_prompt_with_context,
    planning_prompt_with_context,
    rag_prompt_with_context,
    ucagent_prompt_with_context,
)

# Preset configurations - directly reference prompt functions
PRESETS: dict[str, AgentConfig] = {
    "decision": AgentConfig(
        name="decision_agent",
        prompt_fn=decision_prompt_with_context,
        tools=[],
        middleware=["human_in_loop", "tool_retry"],
    ),
    "analysis": AgentConfig(
        name="analysis_agent",
        prompt_fn=analysis_prompt_with_context,
        tools=["retrieval"],
        middleware=["tool_retry", "pii_masking"],
    ),
    "planning": AgentConfig(
        name="planning_agent",
        prompt_fn=planning_prompt_with_context,
        tools=[],
        middleware=["tool_retry", "model_call_limit"],
    ),
    "execution": AgentConfig(
        name="execution_agent",
        prompt_fn=execution_prompt_with_context,
        tools=["retrieval", "mcp"],
        middleware=["tool_retry", "model_call_limit"],
    ),
    "rag_agent": AgentConfig(
        name="rag_agent",
        prompt_fn=rag_prompt_with_context,
        tools=["retrieval"],
        middleware=["tool_retry", "pii_masking", "human_in_loop", "summarization"],
    ),
    "uc_agent": AgentConfig(
        name="uc_agent",
        prompt_fn=ucagent_prompt_with_context,
        tools=["retrieval", "mcp", "memory"],
        middleware=["tool_retry", "pii_masking", "filesystem", "filesystemfilesearch", "human_in_loop", "summarization", "bash"],
    ),
}
