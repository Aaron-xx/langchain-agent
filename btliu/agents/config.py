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

    def __post_init__(self) -> None:
        if self.tools is None:
            self.tools = []
        if self.middleware is None:
            self.middleware = []


# Import all prompt functions
from btliu.agents.dynamic_prompts import (
    rag_prompt_with_context,
    ucagent_prompt_with_context,
)

# Preset configurations - directly reference prompt functions
PRESETS: dict[str, AgentConfig] = {
    "rag_agent": AgentConfig(
        name="rag_agent",
        prompt_fn=rag_prompt_with_context,
        tools=["retrieval"],
        middleware=[
            "todo_listpii_masking",
            "summarization",
            "human_in_loop",
            "tool_retry",
        ],
    ),
    "uc_agent": AgentConfig(
        name="uc_agent",
        prompt_fn=ucagent_prompt_with_context,
        tools=["retrieval", "mcp", "memory"],
        middleware=[
            "todo_listpii_masking",
            "summarization",
            "human_in_loop",
            "tool_retry",
        ],
    ),
}
