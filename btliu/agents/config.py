"""Agent configuration data structures."""

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
