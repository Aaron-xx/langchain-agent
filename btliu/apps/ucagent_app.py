"""UC Agent application with agent management.

This module provides the UcagentApp class for streaming UC agent query
execution through LangGraph agents.
"""

from typing import Any, AsyncGenerator, Optional

from btliu.agents import create_pre_agents
from btliu.common import RuntimeContext


class UcagentApp:
    """UC Agent application with agent management.

    This class manages UC agent initialization and provides streaming
    query execution capabilities.
    """

    def __init__(self, context: RuntimeContext) -> None:
        """Initialize the UC Agent application.

        Args:
            context: Runtime context containing configuration and services
        """
        self.context = context
        self.agents = None
        self.factory = None
        self.ucagent = None

    async def get_agent(self) -> Any:
        """Initialize agents and factory on first use.

        Returns:
            UC agent instance

        Raises:
            ValueError: If UC agent is not found
        """
        agents, factory = await create_pre_agents(self.context)
        self.ucagent = agents.get("uc_agent")
        if self.ucagent is None:
            raise ValueError("UC agent not found")
        return self.ucagent

    async def astream(
        self,
        query: str,
        runtime: Optional[RuntimeContext] = None,
        config: Optional[dict] = None,
    ) -> AsyncGenerator[Any, None]:
        """Stream UC agent query execution through agents.

        Args:
            query: User query string
            runtime: Optional runtime context override (defaults to self.context)
            config: Optional LangGraph config (e.g., {"configurable": {"thread_id": "..."}})

        Yields:
            (token, metadata) tuples from the agent stream
            - token: Message object (AIMessageChunk, ToolMessage, etc.)
            - metadata: Dictionary with langgraph_node, langgraph_step, etc.
        """
        if self.ucagent is None:
            await self.get_agent()

        context = runtime or self.context

        # Build config with thread_id if not provided
        if config is None:
            thread_id = context.get("thread_id")
            if thread_id:
                config = {"configurable": {"thread_id": thread_id}}

        # Stream from agent in messages mode
        # Returns (token, metadata) tuples - filtering handled by CLI layer
        async for chunk in self.ucagent.astream(
            query,
            context=context,
            config=config,
            stream_mode="messages",
        ):
            yield chunk
