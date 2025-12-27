from btliu.agents import create_pre_agents
from btliu.common import RuntimeContext
from typing import Any, AsyncGenerator, Optional

class UcagentApp:
    """RAG Chain application with agent management."""

    def __init__(self, context: RuntimeContext) -> None:
        """Initialize the application with runtime context."""
        self.context = context
        self.agents = None
        self.factory = None
        self.ucagent = None

    async def get_agent(self) -> None:
        """Initialize agents and factory on first use."""
        agents, factory = await create_pre_agents(self.context)
        self.ucagent = agents.get("uc_agent")
        if self.ucagent is None:
            raise ValueError("UC agent not found")

    async def astream(self, query: str, runtime: Optional[RuntimeContext] = None, config: Optional[dict] = None) -> AsyncGenerator[str, Any]:
        """Stream UC query execution through agents.

        Args:
            query: User query string
            runtime: Optional runtime context override
            config: Optional LangGraph config (e.g., {"configurable": {"thread_id": "..."}})
        """

        if self.ucagent is None:
            await self.get_agent()

        context = runtime or self.context

        # Build config with thread_id if not provided
        if config is None:
            thread_id = context.get("thread_id")
            if thread_id:
                config = {"configurable": {"thread_id": thread_id}}

        async for chunk in self.ucagent.astream(query, context=context, config=config, stream_mode="messages"):
            token, metadata = chunk
            if hasattr(token, 'content') and token.content:
                yield token.content