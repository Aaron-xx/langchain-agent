from src.agents import create_pre_agents
from src.common import RuntimeContext
from typing import Any, AsyncGenerator, Optional

class RAGApp:
    """RAG Chain application with agent management."""

    def __init__(self, context: RuntimeContext) -> None:
        """Initialize the application with runtime context."""
        self.context = context
        self.agents = None
        self.factory = None
        self.ragagent = None

    async def get_agent(self) -> None:
        """Initialize agents and factory on first use."""
        self.agents, self.factory = await create_pre_agents(self.context)
        self.ragagent = self.agents["rag_agent"]

    async def astream(self, query: str, runtime: Optional[RuntimeContext] = None) -> AsyncGenerator[str, Any]:
        """Stream RAG query execution through agents."""

        if self.ragagent is None:
            await self.get_agent()

        context = runtime or self.context

        async for chunk in self.ragagent.astream(
            query,
            stream_mode="messages"
        ):
            token, metadata = chunk
            if hasattr(token, 'content') and token.content:
                yield token.content