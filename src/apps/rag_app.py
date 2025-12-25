"""RAG application with agent management."""

from typing import Any, AsyncGenerator

from src.agents import create_pre_agents
from src.common import RuntimeContext


class RAGApp:
    """RAG application with agent management."""

    def __init__(self, context: RuntimeContext) -> None:
        """Initialize the application with runtime context.

        Args:
            context: Runtime context containing configuration and services
        """
        self.context = context
        self.agents: dict[str, Any] | None = None
        self.factory: Any | None = None
        self.ragagent: Any | None = None

    async def get_agent(self) -> Any:
        """Initialize agents and factory on first use.

        Returns:
            RAG agent instance
        """
        agents, factory = await create_pre_agents(self.context)
        self.ragagent = agents.get("rag_agent")
        if self.ragagent is None:
            raise ValueError("RAG agent not found")
        return self.ragagent

    async def astream(
        self,
        query: dict[str, Any],
        runtime: RuntimeContext | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream RAG query execution through agents.

        Args:
            query: Query dictionary
            runtime: Optional runtime context

        Yields:
            Streamed content chunks
        """
        if self.ragagent is None:
            await self.get_agent()

        context = runtime or self.context

        async for chunk in self.ragagent.astream(
            query,
            context=context,
            stream_mode="messages",
        ):
            token, metadata = chunk
            if hasattr(token, "content") and token.content:
                yield token.content
