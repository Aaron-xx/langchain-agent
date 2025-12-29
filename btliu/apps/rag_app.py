"""RAG application with agent management.

This module provides the RAGApp class for streaming RAG query execution
through LangGraph agents.
"""

from typing import Any, AsyncGenerator

from btliu.agents import create_pre_agents
from btliu.common import RuntimeContext


class RAGApp:
    """RAG application with agent management.

    This class manages RAG agent initialization and provides streaming
    query execution capabilities.
    """

    def __init__(
        self,
        context: RuntimeContext,
        store: Any = None,
        checkpointer: Any = None,
    ) -> None:
        """Initialize the RAG application.

        Args:
            context: Runtime context containing configuration and services
            store: Optional store instance (for cross-thread memory)
            checkpointer: Optional checkpointer instance (for persistence)
        """
        self.context = context
        self.store = store
        self.checkpointer = checkpointer
        self.agents: dict[str, Any] | None = None
        self.factory: Any | None = None
        self.ragagent: Any | None = None

    async def get_agent(self) -> Any:
        """Initialize agents and factory on first use.

        Returns:
            RAG agent instance

        Raises:
            ValueError: If RAG agent is not found
        """
        agents, factory = await create_pre_agents(
            self.context, store=self.store, checkpointer=self.checkpointer
        )
        self.ragagent = agents.get("rag_agent")
        if self.ragagent is None:
            raise ValueError("RAG agent not found")
        return self.ragagent

    async def astream(
        self,
        query: dict[str, Any],
        runtime: RuntimeContext | None = None,
        config: dict | None = None,
    ) -> AsyncGenerator[Any, None]:
        """Stream RAG query execution through agents.

        Args:
            query: Query dictionary with messages
            runtime: Optional runtime context override (defaults to self.context)
            config: Optional LangGraph config (e.g., {"configurable": {"thread_id": "..."}})

        Yields:
            (token, metadata) tuples from the agent stream
            - token: Message object (AIMessageChunk, ToolMessage, etc.)
            - metadata: Dictionary with langgraph_node, langgraph_step, etc.
        """
        if self.ragagent is None:
            await self.get_agent()

        context = runtime or self.context

        # Build config with thread_id if not provided
        if config is None:
            thread_id = context.get("thread_id")
            if thread_id:
                config = {"configurable": {"thread_id": thread_id}}

        # Stream from agent in messages mode
        async for chunk in self.ragagent.astream(
            query,
            context=context,
            config=config,
            stream_mode="messages",
        ):
            yield chunk
