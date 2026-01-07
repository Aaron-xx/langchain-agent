"""RAG application with agent management.

This module provides the RAGApp class for streaming RAG query execution
through LangGraph agents.
"""

from typing import Any, AsyncGenerator

from btliu.agents.factory import AgentFactory
from btliu.agents.dynamic_prompts import rag_prompt_with_context
from btliu.common import RuntimeContext
from btliu.config import paths
from btliu.tools import get_all_retrievers


class RAGApp:
    """RAG application with agent management.

    This class manages RAG agent initialization and provides streaming
    query execution capabilities.
    """

    def __init__(
        self,
        context: RuntimeContext,
        store: Any = None,
    ) -> None:
        """Initialize the RAG application.

        Args:
            context: Runtime context containing configuration and services
            store: Optional store instance (for cross-thread memory)
        """
        self.context = context
        self.store = store
        self.ragagent: Any | None = None

    async def get_agent(self) -> Any:
        """Initialize agent on first use.

        Returns:
            RAG agent instance
        """
        llm = self.context.config.chat()

        tools_getter = {
            "retrieval": get_all_retrievers,
        }

        factory = AgentFactory(
            llm=llm,
            store=self.store,
            filesystem_root=str(paths.get_working_dir()),
            tools_getter=tools_getter,
        )

        self.ragagent = await factory.create(
            name="rag_agent",
            prompt_fn=rag_prompt_with_context,
            tool_names=["retrieval"],
            middleware_names=["summarization", "tool_retry"],
        )
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

        # Stream from agent in messages mode
        async for chunk in self.ragagent.astream(
            query,
            context=context,
            config=config,
            stream_mode="messages",
        ):
            yield chunk
