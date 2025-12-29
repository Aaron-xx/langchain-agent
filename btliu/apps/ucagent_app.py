"""UC Agent application with agent management.

IMPORTANT: store/checkpointer are injected separately (not from context)
to avoid pickle issues. See cli.py WORKAROUND for full context.
"""

from typing import Any, AsyncGenerator, Optional

from btliu.agents import create_pre_agents
from btliu.common import RuntimeContext


class UcagentApp:
    """UC Agent application with agent management."""

    def __init__(
        self,
        context: RuntimeContext,
        store: Any = None,
        checkpointer: Any = None,
    ) -> None:
        """Initialize the UC Agent application.

        Args:
            context: Runtime context (config, doc_manager, but NO store/checkpointer)
            store: LangGraph store for cross-thread memory (injected by caller)
            checkpointer: LangGraph checkpointer for persistence (injected by caller)
        """
        self.context = context
        self.store = store
        self.checkpointer = checkpointer
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
        agents, factory = await create_pre_agents(
            self.context, store=self.store, checkpointer=self.checkpointer
        )
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
        """Stream UC agent query execution.

        Args:
            query: User query string
            runtime: Optional runtime context override
            config: Optional LangGraph config (e.g., {"configurable": {"thread_id": "..."}})

        Yields:
            (token, metadata) tuples from the agent stream
        """
        if self.ucagent is None:
            await self.get_agent()

        context = runtime or self.context

        if config is None:
            thread_id = context.get("thread_id")
            if thread_id:
                config = {"configurable": {"thread_id": thread_id}}

        async for chunk in self.ucagent.astream(
            query,
            context=context,
            config=config,
            stream_mode="messages",
        ):
            yield chunk
