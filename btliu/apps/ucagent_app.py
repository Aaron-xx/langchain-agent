"""UC Agent application with agent management."""

from typing import Any, AsyncGenerator, Optional

from btliu.agents import create_pre_agents
from btliu.common import RuntimeContext


class UcagentApp:
    """UC Agent application with agent management."""

    def __init__(
        self,
        context: RuntimeContext,
        store: Any = None,
    ) -> None:
        """Initialize the UC Agent application.

        Args:
            context: Runtime context (config, doc_manager, but NO store)
            store: LangGraph store for cross-thread memory (injected by caller)
        """
        self.context = context
        self.store = store
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
        agents, factory = await create_pre_agents(self.context, store=self.store)
        self.ucagent = agents.get("uc_agent")
        if self.ucagent is None:
            raise ValueError("UC agent not found")
        return self.ucagent

    async def astream(
        self,
        query: dict[str, Any],
        runtime: Optional[RuntimeContext] = None,
        config: Optional[dict] = None,
    ) -> AsyncGenerator[Any, None]:
        """Stream UC agent query execution.

        Args:
            query: Query dictionary with messages (e.g., {"messages": [{"role": "user", "content": "..."}]})
            runtime: Optional runtime context override
            config: Optional LangGraph config (e.g., {"configurable": {"thread_id": "..."}})

        Yields:
            (token, metadata) tuples from the agent stream
        """
        if self.ucagent is None:
            await self.get_agent()

        context = runtime or self.context

        async for chunk in self.ucagent.astream(
            query,
            context=context,
            config=config,
            stream_mode="messages",
        ):
            yield chunk
