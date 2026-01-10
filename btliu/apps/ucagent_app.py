"""UC Agent application with agent management."""

from typing import Any, AsyncGenerator, Optional

from btliu.agents.factory import AgentFactory
from btliu.agents.dynamic_prompts import ucagent_prompt_with_context
from btliu.common import RuntimeContext
from btliu.config import paths
from btliu.tools import get_mcp_tools


class UcagentApp:
    """UC Agent application with agent management."""

    def __init__(
        self,
        context: RuntimeContext,
        store: Any,
        checkpoint: Any,
    ) -> None:
        """Initialize the UC Agent application.

        Args:
            context: Runtime context (config, doc_manager, but NO store)
            store: LangGraph store for cross-thread memory (injected by caller)
        """
        self.context = context
        self.store = store
        self.checkpoint = checkpoint
        self.ucagent = None

    async def get_agent(self) -> Any:
        """Initialize agent on first use.

        Returns:
            UC agent instance
        """
        llm = self.context["config"].chat()

        tools_getter = {
            "mcp": get_mcp_tools,
        }

        factory = AgentFactory(
            llm=llm,
            store=self.store,
            checkpoint=self.checkpoint,
            filesystem_root=str(paths.get_working_dir()),
            tools_getter=tools_getter,
        )

        self.ucagent = await factory.create(
            name="uc_agent",
            prompt_fn=ucagent_prompt_with_context,
            tool_names=["mcp"],
            middleware_names=["todo_list", "summarization", "tool_retry"],
        )
        return self.ucagent

    async def astream(
        self,
        query: dict[str, Any],
        config: Optional[dict] = None,
        runtime: Optional[RuntimeContext] = None,
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
            config=config,
            context=context,
            stream_mode="messages",
        ):
            yield chunk
