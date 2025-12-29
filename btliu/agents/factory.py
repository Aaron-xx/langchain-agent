"""Agent factory for creating configured agents."""

from typing import Any, Callable

from deepagents.backends import (
    CompositeBackend,
    FilesystemBackend,
    StateBackend,
    StoreBackend,
)
from deepagents.middleware import FilesystemMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    PIIMiddleware,
    ToolRetryMiddleware,
)
from langchain.agents.middleware import (
    FilesystemFileSearchMiddleware,
    SummarizationMiddleware,
    ShellToolMiddleware,
)
from langgraph.store.memory import InMemoryStore

from btliu.agents.config import AgentConfig, PRESETS
from btliu.common import RuntimeContext


class AgentFactory:
    """Factory for creating configured agents with tools and middleware.

    NOTE: Middleware are created fresh on each agent creation (not cached).
    This is required because unpicklable middleware are filtered from checkpoint
    state by the custom serializer in btliu/cli/cli.py (see WORKAROUND comment).
    """

    def __init__(
        self,
        llm: Any,
        tools_dict: dict[str, list],
        store: Any | None = None,
        checkpointer: Any | None = None,
        filesystem_root: str = "/tmp/agent_workspace",
    ) -> None:
        """Initialize the agent factory.

        Args:
            llm: Language model instance
            tools_dict: Dictionary mapping tool names to tool instances
            store: Optional store instance (defaults to InMemoryStore)
            filesystem_root: Root directory for agent filesystem access (sandboxed)
        """
        self.llm = llm
        self.tools_dict = tools_dict  # {"retrieval": [...], "mcp": [...]}
        self.store = store or InMemoryStore()
        self.checkpointer = checkpointer
        self.filesystem_root = filesystem_root

    def _get_tools(self, tool_names: list[str]) -> list:
        """Get tools by names.

        Args:
            tool_names: List of tool category names

        Returns:
            List of tool instances
        """
        tools = []
        for name in tool_names:
            if name in self.tools_dict:
                tools.extend(self.tools_dict[name])
        return tools

    def _get_middleware(self, middleware_names: list[str]) -> list:
        """Get middleware by names, creating fresh instances each time.

        Args:
            middleware_names: List of middleware names

        Returns:
            List of middleware instances
        """
        middleware = []
        for name in middleware_names:
            if name == "tool_retry":
                middleware.append(ToolRetryMiddleware(max_retries=3))
            elif name == "model_call_limit":
                middleware.append(ModelCallLimitMiddleware(run_limit=15))
            elif name == "pii_masking":
                middleware.append(PIIMiddleware(pii_type="email", strategy="mask"))
            elif name == "human_in_loop":
                middleware.append(
                    HumanInTheLoopMiddleware(interrupt_on={"final_decision": True})
                )
            elif name == "filesystem":
                middleware.append(
                    FilesystemMiddleware(
                        backend=lambda rt: CompositeBackend(
                            default=StateBackend(rt),
                            routes={
                                "/memories/": StoreBackend(rt),
                                "/workspace/": FilesystemBackend(
                                    root_dir=self.filesystem_root,
                                    virtual_mode=True,
                                ),
                            },
                        )
                    )
                )
            elif name == "summarization":
                middleware.append(
                    SummarizationMiddleware(
                        model=self.llm,
                        trigger=("tokens", 4000),
                        keep=("messages", 10),
                        summary_prompt="请将以下对话历史进行摘要，保留关键决策点和技术细节：\n\n{messages}\n\n摘要:",
                    )
                )
            elif name == "filesystemfilesearch":
                middleware.append(
                    FilesystemFileSearchMiddleware(
                        root_path=self.filesystem_root,
                        use_ripgrep=True,
                        max_file_size_mb=100,
                    )
                )
            elif name == "bash":
                middleware.append(
                    ShellToolMiddleware(
                        workspace_root=self.filesystem_root,
                    )
                )
        return middleware

    def create(self, config: AgentConfig) -> Any:
        """Create an agent from configuration.

        Args:
            config: Agent configuration

        Returns:
            Configured agent instance
        """
        tools = self._get_tools(config.tools)
        middleware = self._get_middleware(config.middleware)

        return create_agent(
            model=self.llm,
            tools=tools,
            middleware=[config.prompt_fn] + middleware,
            store=self.store,
            checkpointer=self.checkpointer,
            context_schema=RuntimeContext,
        )

    def create_from_preset(self, preset_name: str) -> Any:
        """Create agent from preset name.

        Args:
            preset_name: Name of the preset configuration

        Returns:
            Configured agent instance

        Raises:
            ValueError: If preset name is not found
        """
        config = PRESETS.get(preset_name)
        if not config:
            raise ValueError(f"Unknown preset: {preset_name}")
        return self.create(config)

    def create_custom(
        self,
        name: str,
        prompt_fn: Callable,
        tool_names: list[str] | None = None,
        middleware_names: list[str] | None = None,
    ) -> Any:
        """Create custom agent with specified parameters.

        Args:
            name: Agent name
            prompt_fn: Dynamic prompt function
            tool_names: Optional list of tool names
            middleware_names: Optional list of middleware names

        Returns:
            Configured agent instance
        """
        config = AgentConfig(
            name=name,
            prompt_fn=prompt_fn,
            tools=tool_names or [],
            middleware=middleware_names or [],
            checkpointer=self.checkpointer,
            store=self.store,
        )
        return self.create(config)
