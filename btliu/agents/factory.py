"""Agent factory for creating configured agents."""

from typing import Any, Callable

from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend, StoreBackend
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
    ShellToolMiddleware
)
from langgraph.store.memory import InMemoryStore

from btliu.agents.config import AgentConfig, PRESETS
from btliu.common import RuntimeContext


class AgentFactory:
    """Factory for creating configured agents with tools and middleware."""

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
        self.MIDDLEWARE_MAP = self._build_middleware_map()

    def _build_middleware_map(self) -> dict[str, Any]:
        """Build middleware map with filesystem middleware using instance variables.

        Returns:
            Dictionary mapping middleware names to instances
        """
        return {
            "tool_retry": ToolRetryMiddleware(max_retries=3),
            "model_call_limit": ModelCallLimitMiddleware(run_limit=15),
            "pii_masking": PIIMiddleware(pii_type="email", strategy="mask"),
            "human_in_loop": HumanInTheLoopMiddleware(interrupt_on={"final_decision": True}),
            "filesystem": FilesystemMiddleware(
                backend=lambda rt: CompositeBackend(
                    default=StateBackend(rt),  # 临时工作文件（可自动清理）
                    routes={
                        "/memories/": StoreBackend(rt),  # 跨会话持久化
                        "/workspace/": FilesystemBackend(  # 真实文件系统访问
                            root_dir=self.filesystem_root,
                            virtual_mode=True,  # 沙箱模式，安全隔离
                        ),
                    },
                )
            ),
            "filesystemfilesearch": FilesystemFileSearchMiddleware(
                root_path=self.filesystem_root,
                use_ripgrep=True,
                max_file_size_mb=100,
            ),
            "summarization": SummarizationMiddleware(
                model=self.llm,
                trigger=("tokens", 4000),
                keep=("messages", 10),
                summary_prompt="请将以下对话历史进行摘要，保留关键决策点和技术细节：\n\n{messages}\n\n摘要:"
            ),
            "bash": ShellToolMiddleware(
                workspace_root=self.filesystem_root,
            ),
        }

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
        """Get middleware by names.

        Args:
            middleware_names: List of middleware names

        Returns:
            List of middleware instances
        """
        middleware = []
        for name in middleware_names:
            if name in self.MIDDLEWARE_MAP:
                middleware.append(self.MIDDLEWARE_MAP[name])
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