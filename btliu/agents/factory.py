"""Agent factory for creating configured agents."""

import asyncio
from typing import Any, Callable

from deepagents.backends import (
    FilesystemBackend,
)
from deepagents.middleware import FilesystemMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    PIIMiddleware,
    TodoListMiddleware,
    ToolRetryMiddleware,
)
from langchain.agents.middleware import (
    FilesystemFileSearchMiddleware,
    SummarizationMiddleware,
    ShellToolMiddleware,
)
from langgraph.store.memory import InMemoryStore

from btliu.common import RuntimeContext


class AgentFactory:
    """Factory for creating configured agents with tools and middleware.

    NOTE: Middleware are created fresh on each agent creation (not cached).
    This is required because middleware contain unpicklable components that
    should not be persisted to checkpoint.
    """

    def __init__(
        self,
        llm: Any,
        store: Any | None,
        filesystem_root: str,
        tools_getter: dict[str, Callable],
    ) -> None:
        """Initialize the agent factory.

        Args:
            llm: Language model instance
            store: Optional store instance (defaults to InMemoryStore)
            filesystem_root: Root directory for agent filesystem access (sandboxed)
            tools_getter: Dictionary mapping tool names to getter functions
        """
        self.llm = llm
        self.store = store or InMemoryStore()
        self.filesystem_root = filesystem_root
        self.tools_getter = tools_getter

    async def _get_tools(self, tool_names: list[str]) -> list:
        """Get tools by names.

        Args:
            tool_names: List of tool category names

        Returns:
            List of tool instances
        """
        tools = []
        for name in tool_names:
            if name in self.tools_getter:
                getter = self.tools_getter[name]
                if asyncio.iscoroutinefunction(getter):
                    tools.extend(await getter())
                else:
                    tools.extend(getter())
        return tools

    def _create_middleware(self, middleware_names: list[str]) -> list:
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
            elif name == "todo_list":
                middleware.append(TodoListMiddleware())
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
                        backend=FilesystemBackend(
                            root_dir=self.filesystem_root,
                            virtual_mode=True,  # Sandbox paths under root_dir
                        ),
                    )
                )
            elif name == "summarization":
                # 从 profile 读取容量（config.py 中设置的）
                max_tokens = None
                if hasattr(self.llm, "profile") and self.llm.profile:
                    max_tokens = self.llm.profile.get("max_input_tokens")

                fallback_tokens = int(max_tokens * 0.9) if max_tokens else 8000

                middleware.append(
                    SummarizationMiddleware(
                        model=self.llm,
                        trigger=[
                            ("fraction", 0.9),
                            ("tokens", fallback_tokens),
                            ("messages", 50),
                        ],
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

    async def create(
        self,
        name: str,
        prompt_fn: Callable,
        tool_names: list[str] | None = None,
        middleware_names: list[str] | None = None,
    ) -> Any:
        """Create an agent.

        Args:
            name: Agent name
            prompt_fn: Prompt function
            tool_names: Optional list of tool names
            middleware_names: Optional list of middleware names

        Returns:
            Configured agent instance
        """
        tool_names = tool_names or []
        middleware_names = middleware_names or []

        tools = await self._get_tools(tool_names)
        middleware = self._create_middleware(middleware_names)

        return create_agent(
            model=self.llm,
            tools=tools,
            middleware=[prompt_fn] + middleware,
            store=self.store,
            context_schema=RuntimeContext,
        )
