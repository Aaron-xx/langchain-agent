from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    PIIMiddleware,
    ToolRetryMiddleware,
    ModelCallLimitMiddleware,
)
from deepagents.middleware import FilesystemMiddleware
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from .config import AgentConfig, PRESETS
from langgraph.store.memory import InMemoryStore

class AgentFactory:
    """Agent 工厂"""
    
    MIDDLEWARE_MAP = {
        "tool_retry": ToolRetryMiddleware(max_retries=3),
        "model_call_limit": ModelCallLimitMiddleware(run_limit=15),
        "pii_masking": PIIMiddleware(pii_type="email", strategy="mask"),
        "human_in_loop": HumanInTheLoopMiddleware(interrupt_on={"final_decision": True}),
    }
    
    def __init__(self, llm, tools_dict, store=None):
        self.llm = llm
        self.tools_dict = tools_dict  # {"retrieval": [...], "mcp": [...]}
        self.store = store or InMemoryStore()

    def _get_tools(self, tool_names: list) -> list:
        """获取 tools"""
        tools = []
        for name in tool_names:
            if name in self.tools_dict:
                tools.extend(self.tools_dict[name])
        return tools

    def _get_middleware(self, middleware_names: list) -> list:
        """获取 middleware"""
        middleware = []
        for name in middleware_names:
            if name == "filesystem":
                middleware.append(FilesystemMiddleware(
                    backend=lambda rt: CompositeBackend(
                        default=StateBackend(rt),
                        routes={"/memories/": StoreBackend(rt)}
                    )
                ))
            elif name in self.MIDDLEWARE_MAP and self.MIDDLEWARE_MAP[name] != "lazy_create":
                middleware.append(self.MIDDLEWARE_MAP[name])
        return middleware
    
    def create(self, config: AgentConfig):
        """创建 agent"""
        tools = self._get_tools(config.tools)
        middleware = self._get_middleware(config.middleware)
        
        return create_agent(
            model=self.llm,
            tools=tools,
            middleware=[config.prompt_fn] + middleware,
            store=self.store,
        )
    
    def create_from_preset(self, preset_name: str):
        """从预置创建"""
        config = PRESETS.get(preset_name)
        if not config:
            raise ValueError(f"Unknown preset: {preset_name}")
        return self.create(config)
    
    def create_custom(self, name: str, prompt_fn, 
                     tool_names: list = None, middleware_names: list = None):
        """创建自定义 agent"""
        config = AgentConfig(
            name=name,
            prompt_fn=prompt_fn,
            tools=tool_names or [],
            middleware=middleware_names or [],
            store=self.store,
        )
        return self.create(config)