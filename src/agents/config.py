from dataclasses import dataclass
from typing import List, Callable

@dataclass
class AgentConfig:
    """Agent 配置"""
    name: str
    prompt_fn: Callable  # 动态 prompt 函数
    tools: List = None
    middleware: List = None
    store: str = None
    
    def __post_init__(self):
        if self.tools is None:
            self.tools = []
        if self.middleware is None:
            self.middleware = []

# 导入所有 prompt 函数
from src.agents.dynamic_prompts import (
    decision_prompt_with_context,
    analysis_prompt_with_context,
    planning_prompt_with_context,
    execution_prompt_with_context,
    rag_prompt_with_context,
    ucagent_prompt_with_context,
)

# 预置配置 - 直接引用 prompt 函数
PRESETS = {
    "decision": AgentConfig(
        name="decision_agent",
        prompt_fn=decision_prompt_with_context,
        tools=[],
        middleware=["human_in_loop", "tool_retry"],
    ),
    "analysis": AgentConfig(
        name="analysis_agent",
        prompt_fn=analysis_prompt_with_context,
        tools=["retrieval"],
        middleware=["tool_retry", "pii_masking"],
    ),
    "planning": AgentConfig(
        name="planning_agent",
        prompt_fn=planning_prompt_with_context,
        tools=[],
        middleware=["tool_retry", "model_call_limit"],
    ),
    "execution": AgentConfig(
        name="execution_agent",
        prompt_fn=execution_prompt_with_context,
        tools=["retrieval", "mcp"],
        middleware=["tool_retry", "model_call_limit"],
    ),
    "rag_agent": AgentConfig(
        name="rag_agent",
        prompt_fn=rag_prompt_with_context,
        tools=["retrieval"],
        middleware=["tool_retry", "pii_masking", "filesystem", "human_in_loop"],
    ),
    "uc_agent": AgentConfig(
        name="uc_agent",
        prompt_fn=ucagent_prompt_with_context,
        tools=["retrieval", "mcp"],
        middleware=["tool_retry", "pii_masking", "filesystem", "human_in_loop"],
    ),
}