"""Agent creation and management."""

import asyncio
from typing import Any

from langgraph.store.memory import InMemoryStore

from src.agents.factory import AgentFactory
from src.agents.config import PRESETS
from src.common import RuntimeContext
from src.tools import get_all_retrievers, get_mcp_tools


async def create_pre_agents(
    context: RuntimeContext,
) -> tuple[dict[str, Any], AgentFactory]:
    """Create all preset agents with their factory.

    Args:
        context: Runtime context containing configuration

    Returns:
        Tuple of (agents dictionary, AgentFactory instance)
    """
    config = context["config"]
    llm = config.chat()

    retrieval_tools = get_all_retrievers()
    try:
        mcp_tools = await get_mcp_tools()
    except Exception:
        mcp_tools = []

    tools_dict = {
        "retrieval": retrieval_tools,
        "mcp": mcp_tools,
    }
    store = InMemoryStore()

    factory = AgentFactory(llm, tools_dict, store)

    agents: dict[str, Any] = {}
    for preset_name in PRESETS:
        agents[preset_name] = factory.create_from_preset(preset_name)

    return agents, factory
