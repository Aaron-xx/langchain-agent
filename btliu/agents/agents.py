"""Agent creation and management."""

import asyncio
from typing import Any

from btliu.agents.factory import AgentFactory
from btliu.agents.config import PRESETS
from btliu.common import RuntimeContext
from btliu.config import paths
from btliu.tools import get_all_retrievers, get_mcp_tools, get_memory_tools


async def create_pre_agents(
    context: RuntimeContext,
) -> tuple[dict[str, Any], AgentFactory]:
    """Create all preset agents with their factory.

    Args:
        context: Runtime context containing configuration

    Returns:
        Tuple of (agents dictionary, AgentFactory instance)
    """
    config = context.get("config")
    if config is None:
        raise ValueError("Config is required")
    llm = config.chat()

    retrieval_tools = get_all_retrievers()
    try:
        mcp_tools = await get_mcp_tools()
    except Exception:
        mcp_tools = []

    # Get memory tools for cross-thread memory
    memory_tools = get_memory_tools()

    tools_dict = {
        "retrieval": retrieval_tools,
        "mcp": mcp_tools,
        "memory": memory_tools,
    }

    # Get checkpointer and store from context
    store = context.get("store")
    checkpointer = context.get("checkpointer")

    # Use current working directory as agent filesystem root
    working_dir = str(paths.get_working_dir())
    factory = AgentFactory(
        llm,
        tools_dict,
        store=store,
        checkpointer=checkpointer,
        filesystem_root=working_dir
    )

    agents: dict[str, Any] = {}
    for preset_name in PRESETS:
        agents[preset_name] = factory.create_from_preset(preset_name)

    return agents, factory
