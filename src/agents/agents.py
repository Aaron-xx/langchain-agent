import asyncio
from typing import Dict, Tuple, Any
from config import get_config
from tools import get_all_retrievers, get_mcp_tools
from .factory import AgentFactory
from .config import PRESETS
from langgraph.store.memory import InMemoryStore

async def create_pre_agents() -> Tuple[Dict[str, Any], AgentFactory]:
    """Create all preset agents with their factory."""
    config = get_config()
    llm = config.chat_model

    retrieval_tools = get_all_retrievers()
    try:
        mcp_tools = await get_mcp_tools()
    except:
        mcp_tools = []

    tools_dict = {
        "retrieval": retrieval_tools,
        "mcp": mcp_tools,
    }
    store = InMemoryStore()

    factory = AgentFactory(llm, tools_dict, store)

    agents = {}
    for preset_name in PRESETS.keys():
        agents[preset_name] = factory.create_from_preset(preset_name)

    return agents, factory