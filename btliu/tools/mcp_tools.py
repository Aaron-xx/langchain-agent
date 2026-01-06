import logging
from langchain_mcp_adapters.client import MultiServerMCPClient
from btliu.config import get_config

logger = logging.getLogger(__name__)


async def get_mcp_connection_status() -> dict:
    """获取 MCP 服务器连接状态（逐个测试）.

    Returns:
        dict: {
            "server_name": {
                "connected": bool,
                "tools_count": int
            }
        }
    """
    config = get_config()
    mcp_config = config.get("mcp_servers", {})

    if not mcp_config:
        return {}

    results = {}

    for server_name, server_config in mcp_config.items():
        try:
            client = MultiServerMCPClient({server_name: server_config})
            tools = await client.get_tools()
            results[server_name] = {"connected": True, "tools_count": len(tools)}
        except Exception:
            results[server_name] = {"connected": False, "tools_count": 0}

    return results


async def get_mcp_tools(verbose: bool = True):
    """Get all MCP tools from configured servers.

    Args:
        verbose: If True, print connection status messages.

    Returns:
        List of available MCP tools, or empty list if connection fails.
    """
    config = get_config()
    mcp_config = config.get("mcp_servers")

    try:
        mcp_client = MultiServerMCPClient(mcp_config)
        tools = await mcp_client.get_tools()
        if verbose and tools:
            logger.info(
                f"✓ MCP connected: {len(tools)} tools", extra={"color": "success"}
            )
        return tools
    except Exception:
        if verbose:
            logger.warning("! MCP connection failed", extra={"color": "warning"})
        return []
