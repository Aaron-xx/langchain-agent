import logging
from langchain_mcp_adapters.client import MultiServerMCPClient
from btliu.config import get_config

logger = logging.getLogger(__name__)


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
            logger.warning(
                f"! MCP connection failed", extra={"color": "warning"}
            )
        return []
