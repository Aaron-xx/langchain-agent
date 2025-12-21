from langchain_mcp_adapters.client import MultiServerMCPClient
from config import get_config

async def get_mcp_tools():
    """获取所有MCP工具的核心函数"""
    # 从配置读取MCP服务器配置
    config = get_config()
    mcp_config = config.mcp_servers
    # 创建客户端并获取工具
    mcp_client = MultiServerMCPClient(mcp_config)
    tools = await mcp_client.get_tools()
    return tools