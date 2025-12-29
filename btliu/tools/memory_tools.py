"""Cross-thread memory tools using LangGraph's ToolRuntime injection.

IMPORTANT: This module uses tool_runtime.store (injected by LangGraph) instead of
context.get("store") to avoid including unpicklable AsyncPostgresStore in the
agent context. See cli.py WORKAROUND for full context.

The Store is injected by LangGraph via ToolRuntime when tools are called,
not stored in the RuntimeContext passed to agent creation.
"""

import time
from typing import Any

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime


@tool
async def save_memory(
    info: str,
    category: str = "general",
    tool_runtime: ToolRuntime = None,
) -> str:
    """保存用户信息到跨线程记忆.

    此工具将信息持久化到 PostgreSQL Store，可在不同会话间共享。
    记忆按用户隔离（基于 context 中的 user_id）。

    Args:
        info: 要记忆的信息内容
        category: 记忆类别，用于组织不同类型的记忆（默认: general）
                 常用类别: profile, projects, preferences, tasks

    Returns:
        确认消息，包含记忆 ID 和类别

    Example:
        save_memory("用户名叫 Alice，是一名 Python 工程师", "profile")
        save_memory("正在开发 ML 推荐系统项目", "projects")
    """
    if tool_runtime is None:
        return "错误: tool_runtime 不可用"

    context = tool_runtime.context
    if context is None:
        return "错误: context 不可用"

    user_id = context.get("user_id", "anonymous")

    # Get store directly from tool_runtime (injected by LangGraph)
    store = tool_runtime.store

    if store is None:
        return "错误: Store 未配置，无法保存记忆"

    # 创建命名空间: (user_id, category)
    namespace = (user_id, category)

    # 生成记忆 ID（使用时间戳）
    memory_id = f"mem_{int(time.time())}"

    # 存储到 PostgresStore
    await store.aput(
        namespace,
        memory_id,
        {
            "info": info,
            "category": category,
            "timestamp": time.time(),
        },
    )

    return f"已保存记忆 [{category}]: {info}"


@tool
async def recall_memory(
    query: str | None = None,
    category: str = "general",
    limit: int = 10,
    tool_runtime: ToolRuntime = None,
) -> str:
    """从跨线程记忆中检索用户信息.

    检索之前保存的记忆，支持按类别过滤。

    Args:
        query: 可选的搜索关键词（暂未实现，预留参数）
        category: 要检索的记忆类别（默认: general）
        limit: 返回的最大记忆数量（默认: 10）

    Returns:
        格式化的记忆列表，或未找到的提示消息

    Example:
        recall_memory()  # 获取所有 general 类别记忆
        recall_memory(None, "projects", 5)  # 获取最近 5 条项目记忆
    """
    if tool_runtime is None:
        return "错误: tool_runtime 不可用"

    context = tool_runtime.context
    if context is None:
        return "错误: context 不可用"

    user_id = context.get("user_id", "anonymous")

    # Get store directly from tool_runtime (injected by LangGraph)
    store = tool_runtime.store

    if store is None:
        return "错误: Store 未配置，无法检索记忆"

    # 搜索命名空间前缀: (user_id, category)
    namespace_prefix = (user_id, category)

    try:
        items = await store.asearch(namespace_prefix, limit=limit)

        if not items:
            return f"未找到类别 '{category}' 的记忆"

        # 格式化结果
        results = []
        for item in items:
            info = item.value.get("info", "N/A")
            results.append(f"- {info}")

        return f"找到 {len(results)} 条记忆:\n" + "\n".join(results)

    except Exception as e:
        return f"检索记忆时出错: {str(e)}"


@tool
async def list_categories(
    tool_runtime: ToolRuntime = None,
) -> str:
    """列出当前用户的所有记忆类别.

    返回所有已存储记忆的类别及其计数。

    Returns:
        格式化的类别列表，每类显示记忆数量

    Example:
        list_categories()
        # 返回:
        # 记忆类别:
        # - profile: 2 条
        # - projects: 5 条
        # - preferences: 1 条
    """
    if tool_runtime is None:
        return "错误: tool_runtime 不可用"

    context = tool_runtime.context
    if context is None:
        return "错误: context 不可用"

    user_id = context.get("user_id", "anonymous")

    # Get store directly from tool_runtime (injected by LangGraph)
    store = tool_runtime.store

    if store is None:
        return "错误: Store 未配置，无法列出类别"

    try:
        # 搜索用户的所有命名空间
        items = await store.asearch((user_id,), limit=100)

        # 按类别统计
        categories = {}
        for item in items:
            # namespace 是 (user_id, category)
            if len(item.namespace) > 1:
                cat = item.namespace[1]
                categories[cat] = categories.get(cat, 0) + 1

        if not categories:
            return "未找到记忆类别"

        result = "记忆类别:\n"
        for cat, count in categories.items():
            result += f"- {cat}: {count} 条\n"

        return result.strip()

    except Exception as e:
        return f"列出类别时出错: {str(e)}"


def get_memory_tools() -> list[Any]:
    """获取所有记忆工具.

    Returns:
        记忆工具列表: [save_memory, recall_memory, list_categories]
    """
    return [save_memory, recall_memory, list_categories]
