"""Retrieval tools using LangChain/LangGraph official ToolRuntime mechanism."""

from typing import Any

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime


@tool
async def similarity_search(
    query: str,
    k: int = 4,
    tool_runtime: ToolRuntime = None,
) -> str:
    """Search documents by similarity using vector store.

    Args:
        query: Search query string
        k: Number of results to return
        tool_runtime: Tool runtime context

    Returns:
        Formatted search results

    Raises:
        ValueError: If tool_runtime is not provided
    """
    if tool_runtime is None:
        raise ValueError("tool_runtime is required")

    doc_manager = tool_runtime.context["doc_manager"]
    retriever = doc_manager.get_retriever("basic", k=k)
    docs = await retriever.ainvoke(query)
    return "\n".join([f"- {doc.page_content}" for doc in docs])


@tool
async def bm25_search(
    query: str,
    k: int = 4,
    tool_runtime: ToolRuntime = None,
) -> str:
    """Search documents using BM25 keyword matching.

    Args:
        query: Search query string
        k: Number of results to return
        tool_runtime: Tool runtime context

    Returns:
        Formatted search results

    Raises:
        ValueError: If tool_runtime is not provided
    """
    if tool_runtime is None:
        raise ValueError("tool_runtime is required")

    doc_manager = tool_runtime.context["doc_manager"]
    bm25_retriever = doc_manager.get_retriever("bm25", k=k)
    docs = await bm25_retriever.ainvoke(query)
    return "\n".join([f"- {doc.page_content}" for doc in docs])


def get_all_retrievers() -> list[Any]:
    """Get all retrieval tools.

    Returns:
        List of retrieval tool functions
    """
    return [similarity_search, bm25_search]
