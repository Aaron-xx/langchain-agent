"""Retrieval tools for RAG applications.

This module provides LangChain tool wrappers for document retrieval:
- similarity_search: Vector similarity search with relevance scores
- bm25_search: BM25 keyword-based search

The tools integrate with DocumentManager and are designed to work with
LangGraph's ToolRuntime mechanism for agent-based RAG systems.

"""

from typing import Any

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime


@tool
async def similarity_search(
    query: str,
    k: int | None = None,
    tool_runtime: ToolRuntime = None,
) -> str:
    """Search documents by vector similarity with relevance scores.

    Performs semantic search using Qdrant vector store and returns
    results with similarity scores (0-1 range, higher = more relevant).

    Args:
        query: Search query string (natural language question)
        k: Number of results to return (default: use max_retrieved_docs from config)
        tool_runtime: Tool runtime context containing doc_manager

    Returns:
        Formatted search results, one per line:
        "[0.850] Document content here..."
        "[0.723] Another relevant document..."

    Raises:
        ValueError: If tool_runtime is not provided

    Note:
        - Uses cosine similarity (distance-based converted to relevance)
        - Returns 0.0-1.0 range where 1.0 = perfect match
        - Configured similarity_threshold in config.json filters results
    """
    if tool_runtime is None:
        raise ValueError("tool_runtime is required")

    doc_manager = tool_runtime.context["doc_manager"]

    # Use configured k if not provided
    if k is None:
        k = doc_manager.max_retrieved_docs

    # Use similarity_search_with_relevance_scores to get scores
    # This returns (Document, relevance_score) tuples
    vector_store = doc_manager.vector_store
    results = await vector_store.asimilarity_search_with_relevance_scores(query, k=k)

    # Format results with scores: "[score] content"
    formatted_docs = []
    for doc, score in results:
        formatted_docs.append(f"[{score:.3f}] {doc.page_content}")

    return "\n".join(formatted_docs)


@tool
async def bm25_search(
    query: str,
    k: int | None = None,
    tool_runtime: ToolRuntime = None,
) -> str:
    """Search documents using BM25 keyword matching.

    Performs keyword-based search using BM25 algorithm, which ranks
    documents based on term frequency and inverse document frequency.

    Args:
        query: Search query string (keywords work best)
        k: Number of results to return (default: use max_retrieved_docs from config)
        tool_runtime: Tool runtime context containing doc_manager

    Returns:
        Formatted search results, one per line:
        "- Document content here..."
        "- Another document..."

    Raises:
        ValueError: If tool_runtime is not provided

    Note:
        - BM25 does not return relevance scores (keyword matching only)
        - Useful for exact term matching when vector search misses
        - Loads all documents from data directory on first use
    """
    if tool_runtime is None:
        raise ValueError("tool_runtime is required")

    doc_manager = tool_runtime.context["doc_manager"]
    bm25_retriever = doc_manager.get_retriever("bm25", k=k)
    docs = await bm25_retriever.ainvoke(query)
    return "\n".join([f"- {doc.page_content}" for doc in docs])


def get_all_retrievers() -> list[Any]:
    """Get all retrieval tools as a list.

    Returns:
        List of LangChain tool functions: [similarity_search, bm25_search]

    Example:
        tools = get_all_retrievers()
        agent = create_agent(model=llm, tools=tools, ...)
    """
    return [similarity_search, bm25_search]
