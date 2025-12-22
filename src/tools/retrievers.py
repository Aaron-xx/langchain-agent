from typing import List
from langchain.tools import tool, ToolRuntime
from langchain_classic.retrievers import EnsembleRetriever

from src.common import RuntimeContext

@tool
async def similarity_search(query: str, runtime: ToolRuntime[RuntimeContext], k: int = 4) -> str:
    """Search documents by similarity using vector store."""
    doc_manager = runtime.context.doc_manager
    retriever = doc_manager.vector_store.as_retriever(k=k)
    docs = await retriever.ainvoke(query)
    return "\n".join([f"- {doc.page_content}" for doc in docs])

@tool
async def bm25_search(query: str, runtime: ToolRuntime[RuntimeContext], k: int = 4) -> str:
    """Search documents using BM25 keyword matching."""
    doc_manager = runtime.context.doc_manager
    bm25_retriever = doc_manager.get_retriever("bm25")
    docs = await bm25_retriever.ainvoke(query)
    return "\n".join([f"- {doc.page_content}" for doc in docs])

@tool
async def ensemble_search(query: str, runtime: ToolRuntime[RuntimeContext], k: int = 4) -> str:
    """Search documents using combined vector and keyword strategies."""
    doc_manager = runtime.context.doc_manager
    vector_retriever = doc_manager.get_retriever("basic")
    bm25_retriever = doc_manager.get_retriever("bm25")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    docs = await ensemble_retriever.ainvoke(query)
    return "\n".join([f"- {doc.page_content}" for doc in docs])

def get_all_retrievers() -> List:
    """Get all retrieval tools."""
    return [similarity_search, bm25_search, ensemble_search]