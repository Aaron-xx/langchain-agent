"""RAG graph implementation with memory support"""
from re import DEBUG
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from src.common import RuntimeContext
from src.graphs.states import RAGState
from src.graphs.utils import retrieve_node, generate_node, extract_uploaded_files
import logging

from src.graphs.utils import route_node

logger = logging.getLogger(__name__)

class RAGGraph:
    """RAG graph with memory-enabled conversation support"""

    def __init__(self, runtime: Runtime[RuntimeContext]) -> None:
        """Initialize RAG graph with runtime context."""

        if runtime.context is None:
            raise ValueError("context not injected")

        self.doc_manager = runtime.context.doc_manager
        self.memory = MemorySaver()
        self.graph = self.build_graph(runtime)

        logger.info("RAGGraph initialized")

    def build_graph(self, runtime: Runtime[RuntimeContext]) -> StateGraph:
        """Build RAG graph with file extraction, retrieval, and generation nodes."""
        builder = StateGraph(RAGState, context_schema=RuntimeContext)

        builder.add_node("extract_files", extract_uploaded_files)
        builder.add_node("retrieve", retrieve_node)
        builder.add_node("generate", generate_node)

        builder.add_conditional_edges(
            START,
            route_node,
            {"extract_files": "extract_files", "retrieve": "retrieve"}
        )

        builder.add_edge("extract_files", "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)

        return builder.compile(checkpointer=self.memory)

    async def astream(self, query: str, runtime: Runtime[RuntimeContext], thread_id: str = "default"):
        """Stream RAG query execution with context preservation."""
        try:
            config = {
                "configurable": {
                    "thread_id": thread_id
                }
            }

            async for chunk in self.graph.astream(
                    query,
                    config=config,
                    context=runtime.context,
                    stream_mode="messages",
                ):
                token, metadata = chunk
                if hasattr(token, 'content') and token.content:
                    yield token.content

        except Exception as e:
            logger.error(f"RAG stream execution error: {e}")
            yield {
                "query": query,
                "answer": f"Execution error: {str(e)}",
                "sources": [],
                "error": str(e)
            }