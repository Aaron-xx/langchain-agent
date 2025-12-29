"""Shared type definitions for the RAG system."""

from typing import TYPE_CHECKING, Any

from typing_extensions import TypedDict


if TYPE_CHECKING:
    from btliu.config.config import Config
    from btliu.tools.documents import DocumentManager


class DocumentChunk(TypedDict):
    """Document chunk with metadata.

    Attributes:
        content: Chunk text content
        metadata: Associated metadata
        source: Source document path
        page_number: Optional page number
        chunk_id: Unique chunk identifier
    """

    content: str
    metadata: dict[str, Any]
    source: str
    page_number: int | None
    chunk_id: str


class RuntimeContext(TypedDict, total=False):
    """Runtime context containing configuration and services.

    Attributes:
        config: Configuration instance
        doc_manager: Document manager instance
        checkpointer: LangGraph checkpointer for session persistence
        store: LangGraph store for cross-thread memory
        thread_id: Current session thread ID
        user_id: User identifier for cross-thread memory isolation
    """

    config: "Config"
    doc_manager: "DocumentManager"
    checkpointer: Any
    store: Any
    thread_id: str
    user_id: str
