"""Shared type definitions for the RAG system."""

from typing import TYPE_CHECKING, Any

from typing_extensions import TypedDict


if TYPE_CHECKING:
    from src.config.config import Config
    from src.services.document_monitor import DocumentMonitorService
    from src.tools.documents import DocumentManager


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
        doc_monitor: Optional document monitor service
    """

    config: "Config"
    doc_manager: "DocumentManager"
    doc_monitor: "DocumentMonitorService"
