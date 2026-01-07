"""Shared type definitions for the RAG system."""

from typing import TYPE_CHECKING

from typing_extensions import TypedDict


if TYPE_CHECKING:
    from btliu.config.config import Config
    from btliu.tools.documents import DocumentManager


class RuntimeContext(TypedDict, total=False):
    """Runtime context containing configuration and services.

    Attributes:
        config: Configuration instance
        doc_manager: Document manager instance for document retrieval
        user_id: User identifier for memory isolation
    """

    config: "Config"
    doc_manager: "DocumentManager"
    user_id: str
