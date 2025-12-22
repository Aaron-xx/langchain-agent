"""Shared type definitions - common types only"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Literal, TYPE_CHECKING
from typing_extensions import TypedDict
from enum import Enum

if TYPE_CHECKING:
    from src.config.config import Config
    from src.tools.documents import DocumentManager

class DocumentChunk(TypedDict):
    """Document chunk with metadata."""
    content: str
    metadata: Dict[str, Any]
    source: str
    page_number: Optional[int]
    chunk_id: str

@dataclass
class RuntimeContext:
    """Runtime context containing configuration and document manager."""
    config: 'Config'
    doc_manager: 'DocumentManager'
    