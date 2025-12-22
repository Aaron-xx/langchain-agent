from dataclasses import dataclass
import operator
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Any, List, Optional

from src.common.types import DocumentChunk

class RAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # Complete conversation history
    query: str                                             # Original user query
    context: str                                           # Retrieved content
    answer: str                                            # Final answer
    sources: Annotated[list["DocumentChunk"], operator.add]  # Track source documents
    error: Optional[str]                                   # Error handling and logging
