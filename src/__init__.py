"""LangChain RAG System - Source Package.

This is the main entry point for the LangChain RAG system.
Provides access to all core components through a unified interface.
"""

import warnings

# Suppress Pydantic serialization warnings for complex objects in context.
# This is a known issue with LangGraph's context serialization when containing
# non-serializable objects like Config and DocumentManager instances.
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Core components
from src.agents import create_pre_agents
from src.common import (
    DocumentChunk,
    RuntimeContext,
)
from src.config import Config, get_config
from src.graphs import RAGGraph
from src.tools import DocumentManager, get_all_retrievers

# CLI module is imported on-demand to avoid early initialization
# from src.cli import cli_main

__all__ = [
    # Core components
    "RuntimeContext",
    "DocumentChunk",
    "RAGState",
    # Configuration
    "Config",
    "get_config",
    # Tools
    "DocumentManager",
    "get_all_retrievers",
    # Graphs
    "RAGGraph",
    # Agents
    "create_pre_agents",
    # CLI
    "cli_main",
]

# Version information
__version__ = "1.0.0"
__author__ = "LangChain RAG Team"
