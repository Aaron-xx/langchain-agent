"""
LangChain RAG System - Source Package

This is the main entry point for the LangChain RAG system.
Provides access to all core components through a unified interface.
"""

# Core components
from common import (
    RuntimeContext,
    DocumentChunk,
)

from config import Config, get_config

from tools import DocumentManager, get_all_retrievers

from graphs import RAGGraph

from agents import create_pre_agents

from cli import cli_main

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