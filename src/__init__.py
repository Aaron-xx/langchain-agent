"""
LangChain RAG System - Source Package

This is the main entry point for the LangChain RAG system.
Provides access to all core components through a unified interface.
"""

# Suppress Pydantic serialization warnings for complex objects in context
# This is a known issue with LangGraph's context serialization when containing
# non-serializable objects like Config and DocumentManager instances.
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Core components
from src.common import (
    RuntimeContext,
    DocumentChunk,
)

from src.config import Config, get_config

from src.tools import DocumentManager, get_all_retrievers

from src.agents import create_pre_agents

# CLI module is imported on-demand to avoid early initialization
# from src.cli import cli_main

__all__ = [
    # Core components
    "RuntimeContext",
    "DocumentChunk",

    # Configuration
    "Config",
    "get_config",

    # Tools
    "DocumentManager",
    "get_all_retrievers",

    # Agents
    "create_pre_agents",

    # CLI
    "cli_main",
]

# Version information
__version__ = "1.0.0"