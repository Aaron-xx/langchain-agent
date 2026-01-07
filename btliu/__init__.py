"""
LangChain RAG System - Source Package

This is the main entry point for the LangChain RAG system.
Provides access to all core components through a unified interface.

NOTE: Heavy imports (DocumentManager, agents) are lazy-loaded to enable
fast CLI startup. They are imported on-demand when first used.
"""

# Suppress Pydantic serialization warnings for complex objects in context
# This is a known issue with LangGraph's context serialization when containing
# non-serializable objects like Config and DocumentManager instances.
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Core components - lightweight imports
from btliu.common import (  # noqa: E402
    RuntimeContext,
)

# Configuration - lightweight import
from btliu.config import Config, get_config  # noqa: E402

# Heavy imports removed for fast startup:
# - btliu.tools (DocumentManager, get_all_retrievers) -> triggers Qdrant, ML libs
# - btliu.agents (create_pre_agents) -> may trigger heavy agent imports
# These are imported on-demand inside CLI and other modules

# CLI module is imported on-demand to avoid early initialization
# from btliu.cli import cli_main

__all__ = [
    # Core components
    "RuntimeContext",
    # Configuration
    "Config",
    "get_config",
    # Tools (lazy - imported on demand)
    "DocumentManager",
    "get_all_retrievers",
    # Agents (lazy - imported on demand)
    "create_pre_agents",
    # CLI (lazy - imported on demand)
    "cli_main",
]


def __getattr__(name: str):
    """Lazy import heavy modules on first access.

    This enables fast CLI startup by only importing heavy dependencies
    (Qdrant, ML libraries, agents) when actually needed.

    Args:
        name: Attribute being accessed

    Returns:
        The requested attribute

    Raises:
        AttributeError: If name is not a lazy export
    """
    if name == "DocumentManager":
        from btliu.tools.documents import DocumentManager

        return DocumentManager
    elif name == "get_all_retrievers":
        from btliu.tools import get_all_retrievers

        return get_all_retrievers
    elif name == "create_pre_agents":
        from btliu.agents import create_pre_agents

        return create_pre_agents
    elif name == "cli_main":
        from btliu.cli import cli_main

        return cli_main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Version information
__version__ = "1.0.0"
