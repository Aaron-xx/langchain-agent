"""Configuration system for multi-provider LLM models.

This module provides a unified configuration system that supports:
- Multiple LLM providers (OpenAI, Ollama, Anthropic, HuggingFace)
- Environment variable substitution with defaults
- Flexible configuration access (attribute, dict, fuzzy matching)
- Dynamic model loading with caching

NOTE: LangChain imports are deferred to chat() and embedding() methods
to enable instant CLI startup. Do not add LangChain imports at module level.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Import paths module for multi-tier configuration
try:
    from . import paths
except ImportError:
    # Fallback if paths module not available
    paths = None

# Load environment variables: prioritize global, fallback to current directory
if paths is not None:
    _global_env = paths.get_global_config_dir() / ".env"
    _current_env = Path(".env")

    if _global_env.exists():
        load_dotenv(_global_env, override=True)
    elif _current_env.exists():
        load_dotenv(_current_env)
else:
    # Fallback to default behavior if paths not available
    load_dotenv()

# Configure logging
def get_log_level() -> int:
    """Get log level from environment variable.

    Returns:
        Logging level constant (default: WARNING)
    """
    level = os.getenv("LOG_LEVEL", "WARNING").upper()
    return getattr(logging, level, logging.WARNING)


logging.basicConfig(
    level=get_log_level(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Config:
    """Unified configuration system for LLM applications.

    Supports multiple access patterns:
    - Attribute: config.persist_directory
    - Dict: config['vector_store.persist_directory']
    - Fuzzy: config.persistdirectory (underscores removed)
    """

    def __init__(self, config_path: str | None = None) -> None:
        """Initialize configuration system with multi-tiered support.

        Args:
            config_path: Optional explicit path to config file.
                        If None, follows hierarchy: project -> global

        Raises:
            FileNotFoundError: If no config file exists
        """
        # Determine config path
        if config_path is None:
            if paths is not None:
                try:
                    config_path = str(paths.find_config_path())
                except FileNotFoundError:
                    # Create default global config if none exists
                    paths.ensure_global_config()
                    config_path = str(paths.get_global_config_path())
            else:
                config_path = "config.json"

        logger.info(f"Loading configuration: {config_path}")

        # Resolve path expansion
        config_path = Path(config_path).expanduser()

        # Load JSON config
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self._raw: dict[str, Any] = json.load(f)

        logger.info("JSON configuration loaded")

        # Merge with project config if using global config and paths available
        if paths is not None and config_path == paths.get_global_config_path():
            project_config = paths.get_project_config_path()
            if project_config.exists():
                with open(project_config, "r", encoding="utf-8") as f:
                    project_raw = json.load(f)
                    # Deep merge project overrides
                    self._raw = self._deep_merge(self._raw, project_raw)
                logger.info("Merged project configuration overrides")

        # Store config path for reference
        self._config_path = config_path

        # Replace environment variables
        self._raw = self._replace_env(self._raw)
        logger.info("Environment variables replaced")

        # Expand user paths in configuration
        self._raw = self._expand_paths(self._raw)
        logger.info("Paths expanded")

        # Build flattened index
        self._index: dict[str, Any] = {}
        self._flatten(self._raw, "")
        logger.info(f"Index built with {len(self._index)} keys")

        # Model cache
        self._model_cache: dict[str, Any] = {}

    def _flatten(self, obj: Any, prefix: str) -> None:
        """Recursively flatten configuration for multiple access patterns.

        Supports three access methods:
        1. Shorthand: config.persist_directory
        2. Path: config['vector_store.persist_directory']
        3. Fuzzy: config.persistdirectory (underscores removed)

        Args:
            obj: Configuration object (dict, list, or primitive)
            prefix: Current path prefix
        """
        if isinstance(obj, dict):
            for key, value in obj.items():
                full_path = f"{prefix}.{key}" if prefix else key

                # Add to index with multiple access patterns (even for nested dicts)
                self._index[key] = value
                self._index[full_path] = value
                self._index[key.replace("_", "").lower()] = value  # Fuzzy match

                # Continue recursion for nested dicts
                if isinstance(value, dict):
                    self._flatten(value, full_path)

    def _replace_env(self, obj: Any) -> Any:
        """Recursively replace environment variables in configuration.

        Supports two formats:
        - ${VAR}: Get env var, empty if not found
        - ${VAR:-default}: Get env var, use default if not found

        Args:
            obj: Object to process

        Returns:
            Object with environment variables replaced
        """
        if isinstance(obj, str):
            def replacer(match: re.Match[str]) -> str:
                expr = match.group(1)
                if ":-" in expr:
                    var, default = expr.split(":-", 1)
                    return os.getenv(var.strip(), default.strip())
                return os.getenv(expr.strip(), "")

            return re.sub(r"\$\{([^}]+)\}", replacer, obj)

        if isinstance(obj, dict):
            return {k: self._replace_env(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [self._replace_env(v) for v in obj]

        return obj

    # ========== Access Methods ==========

    def __getattr__(self, key: str) -> Any:
        """Attribute access: config.persist_directory.

        NOTE: This is a Python magic method, automatically called by Python
        when accessing attributes. DO NOT call this method directly.

        Usage:
            config.log_level

        Args:
            key: Configuration key

        Returns:
            Configuration value

        Raises:
            AttributeError: If key not found
        """
        # Avoid interfering with private attributes
        if key.startswith("_"):
            return super().__getattribute__(key)

        if key in self._index:
            logger.debug(f"Attribute access: {key}")
            return self._index[key]

        raise AttributeError(f"Configuration not found: {key}")

    def __getitem__(self, key: str) -> Any:
        """Dictionary access: config['vector_store.persist_directory'].

        NOTE: This is a Python magic method, automatically called by Python
        when using square bracket notation. DO NOT call this method directly.

        Usage:
            config['log_level']

        Args:
            key: Configuration key

        Returns:
            Configuration value

        Raises:
            KeyError: If key not found
        """
        if key in self._index:
            logger.debug(f"Dictionary access: {key}")
            return self._index[key]

        raise KeyError(f"Configuration not found: {key}")

    def get(self, key: str, default: Any = None) -> Any:
        """Function access with default: config.get('key', default='value').

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        result = self._index.get(key, default)
        logger.debug(f"Function access: {key} → {result}")
        return result

    def get_group(self, prefix: str) -> dict[str, Any]:
        """Get configuration group: config.get_group('vector_store').

        Returns all keys starting with prefix, with prefix removed.

        Args:
            prefix: Configuration prefix

        Returns:
            Dictionary of matching configurations
        """
        prefix_with_dot = f"{prefix}."
        result = {
            k: v for k, v in self._index.items()
            if k.startswith(prefix_with_dot)
        }

        # Remove prefix, keep only relative path
        result = {
            k[len(prefix_with_dot):]: v
            for k, v in result.items()
        }

        logger.debug(f"Get config group: {prefix} → {len(result)} items")
        return result

    def list_keys(self, pattern: str | None = None) -> list[str]:
        """List all configuration keys.

        Args:
            pattern: Optional filter pattern

        Returns:
            Sorted list of keys
        """
        keys = sorted(self._index.keys())

        if pattern:
            keys = [k for k in keys if pattern.lower() in k.lower()]

        logger.debug(f"List {len(keys)} keys")
        return keys

    def search(self, pattern: str) -> dict[str, Any]:
        """Search configuration by pattern.

        Args:
            pattern: Search pattern (case-insensitive)

        Returns:
            Dictionary of matching configurations
        """
        pattern = pattern.lower()
        result = {
            k: v for k, v in self._index.items()
            if pattern in k.lower()
        }

        logger.info(f"Search '{pattern}' found {len(result)} results")
        return result

    # ========== Model Loading ==========

    def _get_default_provider(self, model_type: str = "chat") -> str:
        """Get default provider based on environment_type.

        Args:
            model_type: Model type ('chat' or 'embedding')

        Returns:
            Provider name
        """
        env_type = self._raw.get("environment_type", "external")
        return "ollama" if env_type == "internal" else "openai"

    def chat(self, provider: str = "openai") -> Any:
        """Get chat model instance with caching.

        Args:
            provider: Provider name (currently ignored, uses config default)

        Returns:
            Chat model instance
        """
        provider = self._get_default_provider("chat")
        cache_key = f"{provider}_chat"

        # Check cache
        if cache_key in self._model_cache:
            logger.info(f"Model cache hit: {cache_key}")
            return self._model_cache[cache_key]

        logger.info(f"Loading {provider} chat model...")

        # Lazy import LangChain to enable instant CLI startup
        from langchain.chat_models import init_chat_model

        # Get configuration parameters
        config_key = f"models.chat.{provider}"
        params = self.get_group(config_key).copy()
        model_name = params.pop("model")
        params.pop("provider", None)

        # Create model using LangChain's init_chat_model
        model = init_chat_model(
            model=model_name,
            model_provider=provider,
            **params
        )

        # Cache
        self._model_cache[cache_key] = model
        logger.info(f"{provider} chat model loaded successfully")
        return model

    def embedding(self, provider: str = "openai") -> Any:
        """Get embedding model instance with caching.

        Args:
            provider: Provider name (currently ignored, uses ollama)

        Returns:
            Embedding model instance
        """
        provider = "ollama"
        cache_key = f"{provider}_embedding"

        # Check cache
        if cache_key in self._model_cache:
            logger.info(f"Model cache hit: {cache_key}")
            return self._model_cache[cache_key]

        logger.info(f"Loading {provider} embedding model...")

        # Lazy import LangChain to enable instant CLI startup
        from langchain.embeddings import init_embeddings

        # Get configuration parameters
        config_key = f"models.embedding.{provider}"
        params = self.get_group(config_key).copy()
        model_name = params.pop("model")
        params.pop("provider", None)

        # Create model using LangChain's init_embeddings
        model = init_embeddings(
            model=model_name,
            provider=provider,
            **params
        )

        # Cache
        self._model_cache[cache_key] = model
        logger.info(f"{provider} embedding model loaded successfully")
        return model

    # ========== Utility Methods ==========

    def get_raw(self) -> dict[str, Any]:
        """Get raw configuration dictionary.

        Returns:
            Copy of raw configuration
        """
        return self._raw.copy()

    def get_stats(self) -> dict[str, Any]:
        """Get system statistics.

        Returns:
            Dictionary with stats (total_keys, cached_models, providers, all_keys)
        """
        return {
            "total_keys": len(self._index),
            "cached_models": len(self._model_cache),
            "providers": list(self.PROVIDERS.keys()),
            "all_keys": self.list_keys(),
        }

    def print_config(self) -> None:
        """Print complete configuration (for debugging).

        Masks sensitive information (keys, tokens, passwords, secrets).
        """
        print("\n" + "=" * 60)
        print("Complete Configuration")
        print("=" * 60)

        # Display grouped by first letter
        current_group = None
        for key in self.list_keys():
            group = key.split(".")[0]
            if group != current_group:
                if current_group is not None:
                    print()
                print(f"\n[{group}]")
                current_group = group

            value = self._index[key]
            # Mask sensitive information
            if any(s in key.lower() for s in ["key", "token", "password", "secret"]):
                value = "***" if value else value
            print(f"  {key}: {value}")

        print("\n" + "=" * 60 + "\n")

    # ========== Multi-tier Configuration Support ==========

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Deep merge two dictionaries.

        Args:
            base: Base dictionary
            override: Override dictionary (takes precedence)

        Returns:
            Merged dictionary
        """
        result = base.copy()
        for key, value in override.items():
            # Skip comment keys (starting with #)
            if key.startswith("#"):
                continue
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _expand_paths(self, obj: Any) -> Any:
        """Expand ~ in path strings.

        Args:
            obj: Object to process

        Returns:
            Object with paths expanded
        """
        if isinstance(obj, str):
            return os.path.expanduser(obj)

        if isinstance(obj, dict):
            return {k: self._expand_paths(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [self._expand_paths(v) for v in obj]

        return obj


def get_config(config_path: str | None = None) -> Config:
    """Create configuration instance.

    Args:
        config_path: Path to configuration file. If None, uses auto-discovery
                    via paths module (project -> global -> create default).

    Returns:
        Config instance

    Example:
        >>> # Auto-discover config file
        >>> config = get_config()
        >>> # Or specify explicit path
        >>> config = get_config('config.json')
    """
    return Config(config_path)