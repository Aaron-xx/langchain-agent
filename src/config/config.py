"""Configuration system for multi-provider LLM models.

This module provides a unified configuration system that supports:
- Multiple LLM providers (OpenAI, Ollama, Anthropic, HuggingFace)
- Environment variable substitution with defaults
- Flexible configuration access (attribute, dict, fuzzy matching)
- Dynamic model loading with caching
"""

import importlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

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

    # Supported LLM providers
    PROVIDERS: dict[str, dict[str, str]] = {
        "openai": {
            "chat": "langchain_openai.ChatOpenAI",
            "embedding": "langchain_openai.OpenAIEmbeddings",
        },
        "ollama": {
            "chat": "langchain_ollama.ChatOllama",
            "embedding": "langchain_ollama.OllamaEmbeddings",
        },
        "anthropic": {
            "chat": "langchain_anthropic.ChatAnthropic",
            "embedding": "langchain_community.embeddings.HuggingFaceEmbeddings",
        },
        "huggingface": {
            "chat": "langchain_huggingface.ChatHuggingFace",
            "embedding": "langchain_huggingface.HuggingFaceEmbeddings",
        },
    }

    def __init__(self, config_path: str = "config.json") -> None:
        """Initialize configuration system.

        Args:
            config_path: Path to JSON configuration file

        Raises:
            FileNotFoundError: If config file does not exist
        """
        logger.info(f"Loading configuration: {config_path}")

        # Load JSON config
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self._raw: dict[str, Any] = json.load(f)

        logger.info("JSON configuration loaded")

        # Replace environment variables
        self._raw = self._replace_env(self._raw)
        logger.info("Environment variables replaced")

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

                if isinstance(value, dict):
                    # Continue recursion
                    self._flatten(value, full_path)
                else:
                    # Add to index with multiple access patterns
                    self._index[key] = value
                    self._index[full_path] = value
                    self._index[key.replace("_", "").lower()] = value  # Fuzzy match

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
        """Get chat model instance.

        Args:
            provider: Provider name (openai|ollama|anthropic|huggingface)

        Returns:
            Chat model instance
        """
        provider = self._get_default_provider("chat")
        return self._get_model(provider, "chat")

    def embedding(self, provider: str = "openai") -> Any:
        """Get embedding model instance.

        Args:
            provider: Provider name

        Returns:
            Embedding model instance
        """
        return self._get_model("ollama", "embedding")

    def _get_model(self, provider: str, model_type: str) -> Any:
        """Internal method: Create and cache model instance.

        Args:
            provider: Provider name
            model_type: Model type ('chat' or 'embedding')

        Returns:
            Model instance

        Raises:
            ValueError: If provider or model_type is not supported
            ImportError: If provider module cannot be imported
        """
        # Check cache
        cache_key = f"{provider}_{model_type}"
        if cache_key in self._model_cache:
            logger.info(f"Model cache hit: {cache_key}")
            return self._model_cache[cache_key]

        # Validate provider
        if provider not in self.PROVIDERS:
            available = ", ".join(self.PROVIDERS.keys())
            raise ValueError(f"Unknown provider: '{provider}'. Supported: {available}")

        if model_type not in self.PROVIDERS[provider]:
            available = ", ".join(self.PROVIDERS[provider].keys())
            raise ValueError(
                f"Provider '{provider}' does not support '{model_type}'. Supported: {available}"
            )

        try:
            logger.info(f"Loading {provider} {model_type} model...")

            # Get class path
            class_path = self.PROVIDERS[provider][model_type]
            module_path, class_name = class_path.rsplit(".", 1)

            # Dynamic import
            try:
                module = importlib.import_module(module_path)
            except ImportError as e:
                raise ImportError(
                    f"Cannot import {module_path}. "
                    f"Install: pip install langchain-{provider}"
                ) from e

            cls = getattr(module, class_name)

            # Get configuration parameters
            config_key = f"models.{model_type}.{provider}"
            params = self.get_group(config_key)

            # Clean parameters (remove unnecessary fields)
            params = {k: v for k, v in params.items() if k != "provider"}

            logger.info(f"Model parameters: {params}")

            # Create instance
            instance = cls(**params)

            # Cache
            self._model_cache[cache_key] = instance

            logger.info(f"{provider} {model_type} model loaded successfully")
            return instance

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

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


def get_config(config_path: str = "config.json") -> Config:
    """Create configuration instance.

    Args:
        config_path: Path to configuration file

    Returns:
        Config instance

    Example:
        >>> config = get_config('config.json')
        >>> value = config.persist_directory
    """
    return Config(config_path)