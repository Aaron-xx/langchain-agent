"""Path management for btliu configuration and data directories.

This module provides centralized path resolution following the hierarchy:
1. Project-level: ./.btliu/
2. Global-level: ~/.btliu/
3. Defaults: Embedded in package

Design principles:
- Global config: ~/.btliu/config.json (created on first run)
- Project config: ./.btliu/config.json (optional, overrides global)
- Global data: ~/.btliu/data/ (primary storage)
- Project data: ./.btliu/data/ (optional, per-project)
- Working directory: Always current working directory
"""

import json
from pathlib import Path

# Package directory (relative to this file)
PACKAGE_DIR = Path(__file__).parent.parent

# Template configuration
DEFAULT_CONFIG_TEMPLATE = PACKAGE_DIR / "defaults" / "config.json.template"


def get_global_config_dir() -> Path:
    """Get global configuration directory.

    Returns:
        Path to ~/.btliu/
    """
    return Path.home() / ".btliu"


def get_global_config_path() -> Path:
    """Get global configuration file path.

    Returns:
        Path to ~/.btliu/config.json
    """
    return get_global_config_dir() / "config.json"


def get_project_config_dir() -> Path:
    """Get project-level configuration directory.

    Returns:
        Path to ./.btliu/ (relative to current working directory)
    """
    return Path.cwd() / ".btliu"


def get_project_config_path() -> Path:
    """Get project-level configuration file path.

    Returns:
        Path to ./.btliu/config.json (may not exist)
    """
    return get_project_config_dir() / "config.json"


def get_global_data_dir() -> Path:
    """Get global data directory.

    Returns:
        Path to ~/.btliu/data/
    """
    return get_global_config_dir() / "data"


def get_project_data_dir() -> Path:
    """Get project-level data directory.

    Returns:
        Path to ./.btliu/data/ (may not exist)
    """
    return get_project_config_dir() / "data"


def get_working_dir() -> Path:
    """Get current working directory.

    This is the directory where the user executed 'btliu' command.

    Returns:
        Current working directory
    """
    return Path.cwd()


def find_config_path() -> Path:
    """Find the active configuration file following hierarchy.

    Hierarchy:
    1. Project config: ./.btliu/config.json (if exists)
    2. Global config: ~/.btliu/config.json (must exist after first run)

    Returns:
        Path to the active config file

    Raises:
        FileNotFoundError: If no config file exists
    """
    project_config = get_project_config_path()
    if project_config.exists():
        return project_config

    global_config = get_global_config_path()
    if global_config.exists():
        return global_config

    raise FileNotFoundError(
        "No configuration file found. Run 'btliu --init' to create default config."
    )


def get_data_dirs() -> list[Path]:
    """Get list of data directories in priority order.

    Returns:
        List of data directories [project, global] if both exist,
        or [global] if only global exists
    """
    dirs = []

    project_data = get_project_data_dir()
    if project_data.exists():
        dirs.append(project_data)

    global_data = get_global_data_dir()
    if global_data.exists():
        dirs.append(global_data)

    return dirs if dirs else [get_global_data_dir()]


def get_documents_dir() -> Path:
    """Get global documents directory.

    Returns:
        Path to global documents dir (~/.btliu/data/documents)
    """
    return get_global_data_dir() / "documents"

def get_process_hash_index() -> Path:
    """Get global documents process hash index file.

    Returns:
        Path to global documents dir (~/.btliu/data/.hash_index.json)
    """
    return get_global_data_dir() / ".hash_index.json"

def ensure_global_config() -> Path:
    """Ensure global configuration exists, creating from template if needed.

    Also creates the default data directory structure at ~/.btliu/data/documents/

    Returns:
        Path to global config file
    """
    global_config = get_global_config_path()
    global_dir = get_global_config_dir()

    if not global_config.exists():
        # Create global config directory
        global_dir.mkdir(parents=True, exist_ok=True)

        # Create data directory structure
        global_data_dir = get_global_data_dir()
        documents_dir = global_data_dir / "documents"
        documents_dir.mkdir(parents=True, exist_ok=True)

        # Copy from template
        if DEFAULT_CONFIG_TEMPLATE.exists():
            import shutil

            shutil.copy(DEFAULT_CONFIG_TEMPLATE, global_config)
        else:
            # Create minimal default config
            default_config = {
                "log_level": "WARNING",
                "environment_type": "external",
                "models": {
                    "chat": {
                        "openai": {
                            "provider": "openai",
                            "model": "gpt-3.5-turbo",
                            "api_key": "${OPENAI_API_KEY:-}",
                        }
                    },
                    "embedding": {
                        "ollama": {
                            "provider": "ollama",
                            "model": "bge-m3:latest",
                            "base_url": "http://localhost:11434",
                        }
                    },
                },
                "vector_store": {
                    "qdrant_url": "http://localhost:6333",
                    "collection_name": "btliu_documents",
                },
                "document_processing": {
                    "data_dir": "~/.btliu/data/documents",
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                    "hash_index_file": "~/.btliu/data/.hash_index.json",
                },
            }

            with open(global_config, "w") as f:
                json.dump(default_config, f, indent=2)

    return global_config


def init_project_config() -> Path:
    """Initialize project-level configuration.

    Creates ./.btliu/ directory with config.json if it doesn't exist.

    Returns:
        Path to project config file
    """
    project_dir = get_project_config_dir()
    project_dir.mkdir(parents=True, exist_ok=True)

    project_config = get_project_config_path()

    if not project_config.exists():
        # Start with minimal config that extends global
        config = {
            "#": "Project-specific configuration overrides - uncomment values to override global settings",
            "document_processing": {"data_dir": "./.btliu/data/documents"},
        }

        with open(project_config, "w") as f:
            json.dump(config, f, indent=2)

    return project_config
