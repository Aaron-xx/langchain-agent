"""Lightweight configuration - supports both JSON and TOML"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class Config:
    """Lightweight config class with JSON/TOML support"""

    # Default configuration
    DEFAULTS = {
        "model": "glm-4",
        "embedding_model": "bge-m3",
        "data_dir": "data",
        "persist_dir": "data/chroma",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "default_mode": "ucagent",
        "mcp_servers": {}
    }

    def __init__(self, **kwargs):
        self._nested_config = kwargs

        self.__dict__.update({
            '_nested_config': kwargs
        })

    @classmethod
    def from_file(cls, config_path: Optional[str] = None) -> 'Config':
        """Load config - auto-detect JSON/TOML format"""
        config_data = {}

        # Try JSON first
        json_path = config_path or "config.json"
        if Path(json_path).exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                logger.info(f"Loaded JSON config from {json_path}")
            except Exception as e:
                logger.error(f"Failed to load JSON: {e}")

        # Try TOML if JSON failed
        if not config_data:
            toml_path = config_path or "config.toml"
            if Path(toml_path).exists():
                try:
                    import toml  # Lazy import
                    with open(toml_path, 'r', encoding='utf-8') as f:
                        config_data = toml.load(f)
                    logger.info(f"Loaded TOML config from {toml_path}")
                except ImportError:
                    logger.warning("Install toml for TOML support: pip install toml")
                except Exception as e:
                    logger.error(f"Failed to load TOML: {e}")

        # Environment variables override (RAG_ prefix)
        for key in cls.DEFAULTS:
            env_value = os.getenv(f"RAG_{key.upper()}")
            if env_value:
                config_data[key] = env_value

        return cls(**config_data)

    def __getattr__(self, name: str) -> Any:
        """Auto access nested config values"""
        # First check if it's a top-level key
        if name in self._nested_config:
            return self._nested_config[name]

        # Then search in all nested dictionaries
        for key in self._nested_config:
            result = self.get(f'{key}.{name}')
            if result is not None:
                return result

        # Return default if nothing found
        return self.DEFAULTS.get(name)

    def get(self, key: str, default: Any = None) -> Any:
        """Dot notation access for nested keys"""
        if '.' in key:
            parts = key.split('.')
            value = self._nested_config
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value
        return getattr(self, key, default)

    @property
    def embedding_model(self):
        """Get embedding model instance with dynamic configuration."""
        import importlib
        import os

        emb_config = self.get('models.embedding.bge-m3')
        if not emb_config or 'provider' not in emb_config:
            from langchain_community.embeddings import OllamaEmbeddings
            return OllamaEmbeddings(model='bge-m3:latest')

        provider = emb_config['provider']
        params = {k: v for k, v in emb_config.items() if k != 'provider'}

        # Process environment variables
        for k, v in params.items():
            if isinstance(v, str) and v.startswith('${') and v.endswith('}'):
                env_expr = v[2:-1]
                if ':-' in env_expr:
                    var, default = env_expr.split(':-', 1)
                    params[k] = os.getenv(var, default)
                else:
                    params[k] = os.getenv(env_expr)

        module_path, class_name = provider.rsplit('.', 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(**params)

    @property
    def chat_model(self):
        """Get chat model instance with dynamic configuration."""
        import importlib
        import os

        chat_config = self.get('models.chat.glm-4')
        if not chat_config or 'provider' not in chat_config:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model='glm-4.5',
                base_url='https://open.bigmodel.cn/api/paas/v4',
                model_kwargs={
                    "stream_options": {"include_usage": True}
                    }
            )

        provider = chat_config['provider']
        params = {k: v for k, v in chat_config.items() if k != 'provider'}

        # Process environment variables
        for k, v in params.items():
            if isinstance(v, str) and v.startswith('${') and v.endswith('}'):
                env_expr = v[2:-1]
                if ':-' in env_expr:
                    var, default = env_expr.split(':-', 1)
                    params[k] = os.getenv(var, default)
                else:
                    params[k] = os.getenv(env_expr)

        module_path, class_name = provider.rsplit('.', 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(**params)

    def save(self, path: str, format: str = "auto"):
        """Save config - auto-detect format from extension"""
        data = {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}

        if format == "auto":
            format = "toml" if path.endswith('.toml') else "json"

        if format == "toml":
            import toml
            with open(path, 'w', encoding='utf-8') as f:
                toml.dump(data, f)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

# Global singleton
_config: Optional[Config] = None

def get_config(config_file: Optional[str] = None) -> Config:
    """Get global config instance"""
    global _config
    if _config is None:
        _config = Config.from_file(config_file)
    return _config