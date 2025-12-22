"""超简洁配置系统 - 全部代码在这一个文件"""

import json
import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv() 

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    """超简洁配置系统 - 核心类"""
    
    # 4 个 Provider 预设
    PROVIDERS = {
        'openai': {
            'chat': 'langchain_openai.ChatOpenAI',
            'embedding': 'langchain_openai.OpenAIEmbeddings',
        },
        'ollama': {
            'chat': 'langchain_ollama.ChatOllama',
            'embedding': 'langchain_ollama.OllamaEmbeddings',
        },
        'anthropic': {
            'chat': 'langchain_anthropic.ChatAnthropic',
            'embedding': 'langchain_community.embeddings.HuggingFaceEmbeddings',
        },
        'huggingface': {
            'chat': 'langchain_huggingface.ChatHuggingFace',
            'embedding': 'langchain_huggingface.HuggingFaceEmbeddings',
        },
    }
    
    def __init__(self, config_path: str = 'config.json'):
        """
        初始化配置系统
        
        Args:
            config_path: 配置文件路径
        """
        logger.info(f"开始加载配置: {config_path}")
        
        # 1. 加载 JSON 配置
        if not Path(config_path).exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self._raw = json.load(f)
        
        logger.info("✅ JSON 配置加载完成")
        
        # 2. 替换环境变量
        self._raw = self._replace_env(self._raw)
        logger.info("✅ 环境变量替换完成")
        
        # 3. 构建索引（扁平化所有键）
        self._index: Dict[str, Any] = {}
        self._flatten(self._raw, '')
        logger.info(f"✅ 索引构建完成，共 {len(self._index)} 个键")
        
        # 4. 模型缓存
        self._model_cache: Dict[str, Any] = {}
    
    def _flatten(self, obj: Any, prefix: str) -> None:
        """
        递归扁平化配置，构建多层级索引
        
        支持三种访问方式：
        1. 简写: di.persist_directory
        2. 路径: di['vector_store.persist_directory']
        3. 模糊: di.persistdirectory
        
        Args:
            obj: 配置对象
            prefix: 当前路径前缀
        """
        if isinstance(obj, dict):
            for key, value in obj.items():
                full_path = f"{prefix}.{key}" if prefix else key
                
                if isinstance(value, dict):
                    # 继续递归
                    self._flatten(value, full_path)
                else:
                    # 添加到索引
                    # 索引策略：支持多种访问方式
                    self._index[key] = value                         # 最后一级键 (简写)
                    self._index[full_path] = value                   # 完整路径
                    self._index[key.replace('_', '').lower()] = value  # 去下划线（模糊）
    
    def _replace_env(self, obj: Any) -> Any:
        """
        递归替换环境变量
        
        支持两种格式：
        - ${VAR}: 获取环境变量，不存在则为空
        - ${VAR:-default}: 获取环境变量，不存在则使用默认值
        
        Args:
            obj: 待处理对象
        
        Returns:
            替换后的对象
        """
        if isinstance(obj, str):
            def replacer(match):
                expr = match.group(1)
                if ':-' in expr:
                    var, default = expr.split(':-', 1)
                    return os.getenv(var.strip(), default.strip())
                else:
                    return os.getenv(expr.strip(), '')
            
            return re.sub(r'\$\{([^}]+)\}', replacer, obj)
        
        elif isinstance(obj, dict):
            return {k: self._replace_env(v) for k, v in obj.items()}
        
        elif isinstance(obj, list):
            return [self._replace_env(v) for v in obj]
        
        return obj
    
    # ========== 访问接口 ==========
    
    def __getattr__(self, key: str) -> Any:
        """
        属性访问方式：di.persist_directory
        
        Args:
            key: 配置键
        
        Returns:
            配置值
        """
        # 避免干扰私有属性
        if key.startswith('_'):
            return super().__getattribute__(key)
        
        if key in self._index:
            logger.debug(f"属性访问: {key}")
            return self._index[key]
        
        raise AttributeError(f"配置不存在: {key}")
    
    def __getitem__(self, key: str) -> Any:
        """
        字典访问方式：di['vector_store.persist_directory']
        
        Args:
            key: 配置键
        
        Returns:
            配置值
        """
        if key in self._index:
            logger.debug(f"字典访问: {key}")
            return self._index[key]
        
        raise KeyError(f"配置不存在: {key}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        函数访问方式：di.get('key', default='value')
        
        Args:
            key: 配置键
            default: 默认值
        
        Returns:
            配置值或默认值
        """
        result = self._index.get(key, default)
        logger.debug(f"函数访问: {key} → {result}")
        return result
    
    def get_group(self, prefix: str) -> Dict[str, Any]:
        """
        获取配置组：di.get_group('vector_store')
        
        返回所有以 prefix 开头的配置
        
        Args:
            prefix: 配置前缀
        
        Returns:
            配置字典
        """
        prefix_with_dot = prefix + '.'
        result = {
            k: v for k, v in self._index.items() 
            if k.startswith(prefix_with_dot)
        }
        
        # 移除前缀，只保留相对路径
        result = {
            k[len(prefix_with_dot):]: v 
            for k, v in result.items()
        }
        
        logger.debug(f"获取配置组: {prefix} → {len(result)} 项")
        return result
    
    def list_keys(self, pattern: str = None) -> List[str]:
        """
        列出所有配置键
        
        Args:
            pattern: 可选的过滤模式
        
        Returns:
            键列表
        """
        keys = sorted(self._index.keys())
        
        if pattern:
            keys = [k for k in keys if pattern.lower() in k.lower()]
        
        logger.debug(f"列出 {len(keys)} 个键")
        return keys
    
    def search(self, pattern: str) -> Dict[str, Any]:
        """
        搜索配置
        
        Args:
            pattern: 搜索模式
        
        Returns:
            匹配的配置字典
        """
        pattern = pattern.lower()
        result = {
            k: v for k, v in self._index.items()
            if pattern in k.lower()
        }
        
        logger.info(f"搜索 '{pattern}' 找到 {len(result)} 个结果")
        return result
    
    # ========== 模型加载 ==========
    def _get_default_provider(self, model_type: str = 'chat') -> str:
        """根据 environment_type 获取默认 provider"""
        # 直接从原始数据读，不用索引
        env_type = self._raw.get('environment_type', 'external')
        
        if env_type == 'internal':
            return 'ollama'
        else:
            return 'openai'
    
    def chat(self, provider: str = 'openai') -> Any:
        """
        获取聊天模型：di.chat('openai')
        
        Args:
            provider: provider 名称 (openai|ollama|anthropic|huggingface)
        
        Returns:
            聊天模型实例
        """
        # if provider is None:
        provider = self._get_default_provider('chat')
        
        return self._get_model(provider, 'chat')
    
    def embedding(self, provider: str = 'openai') -> Any:
        """
        获取 embedding 模型：di.embedding('ollama')
        
        Args:
            provider: provider 名称
        
        Returns:
            embedding 模型实例
        """
        # if provider is None:
        provider = self._get_default_provider('embedding')
        return self._get_model(provider, 'embedding')
    
    def _get_model(self, provider: str, model_type: str) -> Any:
        """
        内部方法：创建和缓存模型实例
        
        Args:
            provider: provider 名称
            model_type: 模型类型 ('chat' 或 'embedding')
        
        Returns:
            模型实例
        """
        # 检查缓存
        cache_key = f"{provider}_{model_type}"
        if cache_key in self._model_cache:
            logger.info(f"模型缓存命中: {cache_key}")
            return self._model_cache[cache_key]
        
        # 检查 provider 是否存在
        if provider not in self.PROVIDERS:
            available = ', '.join(self.PROVIDERS.keys())
            raise ValueError(f"未知 provider: '{provider}'。支持: {available}")
        
        if model_type not in self.PROVIDERS[provider]:
            available = ', '.join(self.PROVIDERS[provider].keys())
            raise ValueError(f"Provider '{provider}' 不支持 '{model_type}'。支持: {available}")
        
        try:
            logger.info(f"开始加载 {provider} {model_type} 模型...")
            
            # 获取类路径
            class_path = self.PROVIDERS[provider][model_type]
            module_path, class_name = class_path.rsplit('.', 1)
            
            # 动态导入
            import importlib
            try:
                module = importlib.import_module(module_path)
            except ImportError as e:
                raise ImportError(
                    f"无法导入 {module_path}。"
                    f"请先安装: pip install langchain-{provider}"
                ) from e
            
            cls = getattr(module, class_name)
            
            # 获取配置参数
            config_key = f"models.{model_type}.{provider}"
            params = self.get_group(config_key)
            
            # 清理参数（移除不必要的字段）
            params = {k: v for k, v in params.items() if k != 'provider'}
            
            logger.info(f"模型参数: {params}")
            
            # 创建实例
            instance = cls(**params)
            
            # 缓存
            self._model_cache[cache_key] = instance
            
            logger.info(f"✅ {provider} {model_type} 模型加载成功")
            return instance
        
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise
    
    # ========== 工具方法 ==========
    
    def get_raw(self) -> Dict[str, Any]:
        """获取原始配置字典"""
        return self._raw.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            'total_keys': len(self._index),
            'cached_models': len(self._model_cache),
            'providers': list(self.PROVIDERS.keys()),
            'all_keys': self.list_keys(),
        }
    
    def print_config(self) -> None:
        """打印完整配置（用于调试）"""
        print("\n" + "="*60)
        print("📋 完整配置信息")
        print("="*60)
        
        # 按首字母分组显示
        current_group = None
        for key in self.list_keys():
            group = key.split('.')[0]
            if group != current_group:
                if current_group is not None:
                    print()
                print(f"\n[{group}]")
                current_group = group
            
            value = self._index[key]
            # 对敏感信息进行脱敏
            if any(s in key.lower() for s in ['key', 'token', 'password', 'secret']):
                value = '***' if value else value
            print(f"  {key}: {value}")
        
        print("\n" + "="*60 + "\n")


def get_config(config_path: str = 'config.json') -> Config:
    """
    创建配置容器
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        Config 实例
    
    使用示例:
        di = get_config('config.json')
        value = di.persist_directory
    """
    return Config(config_path)


# 便利函数：注册自定义 provider
def register_provider(name: str, chat_class: str, embedding_class: str) -> None:
    """
    注册自定义 provider（可选）
    
    使用示例:
        register_provider('my_provider',
            'my_lib.ChatModel',
            'my_lib.Embeddings'
        )
    """
    Config.PROVIDERS[name] = {
        'chat': chat_class,
        'embedding': embedding_class,
    }
    logger.info(f"✅ Provider '{name}' 已注册")


__all__ = ['Config', 'get_config', 'register_provider']