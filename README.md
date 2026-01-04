<div align="center">

# 🤖 btliu

### 企业级 RAG 系统 | 多智能体协作 | 智能文档处理

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.1.1+-brightgreen.svg)](https://python.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0.4+-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**一个功能强大的 RAG (检索增强生成) 系统，支持多智能体协作、智能文档处理和多种检索策略。**

</div>

---

## 📋 目录

- [概述](#概述)
- [核心特性](#核心特性)
- [快速开始](#快速开始)
- [系统架构](#系统架构)
- [核心功能](#核心功能)
- [配置指南](#配置指南)
- [使用指南](#使用指南)
- [开发指南](#开发指南)
- [部署](#部署)
- [故障排除](#故障排除)

---

## 🎯 概述

**btliu** 是一个基于 **LangChain 1.1+** 和 **LangGraph 1.0+** 构建的企业级 RAG 系统。它通过结合大语言模型、向量检索和智能体技术，提供强大的文档问答和知识管理能力。

### 核心价值

- 🚀 **快速响应**: 懒加载架构，CLI 启动时间 < 100ms
- 🧠 **智能检索**: 支持向量相似度、BM25 和混合检索策略
- 🤖 **多智能体**: 专业化 Agent 协作完成复杂任务
- 📚 **文档处理**: 自动处理 PDF、Markdown、JSON 等多种格式
- 🔌 **可扩展**: MCP 协议支持，轻松集成外部服务

### 应用场景

- 📖 **知识库问答**: 基于企业文档的智能问答系统
- 🔍 **技术文档检索**: 快速查找代码和文档中的信息
- 📊 **数据分析**: 结合检索和计算的数据分析任务
- 🛠️ **自动化助手**: 文件操作、系统管理等自动化任务

---

## ✨ 核心特性

### 双模式执行

| 模式 | 适用场景 | 特点 |
|------|----------|------|
| **UCAgent** | 通用任务、文件操作、系统管理 | 通用性强，工具丰富 |
| **RAG** | 文档问答、知识检索 | 检索准确，支持记忆 |

### 智能文档处理

- 📄 **多格式支持**: PDF, TXT, MD, JSON, DOCX, XLSX, PPTX
- 🔄 **智能索引**: 基于 Hash 的增量更新，只处理变更文件
- ✂️ **自适应分割**: 根据内容类型选择最佳分割策略
- 🧹 **内容清洗**: Unicode 规范化、控制字符移除

### 多策略检索

- 🔍 **向量相似度**: 语义搜索，基于 embeddings
- 📝 **BM25**: 关键词匹配，精确检索
- 🎯 **混合检索**: 结合两种策略，提高召回率和准确率

### MCP 集成

- 🔌 **多服务器支持**: 同时连接多个 MCP 服务
- 🔄 **动态工具加载**: 运行时加载 MCP 提供的工具
- ⚡ **自动重连**: 连接断开时自动重连

### 多模型支持

- 🌐 **多提供商**: OpenAI、Ollama、Zhipu AI、DeepSeek
- 🎛️ **灵活配置**: 支持 JSON 和 TOML 配置文件
- 🔧 **环境变量**: 支持环境变量覆盖配置

---

## 🚀 快速开始

### 环境要求

- Python 3.11+
- Docker (可选，用于容器化部署)

### 安装

```bash
# 克隆仓库
git clone https://github.com/your-org/btliu.git
cd btliu

# 安装依赖
pip install -r requirements.txt
```

### 配置

创建 `config.json` 配置文件：

```json
{
  "models": {
    "chat": {
      "default": {
        "provider": "langchain_openai.ChatOpenAI",
        "model": "glm-4.5",
        "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
        "api_key": "${ZHIPU_API_KEY:-}"
      }
    },
    "embedding": {
      "default": {
        "provider": "langchain_ollama.OllamaEmbeddings",
        "model": "bge-m3:latest",
        "base_url": "http://localhost:11434"
      }
    }
  },
  "vector_store": {
    "qdrant_url": "http://localhost:6333",
    "collection_name": "rag_documents",
    "similarity_threshold": 0.6,
    "default_k": 4
  },
  "document_processing": {
    "data_dir": "./data/documents",
    "chunk_size": 1000,
    "chunk_overlap": 200
  }
}
```

### 运行

```bash
# 使用 CLI
btliu

# 或直接运行 Python 模块
python -m btliu
```

### 验证安装

```bash
# 检查版本
btliu --version

# 测试连接
btliu --test-connection
```

---

## 🏗️ 系统架构

```
btliu/
├── __init__.py              # 主入口，懒加载优化
├── __main__.py              # CLI 入口点
│
├── cli/                     # 交互式命令行界面
│   └── cli.py              # PromptToolkit 实现，支持模式切换
│
├── agents/                  # 多智能体系统
│   ├── agents.py           # Agent 创建和管理
│   ├── factory.py          # Agent 工厂类
│   ├── config.py           # Agent 配置预设
│   └── dynamic_prompts.py  # Agent 动态提示词
│
├── apps/                    # 应用层
│   ├── rag_app.py          # RAG 应用封装
│   └── ucagent_app.py      # UCAgent 应用封装
│
├── tools/                   # 工具层
│   ├── documents.py        # 文档管理 (Qdrant)
│   ├── retrievers.py       # 检索策略
│   ├── mcp_tools.py        # MCP 集成
│   └── memory_tools.py     # 跨线程记忆
│
├── config/                  # 配置管理
│   ├── config.py           # 多层配置系统
│   └── paths.py            # 路径管理
│
├── services/                # 服务层
│   └── document_monitor.py # 文档监控服务
│
└── common/                  # 通用组件
    └── types.py            # 类型定义
```

### 数据流

```
用户输入 → CLI → 应用层 (RAG/UCAgent)
                    ↓
              工具层 (检索/文档/MCP)
                    ↓
              配置层 (多模型/多提供商)
                    ↓
              存储层 (Qdrant/PostgreSQL)
```

---

## 🔧 核心功能

### 1. 双模式执行

#### UCAgent 模式

通用智能体模式，适用于广泛的任务：

```bash
[ucagent] > /ucagent
✓ UCAgent mode

[ucagent] > 帮我分析这个 Python 项目的结构
🔄 使用 UCAgent 处理...
```

**特点**：
- 文件系统操作（读写、搜索）
- 代码分析和执行
- 通用问题解答
- PII 数据自动脱敏

#### RAG 模式

检索增强模式，专注于文档问答：

```bash
[rag] > /rag
✓ RAG mode

[rag] > RAG 系统的核心组件有哪些？
🔍 检索相关文档...
RAG 系统的核心组件包括：
1. 文档加载器
2. 文本分割器
3. 嵌入模型
4. 向量数据库
5. 检索器
```

**特点**：
- 智能文档检索
- 上下文感知
- 跨会话记忆
- 引用来源追踪

### 2. 智能文档处理

#### 支持的格式

| 格式 | 扩展名 | 用途 |
|------|--------|------|
| PDF | `.pdf` | 技术文档、论文 |
| 文本 | `.txt`, `.md`, `.rst` | 文档、笔记 |
| 数据 | `.json`, `.csv` | 结构化数据 |
| 办公 | `.docx`, `.xlsx`, `.pptx` | Office 文档 |

#### 智能索引

```python
# 首次加载：处理所有文件
btliu > /reindex
✓ Reindexing documents...
Processed: 15 files, 342 chunks

# 后续加载：只处理变更文件
# 系统自动检测文件变化，只更新新增/修改的文档
```

### 3. 多策略检索

#### 向量相似度检索

```python
@tool
async def similarity_search(query: str, k: int = 4) -> str:
    """基于向量相似度的语义搜索"""
    # 使用 embeddings 计算语义相似度
    # 返回最相关的 k 个文档片段
```

#### BM25 检索

```python
@tool
async def bm25_search(query: str, k: int = 4) -> str:
    """基于 BM25 的关键词检索"""
    # 使用词频统计进行关键词匹配
    # 适用于精确匹配特定术语
```

#### 混合检索

```python
@tool
async def ensemble_search(query: str, k: int = 4) -> str:
    """结合向量和 BM25 的混合检索"""
    # 同时使用两种策略
    # 合并结果提高准确率
```

### 4. MCP 集成

```json
{
  "mcp_servers": {
    "filesystem": {
      "transport": "stdio",
      "command": "npx -y @modelcontextprotocol/server-filesystem",
      "args": ["/path/to/allowed/directory"]
    },
    "custom_service": {
      "transport": "streamable_http",
      "url": "http://localhost:8080/mcp"
    }
  }
}
```

---

## ⚙️ 配置指南

### 配置文件优先级

系统按以下优先级加载配置：

1. **项目配置**: `./config.json`
2. **全局配置**: `~/.btliu/config.json`
3. **默认配置**: 内置默认值

### 环境变量覆盖

任何 JSON 配置值都可以通过环境变量覆盖：

```bash
# 覆盖模型名称
export RAG_MODEL="gpt-4"

# 覆盖数据目录
export RAG_DATA_DIR="/mnt/data/docs"

# 覆盖 API Key
export ZHIPU_API_KEY="your-key-here"
```

### 多模型配置

```json
{
  "models": {
    "chat": {
      "glm-4": {
        "provider": "langchain_openai.ChatOpenAI",
        "model": "glm-4.5",
        "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
        "api_key": "${ZHIPU_API_KEY:-}",
        "temperature": 0.8
      },
      "gpt-4": {
        "provider": "langchain_openai.ChatOpenAI",
        "model": "gpt-4-turbo-preview",
        "api_key": "${OPENAI_API_KEY:-}"
      },
      "ollama": {
        "provider": "langchain_ollama.ChatOllama",
        "model": "qwen3:8b",
        "base_url": "http://localhost:11434"
      }
    }
  }
}
```

### 向量存储配置

```json
{
  "vector_store": {
    "qdrant_url": "http://localhost:6333",
    "collection_name": "rag_documents",
    "similarity_threshold": 0.6,
    "default_k": 4,
    "enable_caching": true
  }
}
```

**参数说明**：
- `similarity_threshold`: 相似度阈值 (0-1)，越高越严格
- `default_k`: 默认返回结果数量
- `enable_caching`: 启用查询缓存

---

## 📖 使用指南

### CLI 命令参考

#### 基础命令

```bash
# 启动系统
btliu

# 显示版本
btliu --version

# 测试连接
btliu --test-connection

# 显示帮助
btliu --help
```

#### CLI 交互命令

```bash
# 模式切换
/ucagent      # 切换到 UCAgent 模式
/rag          # 切换到 RAG 模式

# 文档管理
/reindex      # 重新索引文档
/status        # 显示系统状态

# 会话管理
/save         # 保存当前会话
/restore      # 恢复之前会话

# 系统命令
/help         # 显示帮助
/exit         # 退出系统
```

### 模式切换示例

```bash
# 当前在 UCAgent 模式
[ucagent] > 查询最近的文档更新

# 切换到 RAG 模式
[ucagent] > /rag
✓ RAG mode enabled

# 现在 RAG 模式
[rag] > 查询最近的文档更新
🔍 从文档中检索...
```

### 文档管理

#### 添加文档

```bash
# 将文档放入配置的 data_dir
cp my_document.pdf ~/.btliu/data/documents/

# 系统自动检测并索引新文档
```

#### 重新索引

```bash
# 强制重新索引所有文档
[ucagent] > /reindex
✓ Reindexing all documents...
Processed: 15 documents
Created: 342 chunks
```

### 高级用法

#### 自定义检索参数

```python
# 在代码中自定义检索
from btliu.tools.retrievers import similarity_search

results = await similarity_search(
    query="Python 异步编程",
    k=10,  # 返回前 10 个结果
    threshold=0.7  # 相似度阈值
)
```

#### 使用特定模型

```bash
# 通过环境变量指定模型
RAG_MODEL="gpt-4" btliu

# 或在配置中设置默认模型
```

---

## 👨‍💻 开发指南

### 项目结构详解

```
btliu/
├── __init__.py              # 主入口，懒加载优化
├── __main__.py              # CLI 入口点
│
├── cli/                     # CLI 层
│   └── cli.py              # PromptToolkit 交互界面
│
├── agents/                  # Agent 层
│   ├── agents.py           # Agent 工厂函数
│   ├── factory.py          # AgentFactory 类
│   ├── config.py           # Agent 配置预设
│   └── dynamic_prompts.py  # Agent 提示词模板
│
├── apps/                    # 应用层
│   ├── rag_app.py          # RAG 应用
│   └── ucagent_app.py      # UCAgent 应用
│
├── tools/                   # 工具层
│   ├── documents.py        # DocumentManager 类
│   ├── retrievers.py       # 检索函数
│   ├── mcp_tools.py        # MCP 工具集成
│   └── memory_tools.py     # 记忆工具
│
├── config/                  # 配置层
│   ├── config.py           # ConfigManager 类
│   └── paths.py            # 路径工具函数
│
├── services/                # 服务层
│   └── document_monitor.py # 文件监控服务
│
└── common/                  # 通用层
    └── types.py            # 类型定义
```

### 添加新的 Agent

#### 1. 定义提示词

在 `btliu/agents/dynamic_prompts.py` 中添加：

```python
def custom_agent_prompt() -> str:
    """返回自定义 Agent 的系统提示词"""
    return """你是一个专业的数据分析助手。

你的能力包括：
- 分析数据文件
- 生成可视化报告
- 提供数据洞察

请使用 markdown 格式输出结果。
"""
```

#### 2. 添加配置

在 `btliu/agents/config.py` 中添加：

```python
CUSTOM_PRESET = AgentPreset(
    name="custom",
    prompt_func=custom_agent_prompt,
    tools=["search_files", "read_file", "run_python"],
    middleware=["piih_mask", "tool_retry"]
)
```

#### 3. 注册 Agent

在 `btliu/agents/agents.py` 的 `PRESETS` 字典中添加：

```python
from btliu.agents.config import CUSTOM_PRESET

PRESETS = {
    # ... 其他 presets
    "custom": CUSTOM_PRESET,
}
```

### 添加新的检索策略

在 `btliu/tools/retrievers.py` 中添加：

```python
from langchain_core.tools import tool

@tool
async def my_custom_search(
    query: str,
    k: int = 4,
    runtime: ToolRuntime[RuntimeContext],
) -> str:
    """自定义检索策略实现

    Args:
        query: 搜索查询
        k: 返回结果数量
        runtime: 运行时上下文

    Returns:
        检索结果字符串
    """
    doc_manager = runtime.state["doc_manager"]

    # 你的检索逻辑
    results = doc_manager.vector_store.similarity_search(query, k=k)

    # 格式化结果
    return format_results(results)
```

然后添加到 `get_all_retrievers()` 函数中。

### 测试

```bash
# 运行测试
pytest tests/

# 运行特定测试
pytest tests/test_documents.py -v

# 查看测试覆盖率
pytest --cov=btliu --cov-report=html
```

---

## 🐳 部署

### Docker 部署

#### 1. 构建镜像

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "-m", "btliu"]
```

```bash
# 构建镜像
docker build -t btliu:latest .

# 运行容器
docker run -d \
  -p 8000:8000 \
  -v ~/.btliu:/app/.btliu \
  -e ZHIPU_API_KEY=${ZHIPU_API_KEY} \
  btliu:latest
```

### Docker Compose

```yaml
version: '3.8'

services:
  btliu:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ZHIPU_API_KEY=${ZHIPU_API_KEY}
    volumes:
      - ./data:/app/data
      - ~/.btliu:/app/.btliu
    depends_on:
      - qdrant
      - ollama

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

volumes:
  qdrant_data:
  ollama_data:
```

### 环境变量配置

```bash
# 必需变量
export ZHIPU_API_KEY="your-zhipu-key"
export OPENAI_API_KEY="your-openai-key"

# 可选变量
export RAG_MODEL="glm-4"
export RAG_DATA_DIR="/path/to/documents"
export RAG_LOG_LEVEL="INFO"
export RAG_QDRANT_URL="http://localhost:6333"
```

---

## 🔍 故障排除

### 常见问题

#### 1. 导入错误

**问题**: `ModuleNotFoundError: No module named 'langchain'`

**解决方案**:
```bash
pip install -r requirements.txt
```

#### 2. API 连接失败

**问题**: `Connection refused` 或 `API Error`

**解决方案**:
```bash
# 检查 API Key
echo $ZHIPU_API_KEY

# 测试连接
btliu --test-connection

# 检查网络
curl https://open.bigmodel.cn/api/coding/paas/v1/models
```

#### 3. 文档加载失败

**问题**: PDF 无法加载

**解决方案**:
```bash
# 安装 pymupdf
pip install pymupdf

# 验证文档目录
ls -la ~/.btliu/data/documents/
```

#### 4. Ollama 连接失败

**问题**: `Connection refused` on port 11434

**解决方案**:
```bash
# 启动 Ollama
ollama serve

# 拉取嵌入模型
ollama pull bge-m3:latest

# 测试连接
curl http://localhost:11434/api/tags
```

#### 5. Qdrant 连接失败

**问题**: 无法连接到向量数据库

**解决方案**:
```bash
# 使用 Docker 启动 Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# 检查连接
curl http://localhost:6333/collections
```

### 调试技巧

#### 启用调试日志

```json
{
  "log_level": "DEBUG",
  "debug_mode": true
}
```

#### 查看详细错误

```bash
# 启用详细输出
btliu --verbose

# 或设置环境变量
RAG_LOG_LEVEL=DEBUG btliu
```

#### 性能优化

- 增大 `chunk_size` 减少文档分片
- 调高 `similarity_threshold` 提高检索精度
- 启用缓存减少重复计算
- 使用本地模型降低延迟

---

## 📝 开发路线图

### 已完成 ✅

- [x] 双模式执行系统
- [x] 智能文档处理
- [x] 多策略检索
- [x] MCP 集成
- [x] 多模型支持
- [x] 交互式 CLI

### 进行中 🚧

- [ ] Web UI 界面
- [ ] 更多文档格式支持
- [ ] 分布式部署支持
- [ ] 性能优化和监控

### 计划中 📋

- [ ] 多模态检索（图片、表格）
- [ ] 知识图谱集成
- [ ] 自定义 Agent 编排器
- [ ] 企业级权限管理

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 🤝 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

## 📧 联系

- 问题反馈: [GitHub Issues](https://github.com/Aaron-xx/langchain-agent/issues)

---

<div align="center">

**Built with ❤️ using LangChain & LangGraph**

[⬆ 返回顶部](#)
</div>
