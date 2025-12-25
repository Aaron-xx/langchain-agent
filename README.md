# LangChain RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system built with LangChain 1.1.0 and LangGraph 1.0.4, featuring multiple execution modes, specialized agents, and advanced document processing capabilities.

## Features

- **Dual Execution Modes**:
  - **UCAgent Mode**: Agent-based with specialized task-specific agents
  - **RAG Mode**: Graph-based with memory-enabled conversation support

- **Specialized Agents**:
  - **Decision Agent**: Decision-making with human-in-loop middleware
  - **Analysis Agent**: Data analysis with PII masking protection
  - **Planning Agent**: Strategic planning with call limit controls
  - **Execution Agent**: Task execution with full tool access
  - **RAG Agent**: Retrieval-augmented generation
  - **UCAgent**: Universal agent with filesystem and PII capabilities

- **Advanced Document Management**:
  - Support for PDF, TXT, MD, JSON formats
  - Smart reindexing with hash-based change detection
  - Adaptive splitting for Markdown vs regular content
  - Unicode normalization and content cleaning

- **Multiple Retrieval Strategies**:
  - Vector similarity search
  - BM25 keyword matching
  - Ensemble retrieval combining both approaches

- **MCP Integration**: Model Context Protocol support for external services

- **Flexible Configuration**: JSON/TOML support with environment variable overrides

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your API keys:
```bash
# For GLM (Zhipu AI)
export ZHIPU_API_KEY="your-glm-key-here"

# For DeepSeek
export DEEPSEEK_API_KEY="your-deepseek-key-here"

# For OpenAI-compatible models
export OPENAI_API_KEY="your-openai-key-here"
```

3. Configure Ollama for embeddings (optional):
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull BGE-M3 embedding model
ollama pull bge-m3:latest
```

## Usage

### Running the System

```bash
python main.py
```

### CLI Commands

Once running, you can use these commands:

- **Direct Query**: Just type your question
- `/ucagent` - Switch to UCAgent mode
- `/rag` - Switch to RAG mode
- `/help` - Show available commands
- `/exit` or `/quit` - Exit the system

### Mode Examples

#### UCAgent Mode (Default)
```
[ucagent] > What are the key features of microservices architecture?
🔍 Processing with UCAgent...
The key features of microservices architecture include:
1. Service independence...
```

#### RAG Mode
```
[rag] > /rag
✓ RAG mode
[rag] > Explain the implementation of vector databases
🔍 Retrieving relevant documents...
Vector databases implement...
```

## Configuration

### Configuration Structure

The system uses a hierarchical configuration in `config.json`:

```json
{
  "environment": "development",
  "log_level": "INFO",
  "models": {
    "chat": {
      "glm-4": {
        "provider": "langchain_openai.ChatOpenAI",
        "model": "glm-4.5",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key": "${ZHIPU_API_KEY:-}",
        "temperature": 0.8
      },
      "deepseek": {
        "provider": "langchain_openai.ChatOpenAI",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "${DEEPSEEK_API_KEY:-}"
      }
    },
    "embedding": {
      "bge-m3": {
        "provider": "langchain_ollama.OllamaEmbeddings",
        "model": "bge-m3:latest",
        "base_url": "http://ollama:11434"
      }
    }
  },
  "vector_store": {
    "persist_directory": "./data/chroma_db",
    "collection_name": "rag_documents",
    "similarity_threshold": 0.78,
    "max_retrieved_docs": 4
  },
  "document_processing": {
    "data_dir": "./data/documents",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "enable_smart_index": true
  }
}
```

### Environment Variables

Override any configuration value with environment variables:
```bash
export RAG_MODEL="glm-4"
export RAG_DATA_DIR="/path/to/your/documents"
export RAG_CHUNK_SIZE="1500"
```

### TOML Support

You can also use TOML configuration by creating `config.toml`:
```toml
[models.chat.glm-4]
provider = "langchain_openai.ChatOpenAI"
model = "glm-4.5"
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key = "${ZHIPU_API_KEY:-}"
temperature = 0.8
```

## Document Management

### Adding Documents

Place your documents in the configured data directory (default: `./data/documents/`):

- **PDF files** (.pdf): Automatically extracted with PyPDF
- **Text files** (.txt): Direct content loading
- **Markdown files** (.md): Preserved header structure
- **JSON files** (.json): Nested data extraction

The system automatically:
- Detects file changes using hash-based indexing
- Splits documents adaptively based on content type
- Cleans and normalizes text content
- Stores in ChromaDB for fast retrieval

### Document Processing Features

- **Smart Reindexing**: Only processes changed documents
- **Adaptive Splitting**: Different strategies for Markdown vs text
- **Content Cleaning**: Unicode normalization and control character removal
- **Metadata Preservation**: Source tracking and page numbering

## Architecture

```
src/
├── common/               # Shared types and utilities
│   └── types.py          # Type definitions (RAGState, RuntimeContext)
├── config/               # Configuration management
│   └── config.py         # JSON/TOML configuration with env overrides
├── tools/                # Document management and retrieval
│   ├── documents.py      # Document processing pipeline
│   ├── retrievers.py     # Search strategies (vector, BM25, ensemble)
│   └── mcp_tools.py      # MCP protocol tools
├── apps/                 # Chain-based applications
│   ├── ucagent_app.py    # UCAgent application wrapper
│   └── rag_app.py        # RAG application wrapper
├── agents/               # Agent implementations
│   ├── agents.py         # Agent factory and creation
│   ├── factory.py        # Agent factory class
│   ├── config.py         # Agent configurations
│   └── dynamic_prompts.py# Agent-specific prompts
├── services/             # Service layer
└── cli/                  # Command-line interface
    └── cli.py            # Interactive CLI with mode switching
```

## Examples

### Basic Query in UCAgent Mode
```
[ucagent] > What are the benefits of containerization?
🔄 Using UCAgent with retrieval and MCP tools...
Containerization offers numerous benefits:
1. Portability across environments
2. Resource efficiency
3. Faster deployment cycles
4. Microservices enablement
```

### RAG Mode with Memory
``[rag] > /rag
✓ RAG mode
[rag] > How does Kubernetes work?
🔍 Retrieving from documents...
Kubernetes is a container orchestration platform that...
[rag] > What were the key components mentioned?
🤖 Based on our conversation, the key components were:
1. Control Plane
2. Worker Nodes
3. Pods
4. Services
```

### File Upload Support
When using the web interface (agent-chat-ui), you can upload files directly:
- Files are automatically extracted and saved
- Background processing indexes new documents
- Support for multiple file formats

## Development

### Adding New Retrieval Strategies

1. Create retriever function in `src/tools/retrievers.py`:
```python
@tool
async def custom_search(query: str, runtime: ToolRuntime[RuntimeContext], k: int = 4) -> str:
    """Custom search implementation."""
    # Your implementation here
    return results
```

2. Add to `get_all_retrievers()` function

### Adding New Agents

1. Define prompt function in `src/agents/dynamic_prompts.py`
2. Add configuration in `src/agents/config.py`
3. Import and add to `PRESETS` dictionary

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Ollama Connection**: Check Ollama is running on port 11434
   ```bash
   ollama serve
   ```

3. **Document Loading**: Verify documents are in the correct directory
   ```bash
   ls -la ./data/documents/
   ```

4. **API Key Issues**: Check environment variables are set
   ```bash
   echo $ZHIPU_API_KEY
   ```

5. **Memory Issues**: Adjust chunk_size for large documents
   ```json
   {
     "document_processing": {
       "chunk_size": 500,
       "chunk_overlap": 100
     }
   }
   ```

### Debug Mode

Enable debug logging in config:
```json
{
  "log_level": "DEBUG",
  "debug_mode": true
}
```

## Advanced Configuration

### Model Configuration

You can configure multiple chat models:

```json
{
  "models": {
    "chat": {
      "gpt-4": {
        "provider": "langchain_openai.ChatOpenAI",
        "model": "gpt-4-turbo-preview",
        "api_key": "${OPENAI_API_KEY:-}",
        "temperature": 0.7
      },
      "claude": {
        "provider": "langchain_anthropic.ChatAnthropic",
        "model": "claude-3-opus-20240229",
        "api_key": "${ANTHROPIC_API_KEY:-}"
      }
    }
  }
}
```

### MCP Server Configuration

Configure external MCP servers:

```json
{
  "mcp_servers": {
    "custom_service": {
      "transport": "streamable_http",
      "url": "http://localhost:8080/mcp",
      "timeout": 30,
      "headers": {
        "Authorization": "Bearer ${MCP_TOKEN:-}"
      }
    }
  }
}
```

## Performance Tuning

### Vector Store Optimization

```json
{
  "vector_store": {
    "persist_directory": "./data/chroma_db",
    "similarity_threshold": 0.8,
    "max_retrieved_docs": 5,
    "enable_caching": true,
    "cache_ttl": 3600
  }
}
```

### Document Processing

```json
{
  "document_processing": {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "enable_smart_index": true
  }
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.