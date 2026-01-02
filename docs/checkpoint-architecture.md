# Checkpoint 架构理解

## 概述

本文档记录了对 LangGraph Checkpoint、Store 和 Middleware 架构的正确理解，以及为什么当前使用 `create_agent()` + middleware + checkpoint 的组合存在架构问题。

## 核心概念辨析

### Checkpoint（检查点）

**用途**：保存图执行状态的快照，用于从某个节点恢复执行

**应该包含**：
- 图执行状态（graph state）
- 节点输出（node outputs）
- 通道值（channel values）
- 版本追踪（version tracking）

**不应该包含**：
- Middleware 对象
- 会话数据（session data）
- 用户数据（user data）

```python
# Checkpoint 的正确用途
checkpointer = PostgresSaver.from_conn_string(DB_URL)
graph = graph_builder.compile(checkpointer=checkpointer)

# 每次执行后保存状态快照
result = await graph.invoke(input_data, config={"configurable": {"thread_id": "123"}})
# 可以从 thread_id=123 恢复到这个状态
```

### Store（存储）

**用途**：跨线程/会话的持久化存储

**应该包含**：
- 用户数据（user data）
- 会话记忆（conversational memory）
- 跨线程共享的状态

```python
# Store 的正确用途
store = PostgresStore.from_conn_string(DB_URL)
graph = graph_builder.compile(store=store)

# 跨线程访问用户数据
await store.put(["user", "123"], "preferences", {"theme": "dark"})
```

### Middleware（中间件）

**用途**：运行时逻辑处理

**特点**：
- 不应该被序列化
- 每次运行时重新创建
- 包含无法 pickle 的组件（如 LLM 实例、文件句柄、lambda 函数）

```python
# Middleware 的正确使用
class SummarizationMiddleware:
    def __init__(self, llm):  # LLM 包含 threading._lock
        self.llm = llm

    # Middleware 应该在外部处理，不放入 state
```

## 当前架构的问题

### 问题代码结构

```
create_agent()
    ↓
添加 middleware 到 state
    ↓
Checkpoint 尝试序列化整个 state
    ↓
Middleware 无法序列化 💥
    ↓
需要自定义序列化器变通
```

### 根本原因

`create_agent()` 的设计会导致 middleware 被混入 graph state，当使用 checkpoint 时会尝试序列化这些无法 pickle 的对象。

### 变通代码的代价

当前代码在 `btliu/cli/cli.py` (lines 173-285) 有约 113 行的变通代码：

```python
# ============================================================================
# WORKAROUND: LangGraph Middleware Serialization Issue
# ============================================================================
# 这是为了解决 create_agent + middleware + checkpoint 的组合问题
```

这个变通方案包括：
- `_filter_unpicklable()` 函数
- `_CustomJsonPlusSerializer` 类
- 数据库初始化代码

## 正确的架构模式

### 使用 StateGraph 而非 create_agent

```python
from langgraph.graph import StateGraph
from typing import TypedDict

class AgentState(TypedDict):
    messages: list[BaseMessage]
    # 不包含 middleware！

# 使用 StateGraph 自定义图
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)

# Middleware 在外部处理，不放入 state
compiled = graph.compile(checkpointer=checkpointer)
```

### Middleware 外部化

```python
# Middleware 在运行时注入，不在 state 中
async def agent_node(state: AgentState, config):
    # 从外部获取 middleware，而不是从 state
    middleware = config["configurable"].get("middleware")
    if middleware:
        state = middleware.before_invoke(state)
    # ... 处理逻辑 ...
    return state
```

## 参考资料

- [LangGraph Persistence Documentation](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph v1 Documentation](https://docs.langchain.com/oss/python/releases/langgraph-v1)
- [Type is not msgpack serializable: AIMessage · Issue #5248](https://github.com/langchain-ai/langgraph/issues/5248)
- [Object of type HumanMessage is not JSON serializable · Issue #5511](https://github.com/langchain-ai/langgraph/issues/5511)

## 关键要点

1. **Checkpoint 不是数据持久化方案** - 它是图执行状态的快照
2. **Middleware 不应该被序列化** - 它是运行时对象
3. **Store 才是数据持久化** - 用于会话和用户数据
4. **create_agent() 有局限性** - 复杂场景应使用 StateGraph
