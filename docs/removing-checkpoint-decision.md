# 移除 Checkpoint 决策记录

## 决策概述

**日期**: 2025-12-30

**决策**: 移除 Checkpoint 功能，删除约 113 行的自定义序列化代码

**原因**:
1. `create_agent()` + middleware + checkpoint 的组合存在架构不匹配
2. Checkpoint 不是为数据持久化设计，而是为图执行状态快照设计
3. Middleware 不应该被序列化到 checkpoint 中

## 当前架构问题

### 问题代码位置

**文件**: `btliu/cli/cli.py`

| 行号 | 内容 | 行数 |
|------|------|------|
| 173-210 | WORKAROUND 注释说明 | 38 行 |
| 214-221 | `_UNPICKLABLE_TYPES` 定义 | 8 行 |
| 223-267 | `_filter_unpicklable()` 函数 | 45 行 |
| 269-285 | `_CustomJsonPlusSerializer` 类 | 17 行 |
| 165-171 | 数据库连接池设置 | 7 行 |
| 304-311 | 数据库表创建 | 8 行 |

**总计**: 约 113 行复杂的变通代码

### 架构不匹配图示

```
当前架构（有问题）:

create_agent()
    ↓
middleware 混入 state
    ↓
checkpoint 尝试序列化整个 state
    ↓
middleware 包含无法 pickle 的组件
    ↓
需要自定义序列化器变通 💀
```

## 正确的架构理解

### Checkpoint 的真正用途

根据 [LangGraph 官方文档](https://docs.langchain.com/oss/python/langgraph/persistence)：

> "When you compile a graph with a checkpointer, the checkpointer saves a checkpoint of the graph state at every super-step."

**Checkpoint 的用途**:
- 保存图执行状态的快照
- 支持从某个节点恢复执行
- 支持时间旅行和故障容错
- **不是** 为数据持久化设计的

### Middleware 的特性

Middleware 是运行时对象，包含：
- LLM 实例（带有 `threading._lock`）
- 文件句柄和连接
- Lambda 函数
- 复杂的工具引用

**这些组件不应该、也不能被序列化。**

### Store vs Checkpoint

| 组件 | 用途 | 数据类型 |
|------|------|----------|
| **Checkpoint** | 图执行状态快照 | 节点输出、通道值 |
| **Store** | 跨线程/会话持久化 | 用户数据、会话记忆 |
| **Middleware** | 运行时逻辑处理 | **不应放入 checkpoint** |

## 决策理由

### 理由 1：架构不匹配

`create_agent()` 的设计会导致 middleware 被混入 graph state，这违反了 checkpoint 的设计原则。

```python
# create_agent 的问题
state = {
    "messages": [...],
    "middleware": SummarizationMiddleware(llm),  # 💥 不应该在这里！
}
```

### 理由 2：变通代码过于复杂

需要：
- 自定义序列化器
- 递归过滤函数
- 循环引用检测
- 备用 pickle 机制

**这不是正确的解决方案，而是权宜之计。**

### 理由 3：替代方案更清晰

```python
# 正确的架构
StateGraph
    ↓
简单的 state（只包含消息）
    ↓
middleware 在外部处理
    ↓
checkpoint 只序列化纯粹的图状态
    ↓
store 处理用户数据和会话记忆
```

### 理由 4：功能影响有限

移除 checkpoint 后的影响：
- ❌ 无法从某个节点恢复执行
- ❌ `/save` 和 `/restore` 命令失去意义
- ✅ 用户数据和会话记忆仍然可以通过 Store 保留
- ✅ 对话历史可以通过其他方式实现（如日志）

## 实施计划

### 需要删除的代码

**1. `btliu/cli/cli.py`**

```python
# 删除以下行：
# Lines 173-210: WORKAROUND 注释
# Lines 214-221: _UNPICKLABLE_TYPES
# Lines 223-267: _filter_unpicklable 函数
# Lines 269-285: _CustomJsonPlusSerializer 类
# Lines 165-171: 数据库连接池（checkpointer 相关）
# Lines 304-311: 数据库表创建（checkpointer 相关）
```

**2. `btliu/agents/factory.py`**

```python
# 移除 checkpointer 参数
# 更新 create_agent() 调用
```

**3. `config.json`**

```json
// 移除或注释：
"checkpoint_db_uri": "postgresql://..."
```

### 需要保留的代码

- **Store 相关代码**: 用于用户数据和会话记忆
- **Middleware 逻辑**: 在运行时处理，不序列化

### 替代方案

如果将来需要状态持久化：

1. **使用 MemorySaver**
   ```python
   from langgraph.checkpoint.memory import MemorySaver
   checkpointer = MemorySaver()
   ```

2. **使用自定义 StateGraph**
   ```python
   from langgraph.graph import StateGraph
   graph = StateGraph(AgentState)
   # middleware 在外部处理，不放入 state
   ```

3. **应用层持久化**
   - 记录对话历史到数据库
   - 实现自己的会话管理
   - 不依赖 checkpoint 的状态快照

## 参考资料

### 官方文档

- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph Store](https://docs.langchain.com/oss/python/langgraph/persistence#cross-thread-data)
- [LangGraph v1 Release](https://docs.langchain.com/oss/python/releases/langgraph-v1)

### 相关 GitHub Issues

- [Type is not msgpack serializable: AIMessage #5248](https://github.com/langchain-ai/langgraph/issues/5248)
- [Object of type HumanMessage is not JSON serializable #5511](https://github.com/langchain-ai/langgraph/issues/5511)
- [Checkpoint serialization issues](https://github.com/langchain-ai/langgraph/issues/search?q=checkpoint+serialization&type=issues)

## 决策总结

| 方面 | 内容 |
|------|------|
| **决策** | 移除 Checkpoint 功能 |
| **删除代码** | 约 113 行 |
| **简化效果** | 消除自定义序列化复杂性 |
| **功能损失** | 图状态快照、节点恢复 |
| **保留功能** | Store（用户数据、会话记忆）|
| **未来方案** | StateGraph 或 MemorySaver |

## 经验教训

1. **理解工具的设计目的**: Checkpoint 不是为数据持久化设计的
2. **架构匹配很重要**: `create_agent()` 不适合需要 middleware 持久化的场景
3. **变通不是解决方案**: 自定义序列化器过于复杂，应该重新审视架构
4. **关注官方文档**: LangGraph 官方文档明确说明了 Checkpoint vs Store 的区别
