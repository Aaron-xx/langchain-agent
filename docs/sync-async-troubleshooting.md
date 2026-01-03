# Python 同步/异步调试指南：LangGraph 实战案例

## 问题背景

在开发 LangGraph Agent 时，遇到了一个经典错误：
```
NotImplementedError: StructuredTool does not support sync invocation.
```

这个错误揭示了 Python 异步编程中一个核心概念：**同步和异步代码不能随意混用**。

---

## 核心概念：同步 vs 异步

### 什么是同步（Sync）？

同步代码就像排队买票：
```python
def buy_ticket():
    result = purchase()  # 必须等这里完成，才能继续
    return result
```
- 代码按顺序执行
- 每个操作必须等前一个完成
- 期间程序"阻塞"（block）

### 什么是异步（Async）？

异步代码就像在餐厅点餐：
```python
async def order_food():
    result = await cook()  # 等待期间可以做其他事
    return result
```
- 使用 `async def` 定义
- 使用 `await` 等待异步操作
- 等待期间可以处理其他任务

---

## 问题分析：为什么会报错？

### 错误场景 1：异步工具 + 同步调用

```python
# ❌ 错误写法
@tool
async def recall_memory(...):  # 异步工具
    items = await store.asearch(...)

# 使用 agent.stream()  # 同步调用
for chunk in agent.stream(...):
    ...
```

**问题**：`agent.stream()` 是同步的，它试图同步调用异步工具 `recall_memory`，导致报错。

### 正确做法：保持一致

```python
# ✅ 正确写法一：都用同步
@tool
def recall_memory(...):  # 同步工具
    items = store.search(...)  # 同步方法

for chunk in agent.stream(...):  # 同步调用
    ...

# ✅ 正确写法二：都用异步
@tool
async def recall_memory(...):  # 异步工具
    items = await store.asearch(...)  # 异步方法

async for chunk in agent.astream(...):  # 异步调用
    ...
```

---

## 问题场景 2：中间件不匹配

### 错误代码

```python
# ❌ 错误写法
@wrap_model_call
async def dynamic_model_router(...):  # 异步中间件
    ...
    return await handler(request)

# 使用 agent.stream()  # 同步调用
for chunk in agent.stream(...):
```

**错误信息**：
```
NotImplementedError: Synchronous implementation of wrap_model_call is not available.
```

### 原因

`@wrap_model_call` 装饰器会根据函数定义自动决定：
- `def func()` → 提供同步实现
- `async def func()` → 只提供异步实现

当使用同步 `agent.stream()` 时，需要同步中间件。

### 修复

```python
# ✅ 同步中间件配合同步调用
@wrap_model_call
def dynamic_model_router(...):  # 同步中间件
    ...
    return handler(request)  # 没有 await

for chunk in agent.stream(...):  # 同步调用
```

---

## 问题场景 3：数据库连接池不匹配

### 尝试异步连接池

```python
# ❌ 问题代码
from psycopg_pool import AsyncConnectionPool

pool = AsyncConnectionPool(...)
checkpointer = PostgresSaver(pool)  # PostgresSaver 不支持异步池
```

**结果**：程序卡住（hang），因为 `AsyncConnectionPool.setup()` 需要异步执行。

### 解决方案：使用同步组件

```python
# ✅ 使用同步连接池
from psycopg_pool import ConnectionPool

pool = ConnectionPool(...)  # 同步池
checkpointer = PostgresSaver(pool)  # 同步 saver
store = PostgresStore(pool)  # 同步 store
```

---

## 实用规则：选择同步还是异步？

### 规则 1：保持一致性

整个调用链必须要么全同步，要么全异步：

```
应用入口 (同步/异步)
    ↓
Agent (stream/astream)
    ↓
中间件 (sync/async)
    ↓
工具 (sync/async)
    ↓
数据库 (sync/async)
```

### 规则 2：LangGraph API 对照表

| 组件 | 同步版本 | 异步版本 |
|------|---------|---------|
| Agent 调用 | `agent.stream()` | `async for chunk in agent.astream()` |
| Agent 状态 | `agent.get_state()` | `await agent.aget_state()` |
| Checkpoint | `PostgresSaver` | 使用异步池（实验性） |
| Store | `PostgresStore` | `AsyncPostgresStore` |
| 工具定义 | `@tool def func()` | `@tool async def func()` |
| Store 方法 | `store.put()`, `store.search()` | `await store.aput()`, `await store.asearch()` |

### 规则 3：何时选择异步？

选择异步的情况：
- 需要处理大量并发 I/O（如同时服务多个用户）
- 使用 Web 框架（FastAPI、Tornado 等）
- 明确需要异步性能优势

选择同步的情况：
- 简单脚本或 CLI 工具
- 开发和调试阶段（更容易理解）
- 不需要高并发

---

## 常见错误速查

| 错误信息 | 原因 | 解决方法 |
|---------|------|---------|
| `StructuredTool does not support sync invocation` | 异步工具被同步调用 | 把工具改为同步，或使用异步 API |
| `Synchronous implementation not available` | 异步中间件被同步调用 | 把中间件改为同步 |
| `'async_for' outside async function` | 在非异步函数中使用 `async for` | 改为 `for`，或把外层函数改为 `async def` |
| `'await' outside async function` | 在非异步函数中使用 `await` | 移除 `await`，或把外层函数改为 `async def` |
| 程序卡住不响应 | 同步组件等待异步操作 | 检查连接池、setup 方法是否需要异步 |
| `object is not iterable` | 用 `for` 遍历异步生成器 | 改为 `async for` |

---

## 本次修复清单

### 1. 工具改为同步

```python
# 修改前
@tool
async def save_memory(...):
    await store.aput(...)

@tool
async def recall_memory(...):
    await store.asearch(...)

# 修改后
@tool
def save_memory(...):
    store.put(...)

@tool
def recall_memory(...):
    store.search(...)
```

### 2. 中间件改为同步

```python
# 修改前
@wrap_model_call
async def dynamic_model_router(...):
    return await handler(request)

# 修改后
@wrap_model_call
def dynamic_model_router(...):
    return handler(request)
```

### 3. Agent 调用改为同步

```python
# 修改前
async for chunk in agent.astream(...):
    ...
snapshot = await agent.aget_state(...)

# 修改后
for chunk in agent.stream(...):
    ...
snapshot = agent.get_state(...)
```

### 4. 数据库组件改为同步

```python
# 修改前
pool = AsyncConnectionPool(...)
checkpointer = PostgresSaver(pool)
await checkpointer.setup()

# 修改后
pool = ConnectionPool(...)
checkpointer = PostgresSaver(pool)
checkpointer.setup()
```

### 5. 其他修复

- EOF 处理：检测 `readline()` 返回空时退出循环
- Payload 结构：`user_id` 和 `user_info` 作为 state 传递，不是放在消息里
- 移除未定义变量 `hard_keywords`

---

## 学习要点

1. **一致性是关键**：同步/异步混用是常见错误源头
2. **读懂错误信息**：`NotImplementedError` 往往意味着方法不匹配
3. **API 对照表**：记住 LangGraph 的同步/异步 API 对应关系
4. **调试策略**：从错误发生点向上追溯，检查整个调用链的一致性
5. **渐进开发**：先用同步 API 开发和调试，必要时再迁移到异步

---

## 参考资源

- [Python asyncio 官方文档](https://docs.python.org/3/library/asyncio.html)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- PEP 492 -- Coroutines with async and await syntax
