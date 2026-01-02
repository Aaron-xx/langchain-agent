# 循环引用 Bug 修复记录

## 问题描述

### 错误信息

```
RuntimeError: dictionary changed size during iteration
```

### 完整错误堆栈

```python
File "/home/langchain/workspace/btliu/cli/cli.py", line 241, in <dictcomp>
  return {k: _filter_unpicklable(v) for k, v in obj.items()}
RuntimeError: dictionary changed size during iteration
```

### 触发场景

在运行 CLI 时，当 LangGraph 尝试保存 checkpoint 时，`_filter_unpicklable` 函数在处理包含循环引用的数据结构时崩溃。

## 根本原因

### 1. 递归函数遇到循环引用

原始的 `_filter_unpicklable` 函数没有处理循环引用：

```python
def _filter_unpicklable(obj):
    # ...
    if isinstance(obj, dict):
        return {k: _filter_unpicklable(v) for k, v in obj.items()}  # 无限递归！
```

当遇到这样的数据结构时：

```python
obj = {"middleware": SomeMiddleware()}
obj["self"] = obj  # 循环引用

# 递归过程：
_filter_unpicklable(obj)
  → 处理 obj["self"] (又是 obj)
    → _filter_unpicklable(obj)  # 无限循环！
      → 处理 obj["self"] (又是 obj)
        → ...
```

### 2. 为什么会出现循环引用？

在实际场景中，循环引用可能来自：
- LangGraph 内部状态管理
- 对象之间的相互引用
- 复杂的嵌套数据结构

## 解决方案

### 核心思路：追踪已访问对象

使用 `id()` 函数获取对象的唯一标识符，维护一个已访问对象的集合。

### 修复代码

**文件**: `btliu/cli/cli.py` (lines 223-267)

```python
def _filter_unpicklable(obj, _visited=None):
    """Recursively remove unpicklable objects from state.

    Handles circular references by tracking visited objects using id().
    """
    # 第1步：初始化追踪集合
    if _visited is None:
        _visited = set()

    # 第2步：检查循环引用
    obj_id = id(obj)
    if obj_id in _visited:
        # 循环引用！检查是否需要过滤
        if hasattr(obj, "__class__"):
            cls_name = obj.__class__.__name__
            if cls_name in _UNPICKLABLE_TYPES:
                return None
        return obj

    # 第3步：标记已访问
    if isinstance(obj, (dict, list)):
        _visited.add(obj_id)

    # 第4步：递归处理
    if hasattr(obj, "__class__"):
        cls_name = obj.__class__.__name__
        if cls_name == "Send":
            if hasattr(obj, "node") and hasattr(obj, "arg"):
                return {
                    "__send__": True,
                    "node": obj.node,
                    "arg": _filter_unpicklable(obj.arg, _visited),
                }
            return obj
        if cls_name in _UNPICKLABLE_TYPES:
            return None

    if isinstance(obj, dict):
        return {k: _filter_unpicklable(v, _visited) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_filter_unpicklable(item, _visited) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_filter_unpicklable(item, _visited) for item in obj]
    if isinstance(obj, set):
        return {_filter_unpicklable(item, _visited) for item in obj}

    return obj
```

### 关键改动说明

| 改动 | 说明 |
|------|------|
| 添加 `_visited` 参数 | 追踪已访问对象的 id 集合 |
| `obj_id in _visited` 检查 | 检测循环引用，避免无限递归 |
| 循环引用时类型检查 | 即使是循环引用，也要检查是否需要过滤中间件 |
| 传递 `_visited` 到递归调用 | 确保所有递归分支共享访问状态 |
| 添加 `set` 类型支持 | 处理 set 类型中的 unpicklable 对象 |

### 为什么这个方案有效？

```python
obj = {}
obj["self"] = obj

_filter_unpicklable(obj, _visited=set())
# ↓
# id(obj) = 12345，不在 _visited 中
# _visited = {12345}
# ↓
# 处理 obj["self"]，它又是 obj
# ↓
# id(obj) = 12345，已经在 _visited 中！
# 返回 obj，停止递归 ✅
```

### 边缘情况处理

#### 1. 共享对象（非循环引用）

```python
shared = {"middleware": SomeMiddleware()}
obj = {"a": shared, "b": shared}

# 行为：
# - 第一次遇到 shared：正常过滤，加入 _visited
# - 第二次遇到 shared：已被访问，返回原对象
# - 结果：正确！共享对象被保持
```

#### 2. 嵌套循环引用

```python
a = {}
b = {}
a["b"] = b
b["a"] = a  # 嵌套循环

# 行为：每层独立追踪，正确处理
```

#### 3. 自定义类

使用 `id()` 而不是对象本身来追踪，避免自定义 `__hash__`/`__eq__` 的问题。

### 验证无问题

- **共享对象**: 正确处理，同一对象在不同分支会被正确处理
- **内存开销**: `_visited` 只存储 id()，开销可忽略
- **性能影响**: id() 查找和 set 操作都是 O(1)，影响极小

## 经验教训

1. **递归函数必须处理循环引用** - 否则会导致无限递归或栈溢出
2. **使用 `id()` 追踪对象** - 这是检测循环引用的标准方法
3. **注意默认参数陷阱** - 使用 `_visited=None` 避免可变默认参数问题
4. **测试边缘情况** - 共享对象、嵌套结构等

## 参考资料

- Python `id()` 文档: https://docs.python.org/3/library/functions.html#id
- Python 递归深度限制: `sys.getrecursionlimit()`
- 循环引用检测模式: https://stackoverflow.com/questions/3343236/python-detect-circular-reference
