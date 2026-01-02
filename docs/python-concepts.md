# Python 知识点教学

本文档记录在解决 `RuntimeError: dictionary changed size during iteration` 问题过程中涉及的 Python 核心概念，由浅入深，适合新手学习。

---

## 第一课：字典迭代时不能修改

### 现象

```python
# ❌ 错误示例
d = {"a": 1, "b": 2, "c": 3}
for key, value in d.items():
    if value == 2:
        del d[key]  # 💥 RuntimeError: dictionary changed size during iteration
```

### 为什么会报错？

Python 的字典迭代器会跟踪字典的"大小"（元素数量）。当你迭代时删除或添加元素，迭代器会"迷路"——它不知道下一步该去哪里。

**可视化**：

```
迭代器状态：
    ↓ 指向 "a"
    ↓ 指向 "b"
    ↓ 指向 "c"
    ↓ 完成

如果你删除了 "b"：
    ↓ 指向 "a"
    ↓ 指向 "b"  (已经被删除！) 💥 迷路
```

### 正确的做法

**方法 1：先收集要删除的键**

```python
d = {"a": 1, "b": 2, "c": 3}
keys_to_delete = [k for k, v in d.items() if v == 2]
for key in keys_to_delete:
    del d[key]
```

**方法 2：创建新字典（字典推导式）**

```python
d = {"a": 1, "b": 2, "c": 3}
new_d = {k: v for k, v in d.items() if v != 2}  # ✅ 安全
```

### 练习

```python
# 练习：删除所有值为偶数的项
d = {"a": 1, "b": 2, "c": 3, "d": 4}
# 写出你的代码...

# 答案：
result = {k: v for k, v in d.items() if v % 2 != 0}
print(result)  # {"a": 1, "c": 3}
```

---

## 第二课：递归函数（自己调用自己）

### 基本概念

递归 = 函数调用自己 + 基线条件（停止条件）

```python
def countdown(n):
    if n <= 0:      # 基线条件：停止递归
        print("完成!")
        return
    print(n)
    countdown(n - 1)  # 递归调用

countdown(3)
# 输出: 3, 2, 1, 完成!
```

### 可视化递归过程

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

factorial(3)
# ↓
# factorial(3)
#   → 3 * factorial(2)
#       → 2 * factorial(1)
#           → 1 (基线条件)
#       → 2 * 1 = 2
#   → 3 * 2 = 6
# 结果: 6
```

### 实际应用：处理嵌套结构

```python
# 处理嵌套字典
def process(obj):
    if isinstance(obj, dict):
        return {k: process(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [process(item) for item in obj]
    return obj  # 基线条件

data = {"a": [1, 2], "b": {"c": 3}}
result = process(data)  # 返回相同的结构
```

### 练习

```python
# 练习：计算列表中所有数字的和（包括嵌套列表）
nested = [1, [2, 3], [4, [5]]]

def sum_nested(lst):
    total = 0
    for item in lst:
        if isinstance(item, list):
            total += sum_nested(item)  # 递归
        else:
            total += item
    return total

print(sum_nested(nested))  # 15
```

---

## 第三课：循环引用与无限递归

### 什么是循环引用？

对象引用自己，形成环状结构。

```python
# 对象引用自己
obj = {}
obj["self"] = obj  # obj 包含自己

# 可视化
┌─────────────────┐
│      obj        │
├─────────────────┤
│ "self": ────────┼──→ 回到 obj（循环！）
│ "data": 123     │
└─────────────────┘
```

### 为什么会导致无限递归？

```python
# 没有循环引用检测的递归
def process(obj):
    if isinstance(obj, dict):
        return {k: process(v) for k, v in obj.items()}
    return obj

obj = {}
obj["self"] = obj

process(obj)
# ↓
# 处理 obj，遇到 "self" 键
# ↓
# process(obj) 再次被调用（因为 obj["self"] 就是 obj）
# ↓
# 又处理 obj，又遇到 "self" 键
# ↓
# 无限循环... 💥 RecursionError: maximum recursion depth exceeded
```

### 解决方案：使用 `id()` 追踪

```python
def process(obj, _visited=None):
    if _visited is None:
        _visited = set()

    obj_id = id(obj)
    if obj_id in _visited:
        return obj  # 已访问过，停止递归

    _visited.add(obj_id)  # 标记为已访问

    if isinstance(obj, dict):
        return {k: process(v, _visited) for k, v in obj.items()}
    return obj
```

### 可视化追踪过程

```python
obj = {}
obj["self"] = obj

process(obj, _visited=set())
# ↓
# id(obj) = 12345，不在 _visited 中
# _visited = {12345}
# ↓
# 处理 obj["self"]，它又是 obj
# ↓
# id(obj) = 12345，已经在 _visited 中！
# 返回 obj，停止递归 ✅
```

### 练习

```python
# 练习：检测下面的结构是否有循环引用
a = {}
b = {}
a["b"] = b
b["a"] = a

# 如何检测？
def has_cycle(obj, _visited=None):
    if _visited is None:
        _visited = set()

    obj_id = id(obj)
    if obj_id in _visited:
        return True  # 检测到循环！

    if isinstance(obj, dict):
        _visited.add(obj_id)
        for v in obj.values():
            if isinstance(v, dict):
                if has_cycle(v, _visited):
                    return True
    return False

print(has_cycle(a))  # True
```

---

## 第四课：`id()` - 对象的"身份证号"

### `id()` 是什么？

```python
a = "hello"
b = "hello"
c = a

print(id(a))  # 140234567890000
print(id(b))  # 140234567890000（Python 可能复用字符串）
print(id(c))  # 140234567890000（和 a 相同）

print(a is c)  # True（id 相同）
print(a is b)  # 可能是 True 或 False
```

**`id(obj)`** 返回对象的唯一标识符（内存地址），每个对象有唯一 id。

### 用于检测循环引用

```python
# 两个不同的对象，即使内容相同，id 也不同
a = {"x": 1}
b = {"x": 1}
print(id(a) == id(b))  # False

# 同一个对象，id 相同
obj = {}
obj["self"] = obj
print(id(obj) == id(obj["self"]))  # True（循环引用！）
```

### `id()` vs `==` vs `is`

```python
a = [1, 2, 3]
b = [1, 2, 3]
c = a

print(a == b)  # True（内容相同）
print(a is b)  # False（不同对象）
print(a is c)  # True（同一个对象）
print(id(a) == id(c))  # True
```

| 操作符 | 检查什么 | 示例 |
|--------|----------|------|
| `==` | 值是否相等 | `a == b` |
| `is` | 是否同一个对象 | `a is b` |
| `id()` | 对象的唯一标识 | `id(a)` |

---

## 第五课：可变与不可变对象

### 核心概念

```python
# 不可变对象（不能修改内容）
x = (1, 2, 3)  # 元组
s = "hello"    # 字符串
i = 42         # 数字

# 可变对象（可以修改内容）
lst = [1, 2, 3]  # 列表
d = {"a": 1}      # 字典
st = {1, 2}       # 集合
```

### 为什么只有可变对象能有循环引用？

```python
# 不可变的元组不能修改自身
t = (1, 2, 3)
# t[0] = t  # ❌ TypeError: 'tuple' object does not support item assignment

# 可变的列表可以修改
lst = [1, 2, 3]
lst[0] = lst  # ✅ 可以创建循环引用
```

### 实际影响

```python
# 只追踪可变容器
if isinstance(obj, (dict, list)):
    _visited.add(obj_id)
# 元组不需要追踪，因为它们不能有循环引用
```

---

## 第六课：默认参数的陷阱

### 问题代码

```python
# ❌ 常见错误
def foo(items=[]):  # 默认参数只创建一次！
    items.append(1)
    return items

print(foo())  # [1]
print(foo())  # [1, 1] ⚠️ 不是 [1]！
```

### 为什么？

```python
# Python 的默认参数在函数定义时创建，不是每次调用时创建
def foo(items=[]):  # 这个 [] 对象只创建一次
    ...

# 等价于
_default_list = []
def foo(items=_default_list):
    ...
```

### 可视化问题

```
第一次调用 foo():
  items → [] (默认对象)
  append(1) → [1]

第二次调用 foo():
  items → 同一个 [] 对象！
  已经是 [1] 了
  append(1) → [1, 1] 💥
```

### 正确做法

```python
# ✅ 使用 None 作为默认值
def foo(items=None):
    if items is None:
        items = []  # 每次调用都创建新的
    items.append(1)
    return items

print(foo())  # [1]
print(foo())  # [1] ✅
```

### 实际应用

```python
# 我们的修复中使用的模式
def _filter_unpicklable(obj, _visited=None):
    if _visited is None:
        _visited = set()  # 每次顶层调用都创建新的
    # 递归调用时传入已有的 _visited
    return {k: _filter_unpicklable(v, _visited) for k, v in obj.items()}
```

---

## 总结：知识地图

``┌─────────────────────────────────────────────────────────┐
│                    Python 递归编程                        │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  基础概念                                                  │
│  ├── 字典迭代: 迭代时不能修改                            │
│  ├── 递归函数: 函数调用自己                              │
│  └── 循环引用: 对象引用自己                              │
│                                                           │
│  工具方法                                                  │
│  ├── id(): 对象的唯一标识                                │
│  ├── is(): 身份比较（同一对象）                          │
│  └── ==(): 值比较                                        │
│                                                           │
│  类型系统                                                  │
│  ├── 可变对象: list, dict, set（可以修改）                │
│  └── 不可变对象: tuple, str, int（不能修改）              │
│                                                           │
│  最佳实践                                                  │
│  ├── 默认参数: 使用 None 避免陷阱                        │
│  ├── 递归追踪: 使用 set() 记录已访问对象                 │
│  └── 循环引用: 用 id() 检测                               │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

## 练习建议

1. **理解递归**: 写一个递归函数计算阶乘 `n!`
2. **理解 id()**: 创建几个变量，用 `id()` 和 `is` 比较它们
3. **理解默认参数**: 写一个带默认列表参数的函数，观察行为
4. **理解循环引用**: 创建两个相互引用的字典，打印它们的 `id()`
