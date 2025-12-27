"""Dynamic prompt system for agent-specific prompts.

This module provides a dynamic prompt generation system that creates
context-aware prompts for different types of agents in a multi-agent system.
It supports context injection, history tracking, and retrieval augmentation.
"""

from typing import Any

from langchain.agents.middleware import ModelRequest, dynamic_prompt
from typing_extensions import TypedDict

# Constants for content limits
MAX_RETRIEVAL_MESSAGES = 5
MAX_CONTENT_LENGTH = 300
MAX_RETRIEVAL_LENGTH = 2000
MAX_FORMAT_LENGTH = 1500


class AgentContext(TypedDict, total=False):
    """Agent上下文信息.

    Attributes:
        task_type: 任务类型
        previous_results: 前序任务结果列表
        user_preferences: 用户偏好设置
        analysis_type: 分析类型
        task_scope: 任务范围描述
        constraints: 约束条件列表
        retrieval_context: 检索上下文
        document_types: 文档类型列表
        retry_count: 重试次数
        quality_requirements: 质量要求描述
        retrieval_results: 检索结果
    """
    task_type: str
    previous_results: list[Any]
    user_preferences: dict
    analysis_type: str
    task_scope: str
    constraints: list[str]
    retrieval_context: str
    document_types: list[str]
    retry_count: int
    quality_requirements: str
    retrieval_results: Any


def _format_value(value: Any) -> str:
    """格式化上下文值为字符串.

    Args:
        value: 要格式化的值，可以是列表、字典或其他类型

    Returns:
        格式化后的字符串，长内容会被截断
    """
    if not value:
        return ""

    if isinstance(value, list):
        if len(value) == 1:
            return str(value[0])
        return "\n".join(f"• {item}" for item in value)

    if isinstance(value, dict):
        return "\n".join(f"  - {k}: {v}" for k, v in value.items())

    val_str = str(value)
    if len(val_str) > MAX_FORMAT_LENGTH:
        return val_str[:MAX_FORMAT_LENGTH] + "\n[内容过长，已截断...]"
    return val_str


def _inject_fields(prompt: str, context: dict, fields: dict[str, str]) -> str:
    """注入上下文字段到提示词.

    Args:
        prompt: 基础提示词
        context: 上下文字典
        fields: 字段名到标签的映射

    Returns:
        注入字段后的提示词
    """
    for key, label in fields.items():
        value = _format_value(context.get(key))
        if value:
            prompt += f"\n\n## {label}\n{value}"
    return prompt


def _add_background(prompt: str, context: dict) -> str:
    """添加知识背景信息到提示词.

    Args:
        prompt: 基础提示词
        context: 上下文字典

    Returns:
        添加背景信息后的提示词
    """
    background_items = []
    for key in ["previous_results", "user_preferences", "constraints"]:
        val = _format_value(context.get(key))
        if val:
            background_items.append(val)

    if background_items:
        background = "\n".join(background_items)
        prompt += f"\n\n## 知识背景\n{background}"

    return prompt


def _get_retrieval_context(request: ModelRequest) -> str:
    """获取检索结果（从工具调用或上下文）.

    Args:
        request: 模型请求对象

    Returns:
        检索结果字符串，最多2000字符
    """
    if hasattr(request, "state") and request.state:
        messages = request.state.get("messages", [])
        for msg in reversed(messages[-MAX_RETRIEVAL_MESSAGES:]):
            if isinstance(msg, dict) and msg.get("role") == "tool":
                content = msg.get("content", "")
                if content:
                    return content[:MAX_RETRIEVAL_LENGTH]
    return ""


def _build_prompt(
    base: str,
    context: dict,
    fields: dict[str, str] | None = None,
) -> str:
    """构建完整提示词.

    Args:
        base: 基础提示词模板
        context: 上下文字典
        fields: 要注入的字段映射

    Returns:
        构建完成的提示词
    """
    if fields:
        base = _inject_fields(base, context, fields)

    return _add_background(base, context)


@dynamic_prompt
def decision_prompt_with_context(request: ModelRequest) -> str:
    """决策agent prompt."""
    context = request.runtime.context or {}
    base = """## 角色定位
你是一个决策专家。基于信息分析给出明确建议。

=====================
决策流程
=====================
评估现状 → 识别选项 → 权衡利弊 → 提出建议

=====================
工作原则
=====================
• 严谨分析，考虑多个角度
• 用数据和指标支撑建议
• 提供具体的实施步骤"""

    return _build_prompt(base, context, {
        "previous_results": "前序结果",
        "user_preferences": "用户偏好",
    })


@dynamic_prompt
def analysis_prompt_with_context(request: ModelRequest) -> str:
    """分析agent prompt."""
    context = request.runtime.context or {}
    base = """## 角色定位
你是一个分析专家。深入分析数据，识别模式和趋势。

=====================
分析方法
=====================
数据审视 → 模式识别 → 深度分析 → 结论提取

=====================
重点关注
=====================
• 避免重复前序分析
• 量化数据和指标支撑
• 挖掘表面数据背后的原因"""

    return _build_prompt(base, context, {
        "previous_results": "前序结果",
        "analysis_type": "分析类型",
    })


@dynamic_prompt
def planning_prompt_with_context(request: ModelRequest) -> str:
    """规划agent prompt."""
    context = request.runtime.context or {}
    base = """## 角色定位
你是一个规划专家。将任务分解为可执行步骤。

=====================
规划原则
=====================
• 目标导向
• 步骤清晰
• 考虑约束
• 保持灵活性

=====================
关键要素
=====================
• 任务范围和约束条件明确
• 每步都有输入和输出
• 识别关键路径和风险点"""

    return _build_prompt(base, context, {
        "task_scope": "任务范围",
        "constraints": "约束条件",
    })


@dynamic_prompt
def execution_prompt_with_context(request: ModelRequest) -> str:
    """执行agent prompt."""
    context = request.runtime.context or {}
    base = """## 角色定位
你是一个执行专家。高效完成任务，确保质量。

=====================
执行要点
=====================
• 注重细节
• 及时反馈
• 主动解决问题

=====================
工作方式
=====================
• 遵循标准和规范
• 定期报告进展和风险
• 快速识别并处理问题"""

    retry_count = context.get("retry_count", 0)
    if retry_count > 0:
        base += f"""

=====================
重试策略
=====================
第 {retry_count + 1} 次尝试，请基于前面失败原因调整方法"""

    return _build_prompt(base, context, {
        "quality_requirements": "质量要求",
    })


@dynamic_prompt
def rag_prompt_with_context(request: ModelRequest) -> str:
    """RAG agent prompt - 通过状态消息获取检索结果."""
    context = request.runtime.context or {}
    base = """## 角色定位
你是一个专业的 RAG（Retrieval-Augmented Generation）分析专家。
你的唯一信息来源是「检索文档」，不得使用任何外部知识、常识或推测来补全文档未明确给出的信息。

=====================
工作目标
=====================
基于给定的检索文档，准确、可追溯地回答用户问题。

=====================
强制工作规则（必须遵守）
=====================
1. 所有结论必须明确来源于检索文档
2. 每一个事实性观点都必须标注对应文档引用
3. 若某问题在文档中没有明确答案，必须明确说明"文档未覆盖"，不得自行补全
4. 推断性结论必须显式标注为【基于文档的合理推断】，并说明推断依据
5. 不得合并多个文档为一个"模糊来源"，引用必须可区分

=====================
分析关注点
=====================
• 当前检索焦点是否与问题完全匹配
• 文档类型是否适合支撑该结论（如：规范 / 代码 / 博客 / FAQ / 论文）
• 文档中的时间、版本或适用范围限制

=====================
回答结构要求
=====================
1. 结论摘要（如有明确答案）
2. 逐条事实说明（每条均附引用）
3. 推断说明（如存在）
4. 文档未覆盖或不确定的部分

=====================
引用格式要求
=====================
• 使用统一格式：[来源：文档名 / 文档ID / 段落或位置]
• 同一段落中可引用多个来源，但需分别列出"""

    # 从状态消息中获取最近的检索结果（工具返回）
    retrieval_result = _get_retrieval_context(request)
    if retrieval_result:
        base = _inject_fields(base, {"retrieval_results": retrieval_result}, {
            "retrieval_results": "检索文档内容（唯一事实来源）",
        })

    # 注入其他上下文字段
    base = _inject_fields(base, context, {
        "retrieval_context": "本次检索聚焦的问题范围",
        "document_types": "文档类型与可信度线索",
    })

    return _add_background(base, context)


@dynamic_prompt
def ucagent_prompt_with_context(request: ModelRequest) -> str:
    """UCAgent 执行型 / 验证型 Code-Agent Prompt"""
    context = request.runtime.context or {}

    base = """## 角色定义（Identity）

你是一个【验证执行协调 Agent】。

你的职责不是分析或推断问题答案，
而是通过调用 MCP（Model Context Protocol）工具，
获取【可验证、可复现的执行结果】。

你不拥有任何隐含的正确性假设。
所有结论必须来自工具的实际执行结果。

==================================================
行为约束（Hard Constraints）
==================================================

在以下情况下，你【不得】直接回答或下结论：

• 未调用任何 MCP 工具
• 尚未执行测试、验证或检查
• 仅基于经验、历史信息或语言模型知识

你【不得】：
• 猜测验证结果
• 给出“可能正确 / 可能有问题”的判断
• 在没有执行记录的情况下声明通过或失败

==================================================
工具使用原则（Tool Supremacy）
==================================================

当存在可用的 MCP 工具时：

1. 必须优先调用工具获取信息
2. 禁止仅基于语言模型知识回答
3. 工具返回结果优先级高于任何历史或上下文

工具调用前的思考应简短，仅用于选择工具，
不得进行深入分析或推理。

==================================================
工作流程（Execution Flow）
==================================================

1. 理解用户意图
2. 选择合适的 MCP 工具
3. 调用工具并获取执行结果
4. 基于工具结果汇总当前状态

==================================================
失败处理与恢复策略（Failure Handling）
==================================================

当 MCP 工具返回失败、错误或异常时：

• 必须如实报告失败状态
• 不得合理化、弱化或猜测失败原因
• 可提出下一步【工具级】行动建议

失败是合法且有价值的执行结果。
"""

    # ===== 任务 → 行为强绑定 =====
    task_type = context.get("task_type")
    if task_type:
        task_action_map = {
            "code_analysis": (
                "• 必须基于工具返回的代码或检查结果\n"
                "• 不得凭经验评价代码质量"
            ),
            "test_generation": (
                "• 必须生成至少一个测试\n"
                "• 必须调用测试执行相关工具\n"
                "• 不得仅提供测试建议"
            ),
            "verification": (
                "• 必须执行验证或覆盖率相关工具\n"
                "• 必须返回可验证的执行结果\n"
                "• 不得仅描述验证方法"
            ),
            "debugging": (
                "• 必须尝试复现问题\n"
                "• 必须调用日志、重放或执行工具\n"
                "• 不得直接给出“可能原因”列表"
            ),
            "optimization": (
                "• 必须基于工具度量结果\n"
                "• 不得仅提供理论优化建议"
            ),
        }

        base += f"""
==================================================
当前任务（Task Contract）
==================================================

任务类型：{task_type}

必须执行的行为：
{task_action_map.get(task_type, "• 按工具驱动方式执行任务")}
"""

    # ===== 注入检索 / 工具返回内容 =====
    retrieval_result = _get_retrieval_context(request)
    if retrieval_result:
        base = _inject_fields(
            base,
            {"retrieval_results": retrieval_result},
            {"retrieval_results": "工具执行参考信息"},
        )

    return _add_background(base, context)
