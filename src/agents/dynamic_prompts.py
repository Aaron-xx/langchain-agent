"""Dynamic prompt system with agent-specific prompts"""
from typing import Optional, List, Any
from typing_extensions import TypedDict
from langchain.agents.middleware import dynamic_prompt, ModelRequest

class AgentContext(TypedDict, total=False):
    """Agent上下文信息"""
    task_type: str
    previous_results: List[Any]
    user_preferences: dict
    analysis_type: str
    task_scope: str
    constraints: List[str]
    retrieval_context: str
    document_types: List[str]
    retry_count: int
    quality_requirements: str
    retrieval_results: Any

# 辅助函数：安全获取上下文值
def _safe_get_context(context: dict, key: str, default: Any = None) -> Any:
    """安全获取上下文值"""
    if not context:
        return default
    value = context.get(key, default)
    return value if value is not None else default

def _get_last_result_safely(context: dict) -> Optional[str]:
    """安全获取最后一个结果"""
    results = context.get("previous_results", [])
    if isinstance(results, list) and len(results) > 0:
        return str(results[-1])
    return None

@dynamic_prompt
def decision_prompt_with_context(request: ModelRequest) -> str:
    """决策agent prompt"""
    base_prompt = """你是一个决策专家。基于信息分析给出明确建议。

    决策流程：评估现状→识别选项→权衡利弊→提出建议

    上下文处理：
    - 基于前序结果深化分析
    - 考虑用户偏好和约束"""

    context = request.runtime.context or {}

    # 安全获取前序结果
    last_result = _get_last_result_safely(context)
    if last_result:
        base_prompt += f"\n前序结果：{last_result}"

    # 安全获取用户偏好
    user_prefs = _safe_get_context(context, "user_preferences")
    if user_prefs:
        base_prompt += f"\n用户偏好：{user_prefs}"

    return base_prompt

@dynamic_prompt
def analysis_prompt_with_context(request: ModelRequest) -> str:
    """分析agent prompt"""
    base_prompt = """你是一个分析专家。深入分析数据，识别模式和趋势。

    分析方法：数据审视→模式识别→深度分析→结论提取

    重点关注：
    - 避免重复前序分析
    - 量化数据和指标支撑"""

    context = request.runtime.context or {}

    # 安全获取前序结果（统一使用"前序结果"）
    last_result = _get_last_result_safely(context)
    if last_result:
        base_prompt += f"\n前序结果：{last_result}"

    # 安全获取分析类型
    analysis_type = _safe_get_context(context, "analysis_type")
    if analysis_type:
        base_prompt += f"\n分析类型：{analysis_type}"

    return base_prompt

@dynamic_prompt
def planning_prompt_with_context(request: ModelRequest) -> str:
    """规划agent prompt"""
    base_prompt = """你是一个规划专家。将任务分解为可执行步骤。

    规划原则：目标导向、步骤清晰、考虑约束、保持灵活性

    关键要素：
    - 任务范围和约束条件
    - 资源需求和里程碑"""

    context = request.runtime.context or {}

    # 安全获取任务范围
    task_scope = _safe_get_context(context, "task_scope")
    if task_scope:
        base_prompt += f"\n任务范围：{task_scope}"

    # 安全获取约束条件（处理列表类型）
    constraints = _safe_get_context(context, "constraints", [])
    if constraints:
        if isinstance(constraints, list):
            base_prompt += f"\n约束：{', '.join(str(c) for c in constraints)}"
        else:
            base_prompt += f"\n约束：{constraints}"

    return base_prompt

@dynamic_prompt
def execution_prompt_with_context(request: ModelRequest) -> str:
    """执行agent prompt"""
    base_prompt = """你是一个执行专家。高效完成任务，确保质量。

    执行要点：注重细节、及时反馈、主动解决问题

    注意事项：
    - 根据重试次数调整策略
    - 满足质量要求"""

    context = request.runtime.context or {}

    # 安全获取重试次数
    retry_count = _safe_get_context(context, "retry_count", 0)
    if isinstance(retry_count, int) and retry_count > 0:
        base_prompt += f"\n第{retry_count + 1}次尝试，请调整策略"

    # 安全获取质量要求
    quality_reqs = _safe_get_context(context, "quality_requirements")
    if quality_reqs:
        base_prompt += f"\n质量要求：{quality_reqs}"

    return base_prompt

@dynamic_prompt
def rag_prompt_with_context(request: ModelRequest) -> str:
    """RAG agent prompt"""
    base_prompt = """你是一个RAG专家。基于检索文档回答问题。

    工作原则：严格依据文档、区分事实推断、提供引用

    重点关注：
    - 检索焦点和文档类型
    - 时间范围限制"""

    context = request.runtime.context or {}

    # 安全获取检索上下文
    retrieval_ctx = _safe_get_context(context, "retrieval_context")
    if retrieval_ctx:
        base_prompt += f"\n检索焦点：{retrieval_ctx}"

    # 安全获取文档类型（处理列表类型）
    doc_types = _safe_get_context(context, "document_types", [])
    if doc_types:
        if isinstance(doc_types, list):
            base_prompt += f"\n文档类型：{', '.join(str(dt) for dt in doc_types)}"
        else:
            base_prompt += f"\n文档类型：{doc_types}"

    return base_prompt

@dynamic_prompt
def ucagent_prompt_with_context(request: ModelRequest) -> str:
    """UCAgent专业prompt - IC验证"""
    base_prompt = """# UCAgent - IC验证专家

    专业领域：RTL设计、测试开发、调试分析、形式化验证

    核心工作：
    1. 代码分析：信号完整性、时序、状态机
    2. 问题识别：常见陷阱、边界条件
    3. 测试策略：边界测试、覆盖率"""

    context = request.runtime.context or {}

    # 安全获取任务类型
    task_type = _safe_get_context(context, "task_type")
    if task_type:
        # 任务类型映射字典
        guidance = {
            "code_analysis": "检查位宽匹配、验证信号定义",
            "test_generation": "设计边界测试、编写SVA检查点",
            "debugging": "追踪信号路径、定位错误",
            "verification": "形式化验证属性检查",
            "optimization": "性能优化建议"
        }

        # 获取对应的指导
        task_guidance = guidance.get(task_type)
        if task_guidance:
            base_prompt += f"\n\n当前任务：{task_type}\n指导：{task_guidance}"

    # 安全获取检索结果
    retrieval_results = _safe_get_context(context, "retrieval_results")
    if retrieval_results:
        # 确保结果是字符串格式
        results_str = str(retrieval_results) if not isinstance(retrieval_results, str) else retrieval_results
        # 限制长度避免prompt过长
        if len(results_str) > 1000:
            results_str = results_str[:1000] + "..."
        base_prompt += f"\n\n专业知识参考：\n{results_str}"

    return base_prompt