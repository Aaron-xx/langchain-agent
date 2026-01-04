import logging
import os
import pwd
import sys
from typing import Callable
from dotenv import load_dotenv
import time
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent, AgentState
from langgraph.graph.state import Command
from pydantic import BaseModel, Field

# --- 核心组件：Postgres 持久化检查点 ---
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.prebuilt import ToolRuntime
from psycopg_pool import ConnectionPool
from langchain.tools import tool
import subprocess

#from langchain_community.storage import MongoDBStore
from langchain.agents.middleware import HumanInTheLoopMiddleware, ModelRequest, ModelResponse, SummarizationMiddleware, wrap_model_call
from deepagents.middleware import FilesystemMiddleware
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware.human_in_the_loop import (
    HITLResponse,
    ApproveDecision,
    EditDecision,
    RejectDecision
)
from deepagents import create_deep_agent

logger = logging.getLogger(__name__)
# 配置 API KEY
load_dotenv(override=True)

DB_URI = os.getenv("DATABASE_URL", "postgresql://langchain:langchain_postgres@postgres:5432/langchain")
llm = ChatOpenAI(
    base_url="https://open.bigmodel.cn/api/coding/paas/v4",
    model="glm-4.5",
    temperature=0
)

small_model = ChatOpenAI(
    base_url="https://open.bigmodel.cn/api/coding/paas/v4",
    model="glm-4",
    temperature=0
)
large_model = llm

pool=None

class TaskInfo(BaseModel):
    task_name: str = Field(description="任务的名字")
    task_description: str = Field(description="任务的描述")
    task_status: str = Field(description="任务的状态")
    task_progress: int = Field(description="任务的进度")
    task_result: str = Field(description="任务的结果")

class UserInfo(BaseModel):
    """从文本中提取的用户信息"""
    user_name: str = Field(description="用户的名字")
    additional_info: str = Field(description="关于用户的其他信息，例如岗位，任务等")

class TaskState(AgentState):
    user_id: str
    user_info: UserInfo
    task_info: str


class SSHState(BaseModel):
    host: str = Field(description="本地SSH 配置里的 远程主机Host 名称")
    user: str = Field(description="远程主机的用户")
    passwd: int = Field(description="远程主机的用户的密码")
    port: int = Field(description="登陆远程主机的端口")
    command: str = Field(description="需要在远程主机执行的命令")

@tool
def ssh_run(host: str, command: str) -> str:
    """
    使用系统 SSH 登录远程主机（使用 ~/.ssh/config 配置）并执行命令

    参数:
        host: SSH 配置里的 Host 名称
        command: 要执行的命令

    返回:
        命令执行结果
    """
    try:
        ssh_cmd = ["ssh", host, command]
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"SSH命令执行失败: {e.stderr.strip()}"


# ============ 定义记忆管理工具（使用 BaseStore）===========
# 注意：store 参数使用 InjectedStore 注解，由 LangGraph 自动注入
# InjectedStore() 标记会让 Pydantic 在生成 JSON Schema 时跳过这个参数
# LLM 不会看到 store 参数，只会看到 user_id 和 info
@tool
def save_memory(
    info: str,
    runtime: ToolRuntime = None,
) -> str:
    """
    保存用户信息和任务信息到跨线程记忆.

    此工具将信息持久化到 PostgreSQL Store，可在不同会话间共享。
    记忆按用户id和用户名隔离（基于 context 中的 user_id 和 user_info 中的 user_name）。

    Args:
        info: 要记忆的信息内容和任务信息

    Returns:
        确认消息，包含记忆 ID 和用户名

    Example:
        save_memory("用户名叫 Alice，是一名 Python 工程师", "Alice")
        save_memory("正在开发 ML 推荐系统项目", "Bob")
    """
    state = runtime.state
    if state is None:
        return "错误: context 不可用"

    user_id = state.get("user_id", "unknown_user")
    user_info: UserInfo = state.get("user_info", None)
    task_info = state.get("task_info", None)

    user_name = user_info.user_name if user_info else "unknown_user"

    store = runtime.store
    # 创建命名空间: (user_id, category)
    namespace = (user_id, user_name)

    # 生成记忆 ID（使用时间戳）
    memory_id = f"mem_{int(time.time())}"

    # 存储到 BaseStore（自动持久化）
    store.put(
        namespace,
        memory_id,
        {
            "info": info,
            "user_name": user_name,
            "task_info": task_info,
            "timestamp": time.time(),
        },
    )

    return f"已保存记忆 [{user_name}]: {info}"

@tool
def recall_memory(
    query: str | None = None,
    limit: int = 10,
    runtime: ToolRuntime = None,
) -> str:
    """从跨线程记忆中检索用户信息.

    检索之前保存的记忆，支持按用户名过滤。

    Args:
        query: 可选的搜索关键词（暂未实现，预留参数）
        limit: 返回的最大记忆数量（默认: 10）

    Returns:
        格式化的记忆列表，或未找到的提示消息

    Example:
        recall_memory()  # 获取所有 general 类别记忆
        recall_memory(None, "Bob", 5)  # 获取最近 5 条项目记忆
    """
    state = runtime.state
    if state is None:
        return "错误: state 不可用"

    user_id = state.get("user_id", "anonymous")
    user_info: UserInfo = state.get("user_info", UserInfo(user_name="anonymous"))
    user_name = user_info.user_name

    # Get store directly from runtime (injected by LangGraph)
    store = runtime.store

    if store is None:
        return "错误: Store 未配置，无法检索记忆"

    # 搜索命名空间前缀: (user_id, user_name)
    namespace_prefix = (user_id, user_name)

    try:
        items = store.search(namespace_prefix, limit=limit)

        if not items:
            return f"未找到用户 '{user_name}' 的记忆和任务信息"

        # 格式化结果
        results = []
        for item in items:
            info = item.value.get("info", "N/A")
            task_info = item.value.get("task_info", "N/A")
            results.append(f"- {user_name}: {info}\n- 任务信息: {task_info}")

        return f"找到用户 '{user_name}' 的 {len(results)} 条记忆和任务信息:\n" + "\n".join(results)

    except Exception as e:
        return f"检索用户 '{user_name}' 的记忆和任务信息时出错: {str(e)}"

@wrap_model_call
def dynamic_model_router(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """
    根据对话上下文动态切换模型
    """
    # 获取当前对话的状态（例如消息列表）
    state = request.state
    print(f"--- [Middleware] 当前对话状态: {state} ---")
    messages = state.get("messages", [])
    print(f"--- [Middleware] 当前消息数量: {len(messages)} ---")

    # 获取上下文中的用户角色
    print(f"打印运行时上下文: {request.runtime.context}")

    # === 逻辑判断示例 ===
    # 场景 A: 如果对话轮数超过 5 轮，切换到大模型处理复杂上下文
    if len(messages) > 5:
        print(f"--- [Middleware] 检测到长对话 ({len(messages)} msgs)，切换 ---")
        # 使用 .override() 方法替换本次调用的模型
        request = request.override(model=large_model)

    # 场景 B: 如果用户输入包含特定关键词 (仅作演示，实际可用分类器)
    elif messages and "复杂分析" in messages[-1].content:
         print("--- [Middleware] 检测到复杂任务，切换 ---")
         request = request.override(model=large_model)

    else:
        print("--- [Middleware] 使用默认小模型 ---")
        # 默认使用 create_agent 初始化时传入的模型（即 small_model）

    # 继续执行调用
    return handler(request)

SYSTEM_PROMPT="""
你是一个专业的助手。

你的能力：
1. 记忆用户的重要信息存入长期记忆（跨会话持久化）
2. 从长期记忆中检索用户信息
3. 完成用户下发的其他任务

工作流程：
- 在对话过程设计用户信息时，主动存储
- 当询问问题设计用户信息偏好时时，进行检索
- 记忆是跨会话的，即使在新的对话中也能记住用户信息

注意：调用工具时，必须传入 user_id 参数（从 state 中获取）。
"""

def get_agent(tools: list, middlewares: list):
    print("--- 正在连接 PostgreSQL 数据库 ---")

    # 使用 ConnectionPool 管理数据库连接
    pool = ConnectionPool(conninfo=DB_URI, max_size=20, kwargs={"autocommit": True})

    checkpointer = PostgresSaver(pool)
    store = PostgresStore(pool)

    print("🔧 初始化 Checkpointer 表结构...")
    checkpointer.setup()
    print("✅ Checkpointer 表结构初始化完成")

    print("🔧 初始化 Store 表结构...")
    store.setup()
    print("✅ Store 表结构初始化完成")

    agent = create_agent(
        model=llm,
        tools=tools,
        middleware=middlewares,
        state_schema=TaskState,
        store=store,
        checkpointer=checkpointer,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent


def get_middleware() -> list:
    hitl_middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "write_file": {"allowed_decisions": ["approve", "edit", "reject"], "description": "需要人工批准才能写入文件"},
            "read_file": {"allowed_decisions": ["approve", "edit", "reject"], "description": "需要人工批准才能读取文件"},
            "save_memory": {"allowed_decisions": ["approve", "edit", "reject"], "description": "需要人工批准才能保存记忆"},
            # "recall_memory": {"allowed_decisions": ["approve", "edit", "reject"], "description": "需要人工批准才能检索记忆"},
            "ssh_run": {"allowed_decisions": ["approve", "edit", "reject"], "description": "需要人工批准才能执行SSH命令"},
        },
    )

    summarization_middleware = SummarizationMiddleware(
        model=llm,
        trigger=("tokens", 10000),          # 历史消息 token 数量超过 500 时触发压缩
        keep=("messages", 20),                     # 保留最近 5 条消息
        summary_prompt="请将以下对话历史进行摘要，保留关键决策点和技术细节：\n\n{messages}\n\n摘要:"   # 摘要提示词
    )

    filesystem_middleware = FilesystemMiddleware(
        backend=FilesystemBackend(
            root_dir=pwd.getpwuid(os.getuid()).pw_dir,
            virtual_mode=True,
        ),
    )

    middlewares=[
        hitl_middleware,
        summarization_middleware,
        filesystem_middleware,
        dynamic_model_router
    ]

    return middlewares

async def cleanup(self) -> None:
    """Clean up all resources in the correct order.

    This method should be called from all exit points:
    - Normal exit (/exit command)
    - Keyboard interrupt (Ctrl+C)
    - EOF (Ctrl+D)
    """
    if pool is not None:
        try:
            await pool.close()
        except Exception as e:
            logger.warning(f"Error closing database pool: {e}")


    logger.info("CLI application cleaned up")

def main():
    current_user = pwd.getpwuid(os.getuid()).pw_name
    print(current_user)
    thread_config = {"configurable": {"thread_id": current_user}}

    tools = [save_memory, recall_memory, ssh_run]
    middlewares = get_middleware()
    agent = get_agent(tools=tools, middlewares=middlewares)

    userinfo = UserInfo(
        user_name=current_user,
        additional_info="我喜欢python，我是一名Python工程师",
    )

    while True:
        try:
            print("User > ", end="", flush=True)
            line = sys.stdin.readline()

            # EOF reached (stdin closed)
            if not line:
                break

            line = line.strip()
            if not line:
                print("(输入为空，使用 Ctrl+D 退出)")
                continue

            payload = {
                "messages": [{"role": "user", "content": line}],
                "user_id": current_user,
                "user_info": userinfo
            }
            for chunk in agent.stream(
                payload,
                config=thread_config,
                stream_mode="values"
            ):
                last_msg = chunk["messages"][-1]
                if last_msg.type == "ai" and last_msg.content:
                    print(f"🤖 Agent: {last_msg.content}", end="", flush=True)
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tool_call in last_msg.tool_calls:
                        print(f"   🔧 [调用工具]: {tool_call['name']}")

            snapshot = agent.get_state(config=thread_config)

            if snapshot.tasks:
                print(f"\n--- 🛑 执行已暂停 (HITL Middleware) ---")
                print(f"下一步骤 (Next): {snapshot.next}")
                print(f"任务数量: {len(snapshot.tasks) if snapshot.tasks else 0}")

                last_message = snapshot.values["messages"][-1]
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    tool_call = last_message.tool_calls[0]
                    print(f"\n[待审批操作]:")
                    print(f"  - 参数: {tool_call['args']}")
                    print(f"  - 工具: {tool_call['name']}")
                    approval = input("\n[管理员]: 是否批准执行此操作? (y/n/e[编辑]): ")

                    if approval.lower() == 'y':
                        print("\n[系统]: 操作已批准，继续执行...")
                        hitl_response = HITLResponse(
                            decisions=[ApproveDecision(type="approve")]
                        )
                        for event in agent.stream(
                            Command(resume=hitl_response),
                            config=thread_config,
                            stream_mode="values"
                        ):
                            if "messages" in event:
                                last_msg = event["messages"][-1]
                                if last_msg.type == "ai" and last_msg.content:
                                    print(f"🤖 Agent: {last_msg.content}", end="", flush=True)
                                elif last_msg.type == "tool":
                                    print(f"   🔧 [工具输出]: {last_msg.content}")

                    elif approval.lower() == 'e':
                        print("\n[系统]: 编辑模式...")
                        print(f"当前参数: {tool_call['args']}")
                        new_task_info = input(f" (当前: {tool_call['args'].get('task_info', '')}，留空保持不变): ").strip()
                        updated_args = tool_call['args'].copy()

                        updated_args['task_info'] = new_task_info
                        print(f"新参数: {tool_call['args']}")

                        print(f"\n[系统]: 使用更新后的参数继续执行...")
                        print(f"更新后的参数: {updated_args}")
                        hitl_response = HITLResponse(
                            decisions=[EditDecision(
                                type="edit",
                                edited_action={
                                    "name": tool_call['name'],
                                    "args": updated_args
                                }                                
                            )]
                        )
                        for event in agent.stream(
                            Command(resume=hitl_response),
                            config=thread_config,
                            stream_mode="values"
                        ):
                            if "messages" in event:
                                last_msg = event["messages"][-1]
                                if last_msg.type == "ai" and last_msg.content:
                                    print(f"🤖 Agent: {last_msg.content}", end="", flush=True)
                                elif last_msg.type == "tool":
                                    print(f"   🔧 [工具输出]: {last_msg.content}")

                    elif approval.lower() == 'n':
                        print("\n[系统]: 操作被拒绝。")
                        rejection_reason = input("拒绝原因 (可选): ").strip() or "操作被管理员拒绝"
                        hitl_response = HITLResponse(
                            decisions=[RejectDecision(
                                type="reject",
                                message=rejection_reason
                            )]
                        )
                        for event in agent.stream(
                            Command(resume=hitl_response),
                            config=thread_config,
                            stream_mode="values"
                        ):
                            if "messages" in event:
                                last_msg = event["messages"][-1]
                                if last_msg.type == "ai" and last_msg.content:
                                    print(f"🤖 Agent: {last_msg.content}", end="", flush=True)
                                elif last_msg.type == "tool":
                                    print(f"   🔧 [工具输出]: {last_msg.content}")
                    else:
                        print("无效的输入")
                        continue
                else:
                    print("流程已完成，没有触发中断。")
                    # 打印最终结果
                    if snapshot.values.get("messages"):
                        last_msg = snapshot.values["messages"][-1]
                        if last_msg.type == "ai" and last_msg.content:
                            print(f"\n[最终回复]: {last_msg.content}")
                print()
        except KeyboardInterrupt:
            print("\nbye")
            break  # Exit on second Ctrl+C

        except EOFError:
            print("\nbye")
            break

if __name__ == "__main__":
   main()