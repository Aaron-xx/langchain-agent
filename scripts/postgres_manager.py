#!/usr/bin/env python3
"""
PostgreSQL 检查点和存储可视化管理工具

使用方法:
  python postgres_manager.py --help           # 显示帮助
  python postgres_manager.py --list            # 列出所有 threads (checkpoints)
  python postgres_manager.py --show <thread>  # 显示特定 thread 的对话历史
  python postgres_manager.py --memory          # 列出所有存储的记忆
  python postgres_manager.py --delete <thread> # 删除特定 thread
  python postgres_manager.py --clear           # 清空所有数据
  python postgres_manager.py -i                # 交互模式
"""

import argparse
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from btliu.config.config import get_config
from btliu.common import RuntimeContext
from btliu.tools.documents import DocumentManager


# ANSI 颜色
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BLUE = "\033[34m"
CYAN = "\033[36m"
RESET = "\033[0m"


def format_message(msg, show_full_content=False, max_length=200):
    """格式化显示一条消息

    支持两种格式:
    - LangChain 消息对象 (有 .type 和 .content 属性)
    - 字典格式 (有 "type" 和 "content" 键)
    """
    output = []

    # 获取消息类型和内容 - 兼容对象和字典格式
    if hasattr(msg, 'type'):
        msg_type = msg.type
        content = getattr(msg, 'content', None)
        name = getattr(msg, 'name', None)
    else:
        msg_type = msg.get("type", "unknown")
        content = msg.get("content")
        name = msg.get("name")

    # 显示消息类型
    if msg_type == "human":
        output.append(f"👤 用户")
    elif msg_type == "ai":
        output.append(f"🤖 AI")
    elif msg_type == "system":
        output.append(f"⚙️  系统")
    elif msg_type == "tool":
        tool_name = name or "unknown"
        output.append(f"🔧 工具: {tool_name}")
    else:
        output.append(f"📨 {msg_type}")

    # 显示内容
    if content is not None:
        if isinstance(content, str):
            if not show_full_content and len(content) > max_length:
                content = content[:max_length] + "..."
            output.append(f"📝 内容:\n{content}")
        else:
            output.append(f"📝 内容: {content}")

    return "\n".join(output)


async def list_threads(checkpointer, limit: int = 20):
    """列出所有 threads"""
    print(f"\n{CYAN}=== PostgreSQL Checkpoint Threads ==={RESET}")

    try:
        # 列出所有 checkpoints
        # Use alist() for async iteration from main thread
        # CheckpointTuple is a namedtuple: (config, checkpoint, metadata, parent_config, pending_writes)
        threads = []
        async for checkpoint_tuple in checkpointer.alist(config=None, limit=limit):
            threads.append(checkpoint_tuple)

        if not threads:
            print(f"{RED}❌ 没有任何 thread{RESET}")
            return

        print(f"\n{GREEN}📋 找到 {len(threads)} 个 threads:{RESET}")
        for cpt in threads:
            # cpt.config is the RunnableConfig dict
            config = cpt.config
            configurable = config.get("configurable", {})
            thread_id = configurable.get("thread_id", "unknown")
            checkpoint_ns = configurable.get("checkpoint_ns", "")
            checkpoint_id = configurable.get("checkpoint_id", "unknown")

            print(f"\n{BLUE}🧵 Thread ID:{RESET} {thread_id}")
            if checkpoint_ns:
                print(f"   Namespace: {checkpoint_ns}")
            print(f"   Checkpoint ID: {checkpoint_id}")

            # 显示消息数量
            try:
                # cpt.checkpoint contains the checkpoint data
                checkpoint = cpt.checkpoint
                if "messages" in checkpoint.get("channel_values", {}):
                    msg_count = len(checkpoint["channel_values"]["messages"])
                    print(f"   💬 消息数: {msg_count}")
            except Exception as e:
                print(f"   {YELLOW}⚠️  无法读取消息数: {e}{RESET}")

            print("-" * 80)

    except Exception as e:
        print(f"{RED}❌ 错误: {e}{RESET}")


async def show_thread_history(checkpointer, thread_id: str, limit: int = 10):
    """显示特定 thread 的对话历史"""
    print(f"\n{CYAN}=== Thread: {thread_id} ==={RESET}")

    try:
        # 获取最新的 checkpoint
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint = await checkpointer.aget(config)

        if not checkpoint:
            print(f"{RED}❌ Thread 不存在或为空{RESET}")
            return

        # 显示消息历史
        if "messages" in checkpoint.get("channel_values", {}):
            messages = checkpoint["channel_values"]["messages"]

            # 限制显示数量
            if len(messages) > limit:
                print(f"{YELLOW}ℹ️  显示最近 {limit} 条消息 (共 {len(messages)} 条){RESET}")
            else:
                print(f"{GREEN}💬 消息总数: {len(messages)}{RESET}")

            print(f"\n{GREEN}📋 对话历史:{RESET}")
            for i, msg in enumerate(messages[-limit:], 1):
                print(f"\n{BLUE}--- 消息 {i} ---{RESET}")
                print(format_message(msg))
                print("-" * 80)
        else:
            print(f"{YELLOW}⚠️  没有消息历史{RESET}")

    except Exception as e:
        print(f"{RED}❌ 错误: {e}{RESET}")


async def list_memories(store, user_id: str = "langchain", limit: int = 20):
    """列出所有存储的记忆

    记忆按 (user_id, category) 命名空间组织。
    """
    print(f"\n{CYAN}=== PostgreSQL Store (跨线程记忆) ==={RESET}")
    print(f"{CYAN}User ID: {user_id}{RESET}\n")

    try:
        # 搜索特定用户的所有命名空间
        # 命名空间格式: (user_id, category)
        all_items = []

        # 搜索用户的所有命名空间前缀
        items = await store.asearch((user_id,), limit=limit)

        if items:
            all_items.extend(items)

            # 按类别分组
            by_category: dict[str, list] = {}
            for item in items:
                if len(item.namespace) > 1:
                    category = item.namespace[1]
                    if category not in by_category:
                        by_category[category] = []
                    by_category[category].append(item)

            # 显示分类结果
            for category, items in sorted(by_category.items()):
                print(f"\n{GREEN}📁 类别: {category} ({len(items)} 条){RESET}")
                for item in items:
                    key = item.key
                    value = item.value
                    print(f"\n{BLUE}🔑 Key:{RESET} {key}")
                    if isinstance(value, dict):
                        for k, v in value.items():
                            if isinstance(v, str) and len(v) > 200:
                                v = v[:200] + "..."
                            print(f"   {k}: {v}")
                    else:
                        print(f"   {value}")
                    print("-" * 60)
        else:
            print(f"{YELLOW}⚠️  没有找到用户 '{user_id}' 的记忆{RESET}")
            print(f"{CYAN}提示: 使用 --user-id 指定用户，或查看所有用户{RESET}")

    except Exception as e:
        print(f"{RED}❌ 错误: {e}{RESET}")


async def search_memories(store, query: str, user_id: str = "langchain", limit: int = 10):
    """搜索记忆"""
    print(f"\n{CYAN}🔍 搜索记忆: '{query}' (User: {user_id}){RESET}")

    try:
        # 在用户的所有命名空间中搜索
        found = 0

        # 搜索用户命名空间前缀下的所有内容
        items = await store.asearch((user_id,), query=query, limit=limit)

        if items:
            for item in items:
                if found >= limit:
                    break
                found += 1

                # 显示完整的命名空间
                namespace_str = "/".join(item.namespace)
                print(f"\n{GREEN}📍 命名空间: {namespace_str}{RESET}")
                print(f"{BLUE}🔑 Key:{RESET} {item.key}")

                value = item.value
                if isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, str) and len(v) > 200:
                            v = v[:200] + "..."
                        print(f"   {k}: {v}")
                else:
                    print(f"   {value}")
                print("-" * 60)

        if found == 0:
            print(f"{YELLOW}❌ 没有找到匹配的记忆{RESET}")
        else:
            print(f"\n{GREEN}📋 找到 {found} 条记忆{RESET}")

    except Exception as e:
        print(f"{RED}❌ 错误: {e}{RESET}")


async def delete_thread(checkpointer, thread_id: str):
    """删除特定 thread"""
    try:
        # 先显示 thread 内容
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint = await checkpointer.aget(config)

        if not checkpoint:
            print(f"{RED}❌ Thread '{thread_id}' 不存在{RESET}")
            return

        print(f"\n{YELLOW}⚠️  即将删除 Thread: {thread_id}{RESET}")
        if "messages" in checkpoint.get("channel_values", {}):
            msg_count = len(checkpoint["channel_values"]["messages"])
            print(f"   消息数: {msg_count}")

        confirm = input(f"\n{RED}确认删除？(y/N):{RESET} ").lower().strip()
        if confirm == 'y':
            # 删除 checkpoint - 使用 adelete_thread 而不是 adelete
            await checkpointer.adelete_thread(thread_id)
            print(f"{GREEN}✅ Thread 已删除{RESET}")
        else:
            print(f"{YELLOW}❌ 取消删除{RESET}")

    except Exception as e:
        print(f"{RED}❌ 删除失败: {e}{RESET}")


async def clear_all_data(checkpointer, store):
    """清空所有数据"""
    print(f"\n{RED}⚠️  警告: 即将清空所有数据！{RESET}")
    print(f"   - 所有 checkpoints (对话历史)")
    print(f"   - 所有存储的记忆")

    confirm = input(f"\n{RED}确认清空？(输入 'CLEAR ALL'):{RESET} ").strip()
    if confirm != 'CLEAR ALL':
        print(f"{YELLOW}❌ 取消清空{RESET}")
        return

    try:
        # 列出并删除所有 threads
        # Use alist() for async iteration from main thread
        configs = []
        async for config in checkpointer.alist(config=None):
            configs.append(config)

        deleted_count = 0
        for checkpoint_tuple in configs:
            try:
                # 从 checkpoint_tuple.config 中提取 thread_id
                configurable = checkpoint_tuple.config.get("configurable", {})
                thread_id = configurable.get("thread_id")
                if thread_id:
                    await checkpointer.adelete_thread(thread_id)
                    deleted_count += 1
            except Exception:
                pass

        print(f"{GREEN}✅ 已删除 {deleted_count} 个 threads{RESET}")

        # 删除所有 store 中的记忆
        memory_deleted = 0
        try:
            items = await store.asearch((), limit=10000)
            for item in items:
                try:
                    await store.adelete(item.namespace, item.key)
                    memory_deleted += 1
                except Exception:
                    pass
            print(f"{GREEN}✅ 已删除 {memory_deleted} 条记忆{RESET}")
        except Exception as e:
            print(f"{YELLOW}⚠️  删除记忆时出错: {e}{RESET}")

        print(f"{GREEN}✅ 清空完成{RESET}")

    except Exception as e:
        print(f"{RED}❌ 清空失败: {e}{RESET}")


async def interactive_mode(checkpointer, store, user_id: str = "langchain"):
    """交互式模式"""
    print(f"\n{GREEN}🚀 进入交互模式 (输入 'exit' 退出){RESET}")
    print(f"可用命令: list, show <thread>, memory, search <查询>, delete <thread>, help")
    print(f"当前用户: {user_id} (使用 'user <id>' 切换)")

    while True:
        try:
            cmd = input(f"\n{CYAN}>{RESET} ").strip().split()
            if not cmd:
                continue

            command = cmd[0].lower()

            if command == 'exit':
                break
            elif command == 'help':
                print(f"\n{GREEN}可用命令:{RESET}")
                print(f"  list - 列出所有 threads")
                print(f"  show <thread_id> - 显示特定 thread 的对话历史")
                print(f"  memory - 列出当前用户的所有记忆")
                print(f"  search <查询> - 搜索当前用户的记忆")
                print(f"  delete <thread_id> - 删除特定 thread")
                print(f"  user <id> - 切换用户 ID")
                print(f"  stats - 显示统计信息")
                print(f"  exit - 退出")
            elif command == 'user' and len(cmd) > 1:
                user_id = cmd[1]
                print(f"{GREEN}✓ 用户切换为: {user_id}{RESET}")
            elif command == 'list':
                await list_threads(checkpointer)
            elif command == 'show' and len(cmd) > 1:
                await show_thread_history(checkpointer, cmd[1])
            elif command == 'memory':
                await list_memories(store, user_id)
            elif command == 'search' and len(cmd) > 1:
                query = " ".join(cmd[1:])
                await search_memories(store, query, user_id)
            elif command == 'delete' and len(cmd) > 1:
                await delete_thread(checkpointer, cmd[1])
            elif command == 'stats':
                print(f"\n{GREEN}📊 PostgreSQL 统计信息:{RESET}")
                # 列出 threads
                # Use alist() for async iteration from main thread
                configs = []
                async for config in checkpointer.alist(config=None):
                    configs.append(config)
                print(f"  Thread 数量: {len(configs)}")
            else:
                print(f"{RED}❌ 未知命令，输入 'help' 查看帮助{RESET}")

        except KeyboardInterrupt:
            print(f"\n\n{YELLOW}👋 再见！{RESET}")
            break
        except Exception as e:
            print(f"{RED}❌ 错误: {e}{RESET}")


async def main():
    parser = argparse.ArgumentParser(description='PostgreSQL 检查点和存储管理工具')
    parser.add_argument('--list', action='store_true', help='列出所有 threads')
    parser.add_argument('--show', type=str, help='显示特定 thread 的对话历史')
    parser.add_argument('--memory', action='store_true', help='列出所有存储的记忆')
    parser.add_argument('--search', type=str, help='搜索记忆')
    parser.add_argument('--delete', type=str, help='删除特定 thread')
    parser.add_argument('--clear', action='store_true', help='清空所有数据')
    parser.add_argument('--interactive', '-i', action='store_true', help='进入交互模式')
    parser.add_argument('--limit', type=int, default=20, help='显示数量限制 (默认: 20)')
    parser.add_argument('--user-id', type=str, default='langchain', help='用户 ID (默认: langchain)')

    args = parser.parse_args()

    # 初始化配置
    config = get_config()
    db_uri = config.get("checkpoint_db_uri")

    if not db_uri:
        print(f"{RED}❌ 错误: 未配置 checkpoint_db_uri{RESET}")
        print(f"请在 config.json 中设置 checkpoint_db_uri")
        return

    print(f"{CYAN}🔌 连接到 PostgreSQL{RESET}")
    print(f"   URI: {db_uri}")

    # 创建 checkpointer 和 store
    from psycopg_pool import AsyncConnectionPool
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.store.postgres.aio import AsyncPostgresStore

    pool = AsyncConnectionPool(
        conninfo=db_uri,
        kwargs={"autocommit": True},
        open=False,  # Prevent deprecation warning
    )
    await pool.open()

    try:
        checkpointer = AsyncPostgresSaver(pool)
        store = AsyncPostgresStore(pool)

        # 执行命令
        if args.list:
            await list_threads(checkpointer, args.limit)
        elif args.show:
            await show_thread_history(checkpointer, args.show, args.limit)
        elif args.memory:
            await list_memories(store, args.user_id, args.limit)
        elif args.search:
            await search_memories(store, args.search, args.user_id)
        elif args.delete:
            await delete_thread(checkpointer, args.delete)
        elif args.clear:
            await clear_all_data(checkpointer, store)
        elif args.interactive:
            await interactive_mode(checkpointer, store, args.user_id)
        else:
            # 默认显示统计
            await list_threads(checkpointer)
            print(f"\n{CYAN}💡 使用 --help 查看更多选项，或使用 -i 进入交互模式{RESET}")

    finally:
        await pool.close()


if __name__ == '__main__':
    asyncio.run(main())
