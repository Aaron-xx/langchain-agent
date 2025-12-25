#!/usr/bin/env python3
"""
Qdrant 向量数据库可视化管理工具

使用方法:
  python qdrant_manager.py --help          # 显示帮助
  python qdrant_manager.py --list          # 列出所有 collections
  python qdrant_manager.py --show          # 显示当前 collection 内容
  python qdrant_manager.py --search "查询"  # 搜索文档
  python qdrant_manager.py --delete <id>   # 删除指定 ID 的文档
  python qdrant_manager.py --clear         # 清空当前 collection
"""

import argparse
import json
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.config.config import get_config
from src.tools.documents import DocumentManager


def format_point(point, show_content=True, max_length=200):
    """格式化显示一个向量点"""
    output = []
    output.append(f"\n📄 ID: {point.id}")

    # 显示元数据
    if point.payload and 'metadata' in point.payload:
        metadata = point.payload['metadata']
        output.append(f"📍 来源: {metadata.get('source', '未知')}")

    # 显示内容
    if show_content and 'page_content' in point.payload:
        content = point.payload['page_content']
        if len(content) > max_length:
            content = content[:max_length] + "..."
        output.append(f"📝 内容预览:\n{content}")

    return "\n".join(output)


def list_collections(client: QdrantClient):
    """列出所有 collections"""
    print("\n=== Qdrant Collections ===")
    collections = client.get_collections().collections
    if not collections:
        print("❌ 没有任何 collection")
        return

    for col in collections:
        print(f"\n📚 Collection: {col.name}")
        info = client.get_collection(col.name)
        print(f"   - 文档数量: {info.points_count}")
        print(f"   - 向量维度: {info.config.params.vectors.size}")
        print(f"   - 距离度量: {info.config.params.vectors.distance}")


def show_collection(client: QdrantClient, collection_name: str, limit: int = 10):
    """显示 collection 内容"""
    print(f"\n=== Collection: {collection_name} ===")

    try:
        info = client.get_collection(collection_name)
        print(f"📊 总文档数: {info.points_count}")
        print(f"📏 向量维度: {info.config.params.vectors.size}")
        print(f"📐 距离度量: {info.config.params.vectors.distance}")

        # 获取文档
        points, _ = client.scroll(
            collection_name=collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )

        if not points:
            print("\n❌ Collection 为空")
            return

        print(f"\n📋 显示前 {len(points)} 个文档:")
        for point in points:
            print(format_point(point))
            print("-" * 80)

    except Exception as e:
        print(f"❌ 错误: {e}")


def search_documents(doc_manager: DocumentManager, query: str, k: int = 5):
    """搜索文档"""
    print(f"\n🔍 搜索: '{query}'\n")

    retriever = doc_manager.get_retriever('basic')
    results = retriever.invoke(query)[:k]

    if not results:
        print("❌ 没有找到相关文档")
        return

    print(f"📋 找到 {len(results)} 个相关文档:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. [来源: {doc.metadata.get('source', '未知')}]")
        print(f"   {doc.page_content[:300]}...")
        print("-" * 80)


def delete_document(client: QdrantClient, collection_name: str, point_id: str):
    """删除指定文档"""
    try:
        # 转换 ID
        if point_id.isdigit():
            point_id = int(point_id)

        # 先显示要删除的文档
        point = client.retrieve(
            collection_name=collection_name,
            ids=[point_id],
            with_payload=True
        )

        if point:
            print("\n⚠️  即将删除以下文档:")
            print(format_point(point[0]))

            confirm = input("\n确认删除？(y/N): ").lower().strip()
            if confirm == 'y':
                client.delete(
                    collection_name=collection_name,
                    points_selector=point_id
                )
                print("✅ 文档已删除")
            else:
                print("❌ 取消删除")
        else:
            print(f"❌ 找不到 ID 为 {point_id} 的文档")

    except Exception as e:
        print(f"❌ 删除失败: {e}")


def clear_collection(client: QdrantClient, collection_name: str):
    """清空 collection"""
    print(f"\n⚠️  即将清空 Collection: {collection_name}")
    info = client.get_collection(collection_name)
    print(f"当前文档数量: {info.points_count}")

    confirm = input("\n确认清空所有文档？(输入 'CLEAR' 确认): ").strip()
    if confirm == 'CLEAR':
        # 删除并重新创建 collection
        client.delete_collection(collection_name)
        print("✅ Collection 已清空")

        # 重新创建（需要 DocumentManager）
        config = get_config()
        doc_manager = DocumentManager(config)
        doc_manager._ensure_collection(config.embedding())
        print("✅ Collection 已重新创建")
    else:
        print("❌ 取消清空")


def interactive_mode(doc_manager: DocumentManager):
    """交互式模式"""
    print("\n🚀 进入交互模式 (输入 'exit' 退出)")
    print("可用命令: list, show, search <查询>, help")

    while True:
        try:
            cmd = input("\n> ").strip().split()
            if not cmd:
                continue

            command = cmd[0].lower()

            if command == 'exit':
                break
            elif command == 'help':
                print("\n可用命令:")
                print("  list - 列出所有 collections")
                print("  show - 显示当前 collection 内容")
                print("  search <查询> - 搜索文档")
                print("  stats - 显示统计信息")
                print("  exit - 退出")
            elif command == 'list':
                list_collections(doc_manager._client)
            elif command == 'show':
                collection_name = doc_manager.collection_name
                show_collection(doc_manager._client, collection_name)
            elif command == 'search' and len(cmd) > 1:
                query = " ".join(cmd[1:])
                search_documents(doc_manager, query)
            elif command == 'stats':
                stats = doc_manager.get_stats()
                print(f"\n📊 统计信息:")
                print(f"  Collection: {stats['collection_name']}")
                print(f"  文档数量: {stats['document_count']}")
                print(f"  向量维度: {stats['vector_size']}")
                print(f"  距离度量: {stats['distance_metric']}")
            else:
                print("❌ 未知命令，输入 'help' 查看帮助")

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")


def main():
    parser = argparse.ArgumentParser(description='Qdrant 向量数据库管理工具')
    parser.add_argument('--list', action='store_true', help='列出所有 collections')
    parser.add_argument('--show', action='store_true', help='显示当前 collection 内容')
    parser.add_argument('--search', type=str, help='搜索文档')
    parser.add_argument('--delete', type=str, help='删除指定 ID 的文档')
    parser.add_argument('--clear', action='store_true', help='清空当前 collection')
    parser.add_argument('--interactive', '-i', action='store_true', help='进入交互模式')
    parser.add_argument('--limit', type=int, default=10, help='显示文档数量限制 (默认: 10)')

    args = parser.parse_args()

    # 初始化
    config = get_config()
    doc_manager = DocumentManager(config)
    client = doc_manager._client
    collection_name = doc_manager.collection_name

    print(f"🔌 连接到 Qdrant: {config.get('vector_store.qdrant_url')}")

    try:
        if args.list:
            list_collections(client)
        elif args.show:
            show_collection(client, collection_name, args.limit)
        elif args.search:
            search_documents(doc_manager, args.search)
        elif args.delete:
            delete_document(client, collection_name, args.delete)
        elif args.clear:
            clear_collection(client, collection_name)
        elif args.interactive:
            interactive_mode(doc_manager)
        else:
            # 默认显示统计信息
            stats = doc_manager.get_stats()
            print(f"\n📊 {collection_name} 统计信息:")
            print(f"  文档数量: {stats['document_count']}")
            print(f"  向量维度: {stats['vector_size']}")
            print(f"  距离度量: {stats['distance_metric']}")
            print("\n💡 使用 --help 查看更多选项，或使用 -i 进入交互模式")

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
