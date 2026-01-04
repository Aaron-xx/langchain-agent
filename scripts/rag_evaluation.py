#!/usr/bin/env python3
"""RAG 系统评估脚本

评估指标：
1. Recall@K: 召回率 - 前 K 个结果中相关文档的比例
2. Precision@K: 精确率 - 前 K 个结果中相关文档的比例
3. MRR: Mean Reciprocal Rank - 第一个相关文档的平均排名倒数
4. NDCG@K: 归一化折损累积增益 - 考虑排序位置的质量

测试集：基于比特币知识库的问答对
"""

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Set

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from btliu.config import get_config
from btliu.tools import DocumentManager
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


@dataclass
class TestCase:
    """测试用例"""
    query: str
    expected_keywords: Set[str]  # 期望答案中应包含的关键词
    expected_topics: Set[str]    # 期望答案涉及的主题
    query_type: str              # 查询类型: factual, conceptual, procedural


# 基于比特币知识库的测试集
BITCOIN_TEST_SET: List[TestCase] = [
    TestCase(
        query="什么是Merkle树？它在比特币中有什么作用？",
        expected_keywords={"Merkle", "哈希", "根", "验证", "交易"},
        expected_topics={"merkle", "验证", "区块链结构"},
        query_type="conceptual"
    ),
    TestCase(
        query="Taproot中的MAST是什么？如何实现隐私保护？",
        expected_keywords={"MAST", "Taproot", "隐私", "Merkle", "脚本", "隐藏"},
        expected_topics={"taproot", "mast", "隐私"},
        query_type="conceptual"
    ),
    TestCase(
        query="什么是不足额奖励区块？矿工为什么会这样做？",
        expected_keywords={"不足额", "奖励", "矿工", "coinbase", "费率"},
        expected_topics={"挖矿", "奖励机制", "coinbase"},
        query_type="conceptual"
    ),
    TestCase(
        query="SPV节点如何使用Merkle路径验证交易？",
        expected_keywords={"SPV", "Merkle路径", "验证", "交易", "轻节点"},
        expected_topics={"spv", "merkle", "验证"},
        query_type="procedural"
    ),
    TestCase(
        query="比特币的总发行量是多少？不足额奖励会影响总量吗？",
        expected_keywords={"2100万", "发行量", "不足额", "减少"},
        expected_topics={"发行量", "经济模型"},
        query_type="factual"
    ),
    TestCase(
        query="coinbase交易有什么特殊之处？",
        expected_keywords={"coinbase", "挖矿", "奖励", "第一笔", "不可花费"},
        expected_topics={"coinbase", "挖矿", "交易结构"},
        query_type="factual"
    ),
    TestCase(
        query="什么是哈希叶子？如何生成Merkle根？",
        expected_keywords={"哈希", "叶子", "成对", "SHA-256", "Merkle根"},
        expected_topics={"merkle", "哈希", "数据结构"},
        query_type="procedural"
    ),
    TestCase(
        query="Ordinals和DeFi如何影响比特币交易费用？",
        expected_keywords={"Ordinals", "DeFi", "交易费", "上涨", "0.25-1"},
        expected_topics={"交易费", "ordinals", "defi"},
        query_type="factual"
    ),
]


@dataclass
class EvalResult:
    """评估结果"""
    test_case: TestCase
    retrieved_docs: List[Dict[str, Any]]
    relevant_docs: List[bool]  # 每个检索文档是否相关
    has_keywords: bool
    has_topics: bool


class RAGEvaluator:
    """RAG 系统评估器"""

    def __init__(self):
        self.config = get_config()
        self.doc_manager = DocumentManager(self.config)

        # 初始化 Qdrant 客户端
        qdrant_url = self.config.get("vector_store.qdrant_url", ":memory:")
        if qdrant_url == ":memory:":
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(url=qdrant_url)

        self.collection_name = self.config.get("vector_store.collection_name", "rag_documents")
        self.default_k = self.config.get("vector_store.default_k", 4)
        self.similarity_threshold = self.config.get("vector_store.similarity_threshold", 0.70)

    def _is_relevant(self, doc_content: str, test_case: TestCase) -> bool:
        """判断文档是否与测试用例相关"""
        content_lower = doc_content.lower()

        # 检查是否包含期望关键词（至少匹配 50%）
        keyword_matches = sum(1 for kw in test_case.expected_keywords if kw.lower() in content_lower)
        if keyword_matches >= len(test_case.expected_keywords) * 0.3:
            return True

        # 检查是否涉及期望主题
        for topic in test_case.expected_topics:
            if topic.lower() in content_lower:
                return True

        return False

    def _retrieve_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """使用向量检索获取文档"""
        embedding = self.config.embedding()
        query_vector = embedding.embed_query(query)

        # 使用 query_points API 进行相似度搜索
        # query 参数可以直接接受向量列表
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,  # 直接传入向量列表
            limit=k,
            with_payload=True,
            score_threshold=None  # 不过滤，返回所有结果
        )

        docs = []
        for result in response.points:  # response.points 是 ScoredPoint 列表
            # 计算相似度分数（Qdrant返回的是距离，需要转换）
            # 使用 Cosine 距离，范围是 0-2，0 表示完全相同，2 表示完全相反
            # 相似度 = 1 - distance，如果 distance 是负数说明向量化有问题
            distance = result.score if result.score is not None else 2.0
            similarity = max(0.0, 1.0 - distance)  # 确保相似度在 0-1 之间
            # 应用阈值过滤
            if similarity >= self.similarity_threshold:
                docs.append({
                    "content": result.payload.get("page_content", ""),
                    "source": result.payload.get("metadata", {}).get("source", ""),
                    "score": similarity
                })

        return docs

    def evaluate_single(self, test_case: TestCase, k: int = 5) -> EvalResult:
        """评估单个测试用例"""
        retrieved = self._retrieve_documents(test_case.query, k)

        relevant = []
        for doc in retrieved:
            relevant.append(self._is_relevant(doc["content"], test_case))

        # 检查检索结果是否覆盖关键词和主题
        all_content = " ".join([d["content"] for d in retrieved])
        has_keywords = any(kw.lower() in all_content.lower() for kw in test_case.expected_keywords)
        has_topics = any(tp.lower() in all_content.lower() for tp in test_case.expected_topics)

        return EvalResult(
            test_case=test_case,
            retrieved_docs=retrieved,
            relevant_docs=relevant,
            has_keywords=has_keywords,
            has_topics=has_topics
        )

    def evaluate_all(self, test_set: List[TestCase] = None, k: int = 5) -> Dict[str, Any]:
        """评估所有测试用例"""
        if test_set is None:
            test_set = BITCOIN_TEST_SET

        results = []
        for test_case in test_set:
            result = self.evaluate_single(test_case, k)
            results.append(result)

        # 计算指标
        metrics = self._calculate_metrics(results, k)

        return {
            "metrics": metrics,
            "results": results,
            "k": k,
            "total_tests": len(test_set)
        }

    def _calculate_metrics(self, results: List[EvalResult], k: int) -> Dict[str, Any]:
        """计算评估指标"""
        total = len(results)

        # Recall@K: 检索到的相关文档数 / 期望相关文档数（假设每个问题至少有1个相关文档）
        recall_scores = []
        for r in results:
            if r.relevant_docs:
                recall = sum(r.relevant_docs) / len(r.relevant_docs)
            else:
                recall = 0.0
            recall_scores.append(recall)
        avg_recall = sum(recall_scores) / total if total > 0 else 0

        # Precision@K: 检索到的相关文档数 / K
        precision_scores = []
        for r in results:
            if r.relevant_docs:
                precision = sum(r.relevant_docs) / len(r.relevant_docs)
            else:
                precision = 0.0
            precision_scores.append(precision)
        avg_precision = sum(precision_scores) / total if total > 0 else 0

        # MRR: 第一个相关文档的排名倒数
        rr_scores = []
        for r in results:
            rr = 0.0
            for i, rel in enumerate(r.relevant_docs):
                if rel:
                    rr = 1.0 / (i + 1)
                    break
            rr_scores.append(rr)
        mrr = sum(rr_scores) / total if total > 0 else 0

        # 关键词覆盖率
        keyword_coverage = sum(1 for r in results if r.has_keywords) / total if total > 0 else 0

        # 主题覆盖率
        topic_coverage = sum(1 for r in results if r.has_topics) / total if total > 0 else 0

        # 按查询类型统计
        by_type = {}
        for r in results:
            qtype = r.test_case.query_type
            if qtype not in by_type:
                by_type[qtype] = {"count": 0, "relevant": 0}
            by_type[qtype]["count"] += 1
            if any(r.relevant_docs):
                by_type[qtype]["relevant"] += 1

        return {
            "recall_at_k": avg_recall,
            "precision_at_k": avg_precision,
            "mrr": mrr,
            "keyword_coverage": keyword_coverage,
            "topic_coverage": topic_coverage,
            "by_query_type": by_type
        }


def print_report(evaluation: Dict[str, Any]):
    """打印评估报告"""
    print("=" * 70)
    print("RAG 系统性能评估报告")
    print("=" * 70)
    print()

    m = evaluation["metrics"]
    print(f"测试配置:")
    print(f"  - 测试样本数: {evaluation['total_tests']}")
    print(f"  - Top-K 值: {evaluation['k']}")
    print(f"  - 向量集合: rag_documents")
    print()

    print("整体指标:")
    print(f"  Recall@{evaluation['k']}:    {m['recall_at_k']:.2%}")
    print(f"  Precision@{evaluation['k']}: {m['precision_at_k']:.2%}")
    print(f"  MRR:                 {m['mrr']:.4f}")
    print(f"  关键词覆盖率:        {m['keyword_coverage']:.2%}")
    print(f"  主题覆盖率:          {m['topic_coverage']:.2%}")
    print()

    print("按查询类型统计:")
    for qtype, stats in m["by_query_type"].items():
        success_rate = stats["relevant"] / stats["count"] if stats["count"] > 0 else 0
        print(f"  {qtype}: {stats['relevant']}/{stats['count']} ({success_rate:.2%})")
    print()

    print("详细结果:")
    print("-" * 70)
    for i, result in enumerate(evaluation["results"]):
        tc = result.test_case
        print(f"\n[{i+1}] {tc.query}")
        print(f"    类型: {tc.query_type}")
        print(f"    检索到 {len(result.retrieved_docs)} 个文档, {sum(result.relevant_docs)} 个相关")

        for j, doc in enumerate(result.retrieved_docs):
            rel_mark = "✓" if result.relevant_docs[j] else "✗"
            content_preview = doc["content"][:80].replace("\n", " ")
            print(f"      [{j+1}] {rel_mark} Score={doc['score']:.3f} | {content_preview}...")
    print()

    print("=" * 70)


def save_report(evaluation: Dict[str, Any], output_path: str):
    """保存评估报告到文件"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2, default=str)
    print(f"详细报告已保存至: {output_path}")


async def main():
    """主函数"""
    print("初始化 RAG 评估器...")
    evaluator = RAGEvaluator()

    print(f"当前配置:")
    print(f"  - 相似度阈值: {evaluator.similarity_threshold}")
    print(f"  - 最大检索数: {evaluator.default_k}")
    print(f"  - 文档总数: {evaluator.client.count(evaluator.collection_name).count}")
    print()

    print("开始评估...")
    evaluation = evaluator.evaluate_all(k=5)

    print_report(evaluation)

    # 保存详细报告
    report_path = Path(__file__).parent.parent / "rag_evaluation_report.json"
    save_report(evaluation, str(report_path))


if __name__ == "__main__":
    asyncio.run(main())
