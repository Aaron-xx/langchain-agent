# Qdrant REST API 示例

## 1. 获取所有 Collections
```bash
curl http://localhost:6333/collections
```

## 2. 获取 Collection 信息
```bash
curl http://localhost:6333/collections/rag_documents
```

## 3. 搜索文档
```bash
curl -X POST http://localhost:6333/collections/rag_documents/points/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3, ...],
    "limit": 5,
    "with_payload": true
  }'
```

## 4. 获取所有文档（分页）
```bash
curl -X POST http://localhost:6333/collections/rag_documents/points/scroll \
  -H "Content-Type: application/json" \
  -d '{
    "limit": 10,
    "with_payload": true,
    "with_vectors": false
  }'
```

## 5. 按 ID 获取文档
```bash
curl -X POST http://localhost:6333/collections/rag_documents/points/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "ids": ["0887fe17-57ed-46e5-ba84-7cb2062488b8"],
    "with_payload": true
  }'
```

## 6. 删除文档
```bash
curl -X DELETE http://localhost:6333/collections/rag_documents/points/delete \
  -H "Content-Type: application/json" \
  -d '{
    "points": ["0887fe17-57ed-46e5-ba84-7cb2062488b8"]
  }'
```

## 7. 清空 Collection
```bash
curl -X DELETE http://localhost:6333/collections/rag_documents
```

## 8. 创建过滤搜索
```bash
curl -X POST http://localhost:6333/collections/rag_documents/points/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3, ...],
    "filter": {
      "must": [
        {
          "key": "metadata.source",
          "match": {
            "value": "data/documents/test_qdrant.md"
          }
        }
      ]
    },
    "limit": 5,
    "with_payload": true
  }'
```