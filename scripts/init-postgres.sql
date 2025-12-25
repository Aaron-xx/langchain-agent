-- PostgreSQL 初始化脚本
-- 用于 LangChain/LangGraph 持久化记忆存储

-- 启用 pgvector 扩展 (用于向量相似度搜索)
CREATE EXTENSION IF NOT EXISTS vector;

-- 长期记忆向量表 (用于语义搜索)
CREATE TABLE IF NOT EXISTS long_term_memory (
    id SERIAL PRIMARY KEY,
    namespace TEXT NOT NULL,
    key TEXT NOT NULL,
    value JSONB NOT NULL,
    vector vector(1024),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(namespace, key)
);

-- 创建索引优化查询性能
CREATE INDEX IF NOT EXISTS idx_memory_namespace ON long_term_memory(namespace);
CREATE INDEX IF NOT EXISTS idx_memory_vector ON long_term_memory USING ivfflat (vector vector_cosine_ops) WITH (lists = 100);

-- 授权给 langchain 用户
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO langchain;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO langchain;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO langchain;
