#!/bin/bash
# LangGraph RAG系统停止脚本

echo "🛑 停止LangGraph RAG系统..."

# 停止LangGraph进程
echo "📡 停止LangGraph后端..."
pkill -f "langgraph dev" 2>/dev/null && echo "✅ LangGraph后端已停止" || echo "⚠️  LangGraph后端未运行"

# 停止前端进程
echo "🌐 停止Agent Chat UI前端..."
pkill -f "next dev" 2>/dev/null && echo "✅ Agent Chat UI前端已停止" || echo "⚠️  Agent Chat UI前端未运行"

# 清理端口占用
echo "🧹 清理端口占用..."
lsof -ti:2024 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

echo "✅ 系统已完全停止"