"""Simplified RAG nodes"""
from typing import Dict, Any
from langgraph.runtime import Runtime
from src.common import RuntimeContext
from src.graphs.states import RAGState
from pathlib import Path
from src.config import get_config
from langchain_core.messages import AIMessage
import logging
import asyncio

from .utilts import process_documents_background, save_file

logger = logging.getLogger(__name__)

async def retrieve_node(state: RAGState, runtime: Runtime[RuntimeContext]) -> RAGState:
    """Retrieve relevant documents using vector store."""
    try:
        messages = state["messages"]

        if not messages:
            return {
                "context": "",
            }

        last_message = messages[-1]
        query = last_message.content

        doc_manager = runtime.context.doc_manager
        retriever = doc_manager.vector_store.as_retriever(
            search_kwargs={"k": 4}
        )

        docs = await retriever.ainvoke(query)

        context = "\n\n".join([doc.page_content for doc in docs])

        return {
            "messages": [AIMessage(content=context)],
            "context": context,
        }

    except Exception as e:
        logger.error(f"Retrieval error: {e}", exc_info=True)
        return {
            "context": "",
        }

async def generate_node(state: RAGState, config, runtime: Runtime[RuntimeContext]) -> Dict[str, Any]:
    """Generate response using LLM with recent conversation context."""
    try:
        llm = runtime.context.config.chat()

        messages = state["messages"]

        conversation_history = ""
        if messages:
            recent_messages = messages[-5:]
            conversation_history = "Recent conversation history:\n"
            for msg in recent_messages:
                role = "User" if msg.type == "human" else "Assistant"
                conversation_history += f"{role}: {msg.content}\n"
            conversation_history += "\n"

        current_query = messages[-1].content if messages else state.get('query', '')

        prompt = f"""You are a helpful AI assistant. Answer questions based on conversation history and retrieved documents.

{conversation_history}Retrieved documents:
{state.get('context', 'No context available')}

Current question: {current_query}

Notes:
1. If user asks to remember information, confirm in your response
2. If asked about previous information, search conversation history
3. Provide complete answer combining documents and history

Answer:"""
        answer = []
        async for chunk in llm.astream(prompt, config):
            if hasattr(chunk, 'content') and chunk.content:
                answer += chunk.content

        return {
            "messages": [AIMessage(content=answer)],
            "error": None
        }

    except Exception as e:
        logger.error(f"Generation error: {e}")
        error_msg = f"Sorry, error generating response: {str(e)}"

        return {
            "messages": [AIMessage(content=error_msg)],
            "error": str(e)
        }

async def extract_uploaded_files(state: RAGState) -> Dict[str, Any]:
    """Extract and save uploaded files from message attachments."""
    try:
        config = get_config()
        data_dir = Path(config.get('document_processing.data_dir', './data'))
        await asyncio.to_thread(data_dir.mkdir, parents=True, exist_ok=True)

        messages = state.get('messages', [])
        extracted_files = []

        for message in messages:
            files = message.additional_kwargs.get('files', [])
            for file_info in files:
                if isinstance(file_info, dict) and file_info.get('data'):
                    file_path = await save_file(file_info, data_dir)
                    if file_path:
                        extracted_files.append(str(file_path))

        if extracted_files:
            asyncio.create_task(process_documents_background(extracted_files))

        return {
            "extracted_files": extracted_files,
            "files_processed": bool(extracted_files)
        }

    except Exception as e:
        return {
            "extracted_files": [],
            "files_processed": False,
            "file_extraction_error": str(e)
        }
