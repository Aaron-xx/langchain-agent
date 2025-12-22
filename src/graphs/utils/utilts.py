
import base64
import asyncio
import logging
from pathlib import Path
from src.graphs.states import RAGState
from src.config import get_config
from src.tools.documents import DocumentManager

logger = logging.getLogger(__name__)

async def save_file(file_info: dict, data_dir: Path) -> Path | None:
    """Save a single file from base64 data."""
    try:
        filename = file_info.get('name', 'uploaded_file')
        file_data = file_info.get('data', '')

        if ',' in file_data:
            file_data = file_data.split(',', 1)[1]

        decoded_data = base64.b64decode(file_data)

        safe_filename = "".join(c for c in filename if c.isalnum() or c in '._-') or "uploaded_file"
        file_path = data_dir / safe_filename

        await asyncio.to_thread(lambda: file_path.write_bytes(decoded_data))
        return file_path

    except Exception as e:
        logger.error(f"Failed to save file {file_info.get('name')}: {e}")
        return None

async def process_documents_background(extracted_files: list) -> None:
    """Process documents in background thread."""
    try:
        config = get_config()
        doc_manager = DocumentManager(config)
        await asyncio.to_thread(doc_manager.add_documents, extracted_files)
        logger.info(f"Processed {len(extracted_files)} files")
    except Exception as e:
        logger.error(f"Background document processing failed: {e}")

def has_uploaded_files(state: RAGState) -> bool:
    """Check if message contains file attachments."""
    for msg in state["messages"]:
        files = msg.additional_kwargs.get("files", [])
        if files:
            return True
    return False

async def route_node(state: RAGState) -> str:
    """Determine next node based on state."""
    if has_uploaded_files(state):
        return "extract_files"
    return "retrieve"