"""Document management system using Qdrant vector store.

This module provides a document manager that:
- Loads documents from various file formats (PDF, TXT, MD, JSON)
- Splits documents into chunks using RecursiveCharacterTextSplitter
- Stores and retrieves embeddings using Qdrant vector store
- Supports both vector search and BM25 retrieval
- Smart index: tracks file changes to avoid reprocessing unchanged files
- Documents directory: ~/.btliu/data/documents (global)
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

from langchain_community.document_loaders import (
    JSONLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from btliu.config.paths import get_process_hash_index

# Import paths module for multi-tier data directory support
try:
    from ..config import paths
except ImportError:
    # Fallback if paths module not available
    paths = None

logger = logging.getLogger(__name__)


class DocumentManager:
    """Document manager with Qdrant-based storage and retrieval.

    Supports loading, splitting, indexing, and retrieving documents
    with multiple file format support.
    """

    # Loader mapping for supported file types
    LOADERS: dict[str, Callable] = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": lambda p: TextLoader(p, encoding="utf-8"),
        ".json": lambda p: JSONLoader(p, jq_schema=".", text_content=False),
    }

    def __init__(self, config: Any) -> None:
        """Initialize DocumentManager with Qdrant backend.

        Args:
            config: Configuration object with access methods:
                - vector_store.qdrant_url: Qdrant server URL (default: :memory:)
                - vector_store.collection_name: Collection name
                - vector_store.similarity_threshold: Cosine threshold (default: None)
                - vector_store.max_retrieved_docs: Max docs to return (default: 4)
                - document_processing.chunk_size: Text chunk size (default: 1000)
                - document_processing.chunk_overlap: Chunk overlap (default: 200)
                - document_processing.hash_index_file: Path to hash index file
                - embedding(): Method returning embedding model

        Note:
            Documents directory is managed by paths.py module: ~/.btliu/data/documents
        """
        self.config = config

        # Use paths module for global documents directory
        self.data_dir = paths.get_documents_dir()
        self.hash_index = get_process_hash_index()

        # Document processing configuration
        self.chunk_size = config.get("document_processing.chunk_size", 1000)
        self.chunk_overlap = config.get("document_processing.chunk_overlap", 200)
        self.collection_name = config.get("vector_store.collection_name", "documents")

        # Retrieval configuration
        # similarity_threshold: Cosine distance threshold (lower = more strict)
        # - Qdrant uses distance: 0.0 = identical, 1.0 = unrelated
        # - Converted to relevance score by LangChain: 1.0 - distance
        # - Set to None to disable threshold filtering
        self.similarity_threshold = config.get(
            "vector_store.similarity_threshold", None
        )
        self.default_k = config.get("retrieval.default_k", 6)

        # Initialize Qdrant client
        qdrant_url = config.get("vector_store.qdrant_url", ":memory:")
        if qdrant_url == ":memory:":
            self._client = QdrantClient(":memory:")
        else:
            self._client = QdrantClient(url=qdrant_url)

        # Initialize embeddings
        embedding = config.embedding()

        # Create or use existing collection
        self._ensure_collection(embedding)

        # Initialize vector store
        self._vector_store = QdrantVectorStore(
            client=self._client,
            collection_name=self.collection_name,
            embedding=embedding,
        )

    def _ensure_collection(self, embedding: Any) -> None:
        """Create Qdrant collection if it doesn't exist.

        Args:
            embedding: Embedding model instance
        """
        if not self._client.collection_exists(self.collection_name):
            # Get vector dimension from embedding model
            vector_size = len(embedding.embed_query("sample"))
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Created collection: {self.collection_name}")

    @property
    def vector_store(self) -> QdrantVectorStore:
        """Get vector store instance.

        Returns:
            QdrantVectorStore instance
        """
        return self._vector_store

    def _load_document(self, file_path: str) -> list[Document]:
        """Load single document by file type.

        Args:
            file_path: Path to document file

        Returns:
            List of loaded documents
        """
        ext = Path(file_path).suffix.lower()
        if ext not in self.LOADERS:
            logger.warning(f"Unsupported file format: {ext}")
            return []

        try:
            loader = self.LOADERS[ext](file_path)
            docs = loader.load()

            # Clean text content to remove invalid Unicode characters and noise
            for doc in docs:
                if hasattr(doc, "page_content"):
                    doc.page_content = self._clean_text(doc.page_content)

            # Add source metadata if not present
            for doc in docs:
                if "source" not in doc.metadata:
                    doc.metadata["source"] = file_path

            logger.info(f"Loaded {Path(file_path).name}: {len(docs)} chunks")
            return docs

        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return []

    def _load_all_documents(self) -> list[Document]:
        """Load all documents from data directory.

        Returns:
            List of all loaded documents
        """
        if not self.data_dir.exists():
            logger.warning(f"Data directory not found: {self.data_dir}")
            return []

        all_docs = []
        for file_path in self.data_dir.rglob("*"):
            if file_path.is_file():
                docs = self._load_document(str(file_path))
                all_docs.extend(docs)

        logger.info(f"Total documents loaded: {len(all_docs)}")
        return all_docs

    def _split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents using RecursiveCharacterTextSplitter.

        Args:
            documents: List of documents to split

        Returns:
            List of split documents
        """
        if not documents:
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        return splitter.split_documents(documents)

    def _clean_text(self, text: str) -> str:
        """Clean text by removing invalid Unicode characters and noise.

        Args:
            text: Input text that may contain invalid characters

        Returns:
            Cleaned text with only valid UTF-8 characters
        """
        # Remove surrogate pairs (invalid UTF-16 surrogates in UTF-8)
        text = re.sub(r"[\ud800-\udfff]", "", text)

        # Remove other non-printable/control characters (except common whitespace)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # Remove emojis and other symbols that cause encoding issues
        text = re.sub(r"[\U0001f300-\U0001f9ff]", "", text)  # emojis
        text = re.sub(r"[\U00002600-\U000027bf]", "", text)  # misc symbols
        text = re.sub(r"[\U0001f000-\U0001f02f]", "", text)  # additional symbols

        # Remove URLs
        text = re.sub(r"http\S+", "", text)

        # Normalize whitespace (remove extra spaces)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute hash of file for change detection.

        Args:
            file_path: Path to the file

        Returns:
            MD5 hash (first 16 characters)
        """
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:16]

    def _smart_index(self) -> tuple[bool, set[str], set[str]]:
        """Check which documents changed using hash comparison.

        Returns:
            Tuple of (has_changed, added_files, removed_files):
                - has_changed: True if any file changed
                - added_files: Set of new or modified file paths
                - removed_files: Set of deleted file paths
        """
        hash_file = self.hash_index

        # Scan current files
        current_files = set()
        current_hashes = {}
        for file_path in self.data_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix in self.LOADERS:
                path_str = str(file_path)
                current_files.add(path_str)
                current_hashes[path_str] = self._compute_file_hash(path_str)

        # Load old hash index
        old_hashes = {}
        if hash_file.exists():
            try:
                with open(hash_file) as f:
                    old_hashes = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to read hash index: {e}")

        # Calculate differences
        added_files = current_files - set(old_hashes.keys())
        removed_files = set(old_hashes.keys()) - current_files

        # Detect modified files
        for path, new_hash in current_hashes.items():
            if path in old_hashes and old_hashes[path] != new_hash:
                added_files.add(path)

        has_changed = bool(added_files or removed_files)

        # Always ensure hash index file exists
        hash_file.parent.mkdir(parents=True, exist_ok=True)
        with open(hash_file, "w") as f:
            json.dump(current_hashes, f, indent=2)

        return has_changed, added_files, removed_files

    def del_documents(self, file_path: str) -> bool:
        """Delete all documents with matching source metadata.

        Args:
            file_path: Source file path to delete from vector store

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        logger.info(f"→ Deleting: {file_path}", extra={"color": "info"})

        try:
            filter_obj = Filter(
                must=[
                    FieldCondition(
                        key="metadata.source", match=MatchValue(value=file_path)
                    )
                ]
            )

            self._client.delete(
                collection_name=self.collection_name,
                points_selector=filter_obj,
            )

            # Remove from hash index
            if self.hash_index.exists():
                try:
                    with open(self.hash_index) as f:
                        hashes = json.load(f)
                    if file_path in hashes:
                        del hashes[file_path]
                        with open(self.hash_index, "w") as f:
                            json.dump(hashes, f, indent=2)
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Failed to update hash index: {e}")

            logger.info(f"✓ Deleted: {file_path}", extra={"color": "success"})
            return True
        except Exception as e:
            logger.error(f"✗ Failed to delete: {file_path}", extra={"color": "error"})
            return False

    def reindex(self, documents: list[Document] | None = None) -> None:
        """Reindex all documents by clearing and reloading.

        Args:
            documents: Optional list of documents. If None, loads from data_dir
        """
        logger.info("Starting reindex...")

        # Load documents if not provided
        if documents is None:
            documents = self._load_all_documents()

        if not documents:
            logger.warning("No documents to reindex")
            return

        # Split documents
        split_docs = self._split_documents(documents)

        # Clear existing collection
        try:
            self._client.delete_collection(self.collection_name)
            # Recreate empty collection
            embedding = self.config.embedding()
            self._ensure_collection(embedding)
        except Exception as e:
            logger.warning(f"Could not delete collection: {e}")

        # Add documents to vector store
        self._vector_store.add_documents(split_docs)
        logger.info(f"Reindex complete: {len(split_docs)} chunks stored")

    def add_documents(self, documents: list[Document] | None = None) -> None:
        """Add or update documents incrementally.

        When documents is None, uses smart index to detect changes and only
        processes new or modified files.

        Args:
            documents: Optional list of documents. If None, loads from data_dir
        """
        logger.info("→ Adding documents to vector store...", extra={"color": "info"})

        if documents is None:
            # Use smart index to detect new/modified files
            has_changed, added_files, _ = self._smart_index()
            if not has_changed or not added_files:
                logger.info(
                    "→ No document changes detected", extra={"color": "warning"}
                )
                return

            # Process only new/modified files
            for file_path in added_files:
                logger.info(f"  Processing: {file_path}", extra={"color": "info"})
                docs = self._load_document(file_path)
                if docs:
                    split_docs = self._split_documents(docs)
                    self._vector_store.add_documents(split_docs)
            logger.info(
                f"✓ Added {len(added_files)} documents", extra={"color": "success"}
            )
            return

        # Original logic: process provided documents
        if not documents:
            logger.warning("! No documents to add", extra={"color": "warning"})
            return

        # Split and add documents
        split_docs = self._split_documents(documents)
        self._vector_store.add_documents(split_docs)

        logger.info(
            f"✓ Added {len(split_docs)} document chunks", extra={"color": "success"}
        )

    def get_stats(self) -> dict[str, Any]:
        """Get collection statistics.

        Returns:
            Dict with collection metadata and document count
        """
        collection_info = self._client.get_collection(self.collection_name)
        return {
            "status": "initialized",
            "document_count": collection_info.points_count,
            "collection_name": self.collection_name,
            "vector_size": collection_info.config.params.vectors.size,
            "distance_metric": collection_info.config.params.vectors.distance.value,
        }

    def get_retriever(
        self,
        retriever_type: str = "basic",
        k: int | None = None,
        documents: list[Document] | None = None,
    ) -> Any:
        """Get retriever by type.

        Args:
            retriever_type: Type of retriever ("basic" or "bm25")
                - "basic": Vector similarity search with optional threshold filtering
                - "bm25": BM25 keyword-based search
            k: Number of docs to retrieve (default: max_retrieved_docs from config)
            documents: Required for BM25 retriever (loaded from data dir if None)

        Returns:
            Retriever instance (VectorStoreRetriever or BM25Retriever)

        Raises:
            ValueError: If unsupported retriever_type or no documents for BM25

        Note:
            Basic retriever configuration:
            - Uses similarity_score_threshold if similarity_threshold is set
            - Threshold acts as distance filter (only return docs with distance < threshold)
            - Lower threshold = more strict (fewer results)
        """
        if retriever_type == "basic":
            # Use configured k if not provided
            if k is None:
                k = self.default_k

            # Build retriever kwargs
            retriever_kwargs = {"k": k}

            # Add similarity threshold filtering if configured
            if self.similarity_threshold is not None:
                # search_type="similarity_score_threshold" filters by distance
                # score_threshold is the maximum distance allowed
                retriever_kwargs["search_type"] = "similarity_score_threshold"
                retriever_kwargs["search_kwargs"] = {
                    "score_threshold": self.similarity_threshold
                }

            return self._vector_store.as_retriever(**retriever_kwargs)

        if retriever_type == "bm25":
            # BM25 requires documents loaded from data directory
            if documents is None:
                documents = self._load_all_documents()
            if not documents:
                raise ValueError("No documents available for BM25 retriever")
            return BM25Retriever.from_documents(documents)

        raise ValueError(f"Unsupported retriever type: {retriever_type}")
