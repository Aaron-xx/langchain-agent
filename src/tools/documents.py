"""Document management system with loading, splitting, and storage capabilities."""
import hashlib
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    JSONLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

class DocumentManager:
    """Document manager with loading, splitting, and storage functionality."""

    def __init__(self, config) -> None:
        self.config = config
        self.data_dir = Path(config.get('document_processing.data_dir'))
        self.persist_dir = config.get('vector_store.persist_directory')
        self.hash_index = Path(config.get('document_processing.hash_index_file'))
        self.collection_name = config.get('vector_store.collection_name', 'rag_documents')
        self.chunk_size = config.get('document_processing.chunk_size', 1000)
        self.chunk_overlap = config.get('document_processing.chunk_overlap', 200)

        embedding_model = config.embedding_model
        self._vector_store = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=embedding_model,
            collection_name=self.collection_name
        )

    @property
    def vector_store(self):
        """Get vector store instance."""
        return self._vector_store

    def _load_document(self, file_path: str) -> List[Document]:
        """Load document based on file extension."""
        loaders = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.md': lambda p: TextLoader(p, encoding="utf-8"),
            '.json': lambda p: JSONLoader(p, jq_schema='.', text_content=False),
        }

        ext = Path(file_path).suffix
        if ext not in loaders:
            logger.warning(f"Unsupported file format: {ext}")
            return []

        try:
            loader = loaders[ext](file_path)
            docs = loader.load()
            logger.info(f"Loaded {Path(file_path).name}: {len(docs)} document chunks")
            return docs
        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {e}")
            return []

    def _load_all_documents(self) -> List[Document]:
        """Load all documents from directory."""
        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            return []

        all_docs = []
        for file_path in self.data_dir.rglob('*'):
            if file_path.is_file():
                docs = self._load_document(str(file_path))
                all_docs.extend(docs)

        return all_docs

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents adaptively based on content type.

        Uses MarkdownHeaderTextSplitter for .md files to preserve structure,
        and RecursiveCharacterTextSplitter for other document types.
        """
        if not documents:
            return []

        has_markdown = any(
            Path(doc.metadata.get('source', '')).suffix == '.md'
            for doc in documents
        )

        if has_markdown:
            md_docs = []
            other_docs = []

            for doc in documents:
                if Path(doc.metadata.get('source', '')).suffix == '.md':
                    md_docs.append(doc)
                else:
                    other_docs.append(doc)

            if md_docs:
                md_splitter = MarkdownHeaderTextSplitter(
                    headers_to_split_on=[
                        ("#", "Header 1"),
                        ("##", "Header 2"),
                        ("###", "Header 3"),
                    ]
                )
                md_content = "\n\n".join([doc.page_content for doc in md_docs])
                split_md_docs = md_splitter.split_text(md_content)
                for i, doc in enumerate(split_md_docs):
                    if md_docs:
                        doc.metadata.update(md_docs[0].metadata)
            else:
                split_md_docs = []

            all_docs = split_md_docs + other_docs

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            return splitter.split_documents(all_docs)
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            return splitter.split_documents(documents)

    def _clean_documents(self, documents: List[Document]) -> List[Document]:
        """Clean document content by normalizing Unicode and removing control characters."""
        import re
        import unicodedata

        cleaned = []
        for doc in documents:
            try:
                content = unicodedata.normalize('NFKC', doc.page_content)
                content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
                content = re.sub(r'\s+', ' ', content).strip()
                content = ''.join(
                    c for c in content
                    if c.isprintable() or ord(c) >= 0x4e00
                )

                if len(content) > 10:
                    cleaned.append(Document(
                        page_content=content,
                        metadata=doc.metadata
                    ))
            except Exception as e:
                logger.warning(f"Document cleaning failed: {e}")
                cleaned.append(doc)

        return cleaned

    def _smart_index(self) -> bool:
        """Check if documents need reindexing."""
        index_file = self.hash_index

        documents = self._load_all_documents()

        def doc_hash(doc: Document) -> str:
            content = f"{doc.metadata.get('source', '')}{doc.page_content[:200]}"
            return hashlib.md5(content.encode()).hexdigest()[:16]

        current_hashes = {
            doc.metadata.get('source', f'doc_{i}'): doc_hash(doc)
            for i, doc in enumerate(documents)
        }

        old_hashes = {}
        if index_file.exists():
            try:
                with open(index_file) as f:
                    old_hashes = json.load(f)
            except Exception as e:
                logger.error(f"Failed to read index: {e}")

        has_changed = current_hashes != old_hashes
        if has_changed:
            index_file.parent.mkdir(parents=True, exist_ok=True)
            with open(index_file, 'w') as f:
                json.dump(current_hashes, f)

        return has_changed

    def reindex(self, documents: List[Document]):
        """Reindex documents - public method."""
        logger.info("Starting reindex...")

        clean_docs = self._clean_documents(documents)
        split_docs = self._split_documents(clean_docs)

        self._vector_store.delete_collection()

        self._vector_store.add_documents(split_docs)

        logger.info(f"✅ Reindex complete: {len(split_docs)} document chunks")

    def add_documents(self, documents: List[Document] = None):
        """Add documents with incremental update."""
        if documents is None:
            if not self._smart_index():
                logger.info("✅ Documents unchanged, skipping update")
                return
            documents = self._load_all_documents()

        if documents:
            clean_docs = self._clean_documents(documents)
            split_docs = self._split_documents(clean_docs)

            self._vector_store.add_documents(split_docs)

            logger.info(f"✅ Document update complete: {len(split_docs)} document chunks")

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        collection = self._vector_store._collection
        count = collection.count()

        return {
            "status": "initialized",
            "document_count": count,
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_dir)
        }

    def get_retriever(self, types: str, documents: List[Document] = None):
        """Get retriever by type. Returns vector store retriever for 'basic', BM25Retriever for 'bm25'."""
        if types == "basic":
            return self._vector_store.as_retriever()
        elif types == "bm25":
            return BM25Retriever.from_documents(documents)
        else:
            raise ValueError(f"Unsupported type: {types}")