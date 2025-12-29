"""Exception hierarchy"""


class RAGError(Exception):
    """Base exception for RAG system"""

    pass


class DocumentError(RAGError):
    """Document processing errors"""

    pass


class RetrievalError(RAGError):
    """Retrieval errors"""

    pass


class ConfigurationError(RAGError):
    """Configuration errors"""

    pass
