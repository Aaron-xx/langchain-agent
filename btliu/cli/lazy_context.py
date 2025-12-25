"""Lazy initialization context for instant CLI startup.

This module provides a lazy-loading runtime context that allows the CLI
to start instantly (< 100ms) while heavy components (Qdrant, ML models)
initialize in the background.

Example:
    lazy_ctx = get_lazy_context()
    lazy_ctx.start_background_init()

    # CLI is now ready to accept input

    # When user makes a query:
    await lazy_ctx.wait_ready()  # Wait if not ready
    context = lazy_ctx.get()      # Get the full context
"""
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LazyContext:
    """Lazy-initialized runtime context for instant CLI startup.

    This class manages background initialization of heavy components
    while allowing the CLI to start immediately.

    Attributes:
        _ready: Whether background initialization is complete
        _init_task: The async task running initialization
        _context: The fully initialized RuntimeContext
        _doc_monitor: The document monitor service (if enabled)
    """

    def __init__(self) -> None:
        """Initialize lazy context (instant, no heavy imports)."""
        self._ready = False
        self._init_task: Optional[asyncio.Task] = None
        self._context = None
        self._doc_monitor = None
        self._init_error: Optional[Exception] = None

    def is_ready(self) -> bool:
        """Check if background initialization is complete.

        Returns:
            True if initialization is complete and successful
        """
        return self._ready

    def get_error(self) -> Optional[Exception]:
        """Get initialization error if any.

        Returns:
            The exception that occurred during initialization, or None
        """
        return self._init_error

    async def wait_ready(self, timeout: float = 30.0) -> None:
        """Wait for background initialization to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Raises:
            RuntimeError: If initialization not started or timed out
            Exception: If initialization failed with an error
        """
        if self._init_task is None:
            raise RuntimeError("Initialization not started")

        try:
            await asyncio.wait_for(self._init_task, timeout=timeout)
        except asyncio.TimeoutError:
            raise RuntimeError(f"Initialization timeout after {timeout}s")
        except Exception:
            # Re-raise any initialization error
            if self._init_error:
                raise self._init_error
            raise

    def get(self):
        """Get the initialized context.

        Returns:
            The RuntimeContext instance

        Raises:
            RuntimeError: If context is not ready yet
        """
        if not self._ready:
            raise RuntimeError(
                "Context not ready, call wait_ready() first"
            )
        return self._context

    async def _initialize(self) -> None:
        """Background initialization task.

        This runs in an async task and loads all heavy components:
        - Config (with environment variable substitution)
        - DocumentManager (Qdrant connection, collection setup)
        - RuntimeContext
        - DocumentMonitorService (if enabled)

        Heavy imports happen here, not at CLI startup.
        """
        try:
            logger.info("Background initialization started")

            # Heavy imports happen here (in background)
            from btliu.config import get_config
            from btliu.tools import DocumentManager
            from btliu.common import RuntimeContext

            # Load configuration
            config = get_config()
            logger.info("Config loaded")

            # Initialize document manager
            doc_manager = DocumentManager(config)
            logger.info("DocumentManager initialized")

            # Create runtime context
            self._context = RuntimeContext(config=config, doc_manager=doc_manager)
            logger.info("RuntimeContext created")

            # Start document monitor if enabled
            if config.get('document_monitor.enabled', False):
                from btliu.services import DocumentMonitorService
                self._doc_monitor = DocumentMonitorService(self._context)
                # Check if start() is async or sync
                import asyncio
                if asyncio.iscoroutinefunction(self._doc_monitor.start):
                    await self._doc_monitor.start()
                else:
                    self._doc_monitor.start()
                logger.info("DocumentMonitor started")

            self._ready = True
            logger.info("Background initialization complete")

        except Exception as e:
            logger.error(f"Background initialization failed: {e}")
            self._init_error = e
            # Don't set _ready=True on error

    def start_background_init(self) -> None:
        """Start background initialization.

        Creates an async task that initializes all heavy components
        in the background. Safe to call multiple times.
        """
        if self._init_task is not None:
            logger.debug("Background initialization already started")
            return

        logger.info("Starting background initialization")
        self._init_task = asyncio.create_task(self._initialize())

    async def cleanup(self) -> None:
        """Clean up resources.

        Stops document monitor if running.
        """
        if self._doc_monitor:
            await self._doc_monitor.stop()
            logger.info("DocumentMonitor stopped")


# Global singleton
_lazy_context: Optional[LazyContext] = None


def get_lazy_context() -> LazyContext:
    """Get or create the global lazy context.

    Returns:
        The global LazyContext singleton
    """
    global _lazy_context
    if _lazy_context is None:
        _lazy_context = LazyContext()
    return _lazy_context


__all__ = ['LazyContext', 'get_lazy_context']
