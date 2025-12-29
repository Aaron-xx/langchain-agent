"""Document monitoring service for automatic file change detection and processing."""
import fnmatch
import logging
import os
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from btliu.config import paths

logger = logging.getLogger(__name__)


@dataclass
class MonitorConfig:
    """Configuration for document monitoring service."""
    enabled: bool = True
    debounce_time: float = 2.0
    batch_size: int = 10
    max_retries: int = 3
    log_events: bool = True
    ignore_patterns: List[str] = None

    def __post_init__(self):
        if self.ignore_patterns is None:
            self.ignore_patterns = ["*.tmp", "*.swp", "~*", ".DS_Store", "*.bak"]


class FileChangeHandler(FileSystemEventHandler):
    """Handles file system events for document monitoring."""

    def __init__(self, update_queue: 'DocumentUpdateQueue', config: MonitorConfig):
        super().__init__()
        self.update_queue = update_queue
        self.config = config

    def _should_ignore(self, file_path: str) -> bool:
        """Check if file should be ignored based on patterns."""
        filename = os.path.basename(file_path)
        for pattern in self.config.ignore_patterns:
            if fnmatch.fnmatch(filename, pattern):
                return True
        return False

    def on_created(self, event):
        """Handle file creation event."""
        if not event.is_directory and not self._should_ignore(event.src_path):
            if self.config.log_events:
                logger.debug(f"File created: {event.src_path}")
            self.update_queue.enqueue_change(event.src_path, 'created')

    def on_deleted(self, event):
        """Handle file deletion event."""
        if not event.is_directory and not self._should_ignore(event.src_path):
            if self.config.log_events:
                logger.debug(f"File deleted: {event.src_path}")
            self.update_queue.enqueue_change(event.src_path, 'deleted')


class DocumentUpdateQueue:
    """Manages asynchronous document update processing with debouncing."""

    def __init__(self, doc_manager, config: MonitorConfig):
        self.doc_manager = doc_manager
        self.config = config
        self._created_files: set[str] = set()
        self._deleted_files: set[str] = set()
        self._processing = False
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def enqueue_change(self, file_path: str, event_type: str):
        """Enqueue a file change event with debouncing."""
        with self._lock:
            if event_type == 'created':
                self._created_files.add(file_path)
                self._deleted_files.discard(file_path)
            elif event_type == 'deleted':
                self._deleted_files.add(file_path)
                self._created_files.discard(file_path)

            # Cancel existing timer if any
            if self._timer:
                self._timer.cancel()

            # Schedule processing after debounce time
            self._timer = threading.Timer(
                self.config.debounce_time,
                self._trigger_processing
            )
            self._timer.start()

    def _trigger_processing(self):
        """Trigger processing in a thread-safe manner."""
        # Run processing in a separate thread to avoid blocking
        threading.Thread(target=self._process_pending, daemon=True).start()

    def _process_pending(self):
        """Process all pending file changes."""
        with self._lock:
            if not self._created_files and not self._deleted_files:
                return

            if self._processing:
                return

            self._processing = True
            created = self._created_files.copy()
            deleted = self._deleted_files.copy()
            self._created_files.clear()
            self._deleted_files.clear()

        try:
            logger.debug(f"Processing {len(created)} created, {len(deleted)} deleted")

            # Handle deleted files
            for file_path in deleted:
                self.doc_manager.del_documents(file_path)

            # Handle created files - add_documents uses smart index internally
            if created:
                self.doc_manager.add_documents()

            logger.debug("Successfully processed file changes")

        except Exception as e:
            logger.error(f"Failed to process document updates: {e}", exc_info=True)

        finally:
            with self._lock:
                self._processing = False

    async def stop(self):
        """Stop the update queue and process any remaining changes."""
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None

        # Process any remaining changes
        if self._created_files or self._deleted_files:
            self._process_pending()


class DocumentMonitorService:
    """Main service for monitoring document directory changes."""

    def __init__(self, context):
        """Initialize monitor service with runtime context."""
        self.context = context
        self.config = self._load_config()

        # Set logger level based on global configuration
        log_level_str = context['config'].get('log_level', 'WARNING').upper()
        log_level = getattr(logging, log_level_str, logging.WARNING)
        logger.setLevel(log_level)

        self.observer: Optional[Observer] = None
        self.update_queue: Optional[DocumentUpdateQueue] = None
        self._running = False

    def _load_config(self) -> MonitorConfig:
        """Load configuration from context config."""
        config_dict = self.context['config'].get('document_monitor', {})

        # Convert to MonitorConfig with defaults
        return MonitorConfig(
            enabled=config_dict.get('enabled', True),
            debounce_time=config_dict.get('debounce_time', 2.0),
            batch_size=config_dict.get('batch_size', 10),
            max_retries=config_dict.get('max_retries', 3),
            log_events=config_dict.get('log_events', True),
            ignore_patterns=config_dict.get('ignore_patterns', [
                "*.tmp", "*.swp", "~*", ".DS_Store", "*.bak"
            ])
        )

    def start(self):
        """Start the document monitoring service."""
        if not self.config.enabled:
            logger.debug("Document monitoring is disabled")
            return

        if self._running:
            logger.debug("Document monitor is already running")
            return

        try:
            # Get documents directory from paths module
            documents_dir = paths.get_documents_dir()
            if not documents_dir.exists():
                logger.warning(f"Documents directory does not exist: {documents_dir}")
                return

            # Create update queue
            self.update_queue = DocumentUpdateQueue(
                self.context['doc_manager'],
                self.config
            )

            # Set up file system observer
            self.observer = Observer()
            event_handler = FileChangeHandler(self.update_queue, self.config)

            # Watch for changes in documents directory
            self.observer.schedule(
                event_handler,
                str(documents_dir),
                recursive=True
            )

            # Start observer
            self.observer.start()
            self._running = True

            logger.debug(f"Document monitoring started for: {documents_dir}")

        except Exception as e:
            logger.error(f"Failed to start document monitor: {e}", exc_info=True)
            self._cleanup()

    async def stop(self):
        """Stop the document monitoring service."""
        if not self._running:
            return

        logger.debug("Stopping document monitoring service...")
        self._running = False

        # Stop update queue and process remaining changes
        if self.update_queue:
            await self.update_queue.stop()

        self._cleanup()
        logger.debug("Document monitoring service stopped")

    def _cleanup(self):
        """Clean up resources."""
        if self.observer:
            try:
                self.observer.stop()
                self.observer.join(timeout=5.0)
            except Exception as e:
                logger.error(f"Error stopping file observer: {e}")
            finally:
                self.observer = None

        self.update_queue = None

    def is_running(self) -> bool:
        """Check if the monitor service is running."""
        return self._running

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the monitor service."""
        created_count = 0
        deleted_count = 0
        if self.update_queue:
            with self.update_queue._lock:
                created_count = len(self.update_queue._created_files)
                deleted_count = len(self.update_queue._deleted_files)

        return {
            "running": self._running,
            "enabled": self.config.enabled,
            "documents_dir": str(paths.get_documents_dir()),
            "debounce_time": self.config.debounce_time,
            "pending_created": created_count,
            "pending_deleted": deleted_count
        }
