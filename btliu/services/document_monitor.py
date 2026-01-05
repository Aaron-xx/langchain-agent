"""Document monitoring service for automatic file change detection and processing."""

import fnmatch
import logging
import os
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from btliu.config import paths

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Layer
# ============================================================================


@dataclass
class MonitorConfig:
    """Configuration for document monitoring service."""

    enabled: bool = True
    debounce_time: float = 2.0
    log_events: bool = True
    ignore_patterns: List[str] = field(
        default_factory=lambda: ["*.tmp", "*.swp", "~*", ".DS_Store", "*.bak"]
    )


# ============================================================================
# Event Filter Layer - Extensible filtering mechanism
# ============================================================================


class EventFilter(ABC):
    """Base class for event filters."""

    @abstractmethod
    def should_ignore(self, file_path: str) -> bool:
        """Check if file should be ignored."""
        pass


class PatternEventFilter(EventFilter):
    """Filter based on file pattern matching."""

    def __init__(self, patterns: List[str]):
        self._compiled_patterns = [re.compile(fnmatch.translate(p)) for p in patterns]

    def should_ignore(self, file_path: str) -> bool:
        filename = os.path.basename(file_path)
        return any(p.match(filename) for p in self._compiled_patterns)


# ============================================================================
# Event Hooks - For extensibility
# ============================================================================


@dataclass
class EventHooks:
    """Event hooks for monitoring lifecycle events."""

    on_before_process: Optional[Callable[[set, set], None]] = None
    on_after_process: Optional[Callable[[], None]] = None
    on_error: Optional[Callable[[Exception], None]] = None


# ============================================================================
# File System Event Handler
# ============================================================================


class FileChangeHandler(FileSystemEventHandler):
    """Handles file system events with dependency injection."""

    def __init__(
        self,
        on_change: Callable[[str, str], None],
        filter: EventFilter,
        log_events: bool,
    ):
        super().__init__()
        self.on_change = on_change
        self.filter = filter
        self.log_events = log_events

    def on_any_event(self, event):
        """Handle all file system events with unified logic."""
        if event.is_directory:
            return

        if self.filter.should_ignore(event.src_path):
            return

        event_type = None
        if event.event_type == "created":
            event_type = "created"
        elif event.event_type == "deleted":
            event_type = "deleted"

        if event_type and self.log_events:
            logger.debug(f"File {event_type}: {event.src_path}")

        if event_type:
            self.on_change(event.src_path, event_type)


# ============================================================================
# Document Update Queue - Debouncing and retry
# ============================================================================


class DocumentUpdateQueue:
    """Manages asynchronous document update processing with debouncing."""

    def __init__(
        self,
        doc_manager,
        config: MonitorConfig,
        filter: Optional[EventFilter] = None,
        hooks: Optional[EventHooks] = None,
    ):
        self.doc_manager = doc_manager
        self.config = config
        self.filter = filter or PatternEventFilter(config.ignore_patterns)
        self.hooks = hooks or EventHooks()

        self._created_files: set[str] = set()
        self._deleted_files: set[str] = set()
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

        # Progress tracking state
        self._progress_state: dict[str, Any] = {
            "enabled": False,
            "current_task": None,
            "processed": [],
            "failed": [],
            "total": 0,
        }

    def enqueue_change(self, file_path: str, event_type: str):
        """Enqueue a file change event with debouncing."""
        with self._lock:
            if event_type == "created":
                self._created_files.add(file_path)
                self._deleted_files.discard(file_path)
            elif event_type == "deleted":
                self._deleted_files.add(file_path)
                self._created_files.discard(file_path)

            # Reset timer
            if self._timer:
                self._timer.cancel()

            self._timer = threading.Timer(
                self.config.debounce_time, self._trigger_processing
            )
            self._timer.start()

    def _trigger_processing(self):
        """Trigger processing in a separate thread."""
        threading.Thread(target=self._process_pending, daemon=True).start()

    # -------------------------------------------------------------------------
    # Hash Index Management
    # -------------------------------------------------------------------------

    def _get_hash_index_path(self):
        """Get path to hash index file."""
        from btliu.config import paths

        return paths.get_process_hash_index()

    def _load_hash_index(self) -> dict:
        """Load current hash index."""
        import json

        hash_file = self._get_hash_index_path()
        if not hash_file.exists():
            return {}
        try:
            with open(hash_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load hash index: {e}")
            return {}

    def _save_hash_index(self, hashes: dict) -> None:
        """Save hash index to file."""
        import json

        hash_file = self._get_hash_index_path()
        hash_file.parent.mkdir(parents=True, exist_ok=True)
        with open(hash_file, "w") as f:
            json.dump(hashes, f, indent=2)

    def _add_to_hash_index(self, file_paths: set[str]) -> None:
        """Add files to hash index."""
        if not file_paths:
            return
        hashes = self._load_hash_index()
        for file_path in file_paths:
            if os.path.exists(file_path):
                hashes[file_path] = self._compute_file_hash(file_path)
        self._save_hash_index(hashes)

    def _remove_from_hash_index(self, file_paths: set[str]) -> None:
        """Remove files from hash index."""
        if not file_paths:
            return
        hashes = self._load_hash_index()
        for file_path in file_paths:
            hashes.pop(file_path, None)
        self._save_hash_index(hashes)

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute MD5 hash of file."""
        import hashlib

        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:16]

    def _process_pending(self):
        """Process all pending file changes with debouncing and progress tracking."""
        with self._lock:
            if not self._created_files and not self._deleted_files:
                return

            created = list(self._created_files)
            deleted = list(self._deleted_files)
            self._created_files.clear()
            self._deleted_files.clear()

            # Set up task state for progress tracking
            if created or deleted:
                self._progress_state["current_task"] = {
                    "type": "add" if created else "del",
                    "files": list(created) + list(deleted),
                }
                self._progress_state["processed"] = []
                self._progress_state["failed"] = []
                self._progress_state["total"] = len(created) + len(deleted)

        # Auto-print processing start message (outside lock)
        if created or deleted:
            total = len(created) + len(deleted)
            import sys

            if created:
                print(f"正在处理 {total} 个文件...", file=sys.stderr)
            else:
                print(f"正在删除 {total} 个文件...", file=sys.stderr)

        try:
            if self.hooks.on_before_process:
                self.hooks.on_before_process(set(created), set(deleted))

            logger.debug(f"Processing {len(created)} created, {len(deleted)} deleted")

            # Handle deleted files
            for file_path in deleted:
                self._update_progress(file_path, "processing")
                if self.doc_manager.del_documents(file_path):
                    self._update_progress(file_path, "success")
                else:
                    self._update_progress(file_path, "failed")
            if deleted:
                self._remove_from_hash_index(set(deleted))

            # Handle created files
            if created:
                for file_path in created:
                    self._update_progress(file_path, "processing")
                self.doc_manager.add_documents(file_paths=set(created))
                # Batch mark as success (add_documents handles individual failures)
                for file_path in created:
                    self._update_progress(file_path, "success")
                self._add_to_hash_index(set(created))

            logger.debug("Successfully processed file changes")

            if self.hooks.on_after_process:
                self.hooks.on_after_process()

        except Exception as e:
            logger.error(f"Failed to process document updates: {e}", exc_info=True)
            if self.hooks.on_error:
                self.hooks.on_error(e)

        finally:
            # Always clear task state, even if exception occurred
            should_print = False
            processed = 0
            total = 0
            with self._lock:
                if (
                    self._progress_state["current_task"]
                    and self._progress_state["enabled"]
                ):
                    should_print = True
                    processed = len(self._progress_state["processed"])
                    total = self._progress_state["total"]
                self._progress_state["current_task"] = None

            if should_print:
                import sys

                print(
                    f"\r✓ 完成 {processed}/{total} 文件{' ' * 30}",
                    file=sys.stderr,
                )

    def stop(self):
        """Stop the update queue and process remaining changes."""
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None

        # Process remaining changes
        if self._created_files or self._deleted_files:
            self._process_pending()

    # -------------------------------------------------------------------------
    # Progress Tracking
    # -------------------------------------------------------------------------

    def show_progress_status(self) -> None:
        """Display current progress status - called by /embeding command."""
        # Acquire lock only to read data, release before printing
        with self._lock:
            if not self._progress_state["current_task"]:
                print("当前无文档处理任务")
                return

            task_type = self._progress_state["current_task"]["type"]
            total = self._progress_state["total"]
            processed = len(self._progress_state["processed"])
            failed = len(self._progress_state["failed"])

        # Print without holding lock to avoid blocking other threads
        if task_type == "add":
            if processed == 0 and total > 0:
                print(f"正在处理 {total} 个文件...")
            elif processed < total:
                print(f"文档添加: {processed}/{total} 文件")
            else:
                print(f"文档添加: {processed}/{total} 文件")
        else:
            if processed == 0 and total > 0:
                print(f"正在处理 {total} 个文件...")
            elif processed < total:
                print(f"文档删除: {processed}/{total} 文件")
            else:
                print(f"文档删除: {processed}/{total} 文件")

        if failed:
            print(f"失败: {failed} 个文件")

        if processed >= total and total > 0:
            success_count = processed - failed
            print(f"✓ 完成！成功: {success_count}, 失败: {failed}")

    def toggle_progress(self) -> bool:
        """Toggle progress display on/off.

        Returns:
            New enabled state (True if enabled, False if disabled)
        """
        with self._lock:
            self._progress_state["enabled"] = not self._progress_state["enabled"]
            return self._progress_state["enabled"]

    def _update_progress(self, file_path: str, status: str) -> None:
        """Update progress state during file processing.

        Args:
            file_path: Path to the file being processed
            status: Status type - "processing", "success", or "failed"
        """
        # For success/failed, just update state
        if status in ("success", "failed"):
            with self._lock:
                if not self._progress_state["current_task"]:
                    return
                if status == "success":
                    self._progress_state["processed"].append(file_path)
                else:
                    self._progress_state["failed"].append(file_path)
            return

        # For processing, read state and print without holding lock
        should_print = False
        processed = 0
        total = 0

        with self._lock:
            if not self._progress_state["current_task"]:
                return
            if self._progress_state["enabled"] and status == "processing":
                should_print = True
                processed = len(self._progress_state["processed"])
                total = self._progress_state["total"]

        if should_print:
            import sys
            from pathlib import Path

            print(
                f"\r→ [{processed + 1}/{total}] {Path(file_path).name}",
                end="",
                flush=True,
                file=sys.stderr,
            )


# ============================================================================
# Main Monitor Service
# ============================================================================


class DocumentMonitorService:
    """Main service for monitoring document directory changes."""

    def __init__(self, context):
        """Initialize monitor service with runtime context."""
        self.context = context
        self.config = self._load_config()
        self._setup_logging()

        self.observer: Optional[Observer] = None
        self.update_queue: Optional[DocumentUpdateQueue] = None
        self._running = False

    def _load_config(self) -> MonitorConfig:
        """Load configuration from context config."""
        config_dict = self.context["config"].get("document_monitor", {})
        return MonitorConfig(
            enabled=config_dict.get("enabled", True),
            debounce_time=config_dict.get("debounce_time", 2.0),
            log_events=config_dict.get("log_events", True),
            ignore_patterns=config_dict.get(
                "ignore_patterns", ["*.tmp", "*.swp", "~*", ".DS_Store", "*.bak"]
            ),
        )

    def _setup_logging(self):
        """Configure logging level based on global config."""
        log_level_str = self.context["config"].get("log_level", "WARNING").upper()
        log_level = getattr(logging, log_level_str, logging.WARNING)
        logger.setLevel(log_level)

    def start(self) -> bool:
        """Start the document monitoring service. Returns True if successful."""
        if not self.config.enabled:
            logger.debug("Document monitoring is disabled")
            return False

        if self._running:
            logger.debug("Document monitor is already running")
            return True

        try:
            documents_dir = paths.get_documents_dir()
            hash_index = paths.get_process_hash_index()

            # Create directories if they don't exist
            documents_dir.mkdir(parents=True, exist_ok=True)
            hash_index.parent.mkdir(parents=True, exist_ok=True)

            # Create hash index if it doesn't exist
            if not hash_index.exists():
                hash_index.touch()
                logger.info(f"Created hash index file: {hash_index}")

            # Initialize components
            self.update_queue = DocumentUpdateQueue(
                self.context["doc_manager"], self.config
            )

            event_filter = PatternEventFilter(self.config.ignore_patterns)
            event_handler = FileChangeHandler(
                lambda path, etype: self.update_queue.enqueue_change(path, etype),
                event_filter,
                self.config.log_events,
            )

            # Set up observer
            self.observer = Observer()
            self.observer.schedule(event_handler, str(documents_dir), recursive=True)
            self.observer.schedule(
                event_handler, str(hash_index.parent), recursive=True
            )

            self.observer.start()
            self._running = True

            logger.debug(f"Document monitoring started for: {documents_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to start document monitor: {e}", exc_info=True)
            self._cleanup()
            return False

    def stop(self):
        """Stop the document monitoring service."""
        if not self._running:
            return

        logger.debug("Stopping document monitoring service...")
        self._running = False

        if self.update_queue:
            self.update_queue.stop()

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
            "pending_deleted": deleted_count,
        }
