#!/usr/bin/env python3
"""CLI module for btliu with instant startup.

Background initialization runs in a separate thread, so CLI starts
instantly (< 100ms) and initialization happens in parallel.

Refactored to use class-based encapsulation for better state management
and resource lifecycle.
"""

import asyncio
import logging
import sys
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from typing import Any, Optional

from prompt_toolkit import PromptSession, print_formatted_text, ANSI
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory

# Try to import paths module for working directory display (lightweight)
try:
    from ..config import paths as paths_module
except (ImportError, ValueError):
    paths_module = None

logger = logging.getLogger(__name__)


# ============================================================================
# COMPLETER
# ============================================================================


class CLIDynamicCompleter(Completer):
    """Dynamic command completer for CLI commands."""

    def get_completions(self, document, complete_event):
        cmds = ["/ucagent", "/rag", "/save", "/restore", "/exit", "/help"]
        for w in cmds:
            if w.startswith(document.text):
                yield Completion(w, start_position=-len(document.text))


# ============================================================================
# CLI APPLICATION CLASS
# ============================================================================


class CLIApplication:
    """Encapsulates all CLI state and resource management.

    This class manages:
    - Background initialization of context
    - Database connection pool lifecycle
    - Document monitor lifecycle
    - User session state
    - Query execution

    All resources are properly cleaned up on exit via the cleanup() method.
    """

    # Class-level constants
    PROMPT_COMPLETER = CLIDynamicCompleter()
    HISTORY_FILE = os.path.expanduser("~/.cli_history")
    SPINNER_CHARS = cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠏")

    def __init__(self):
        """Initialize CLI application with all state."""
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)

        # Core state
        self._context: Optional[dict] = None
        self._init_future: Optional[Any] = None
        self._doc_monitor: Optional[Any] = None
        self._db_pool: Optional[Any] = None
        self._store: Optional[Any] = None

        # CLI state
        self.mode: str = "ucagent"
        self.setup_done: bool = False
        # Tracks if AI started responding: used to stop spinner when first token arrives
        self.first_token_received: bool = False
        self.current_task: Optional[asyncio.Task] = None

        # Async utilities
        self.stdout_lock: asyncio.Lock = asyncio.Lock()
        self.prompt_session: PromptSession = PromptSession(
            history=FileHistory(self.HISTORY_FILE), completer=self.PROMPT_COMPLETER
        )

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def start_background_init(self):
        """Start background initialization in a separate thread."""
        self._init_future = self._executor.submit(self._init_context)

    def _init_context(self) -> dict:
        """Initialize context in background thread.

        This runs in a separate thread, so CLI is not blocked.
        All heavy imports happen here.

        Note: AsyncConnectionPool is NOT initialized here because it requires
        an async event loop. It will be initialized after the event loop is running.
        """
        from btliu.config import get_config
        from btliu.tools import DocumentManager
        from btliu.common import RuntimeContext
        from btliu.services import DocumentMonitorService

        config = get_config()
        doc_manager = DocumentManager(config)

        # Create context without async components
        context = RuntimeContext(
            config=config,
            doc_manager=doc_manager,
            store=None,  # Deferred init
            thread_id=str(uuid.uuid4()),
        )

        # Start document monitor if enabled
        if config.get("document_monitor", {}).get("enabled", False):
            try:
                self._doc_monitor = DocumentMonitorService(context)
                self._doc_monitor.start()
            except Exception as e:
                logger.error(f"Failed to start document monitor: {e}")

        self._context = context
        return context

    def ensure_context(self) -> dict:
        """Ensure context is initialized, waits if necessary."""
        if self._context is None:
            self._context = self._init_future.result()
        return self._context

    async def initialize_database(self) -> None:
        """Initialize database connection pool and store.

        This must be called from within an async event loop.
        Only runs once (self._db_pool check).
        """
        if self._db_pool is not None:
            return  # Already initialized

        context = self.ensure_context()
        config = context.get("config")
        db_uri = config.get("postgresql_uri", None) if config else None

        if not db_uri:
            return

        from psycopg_pool import AsyncConnectionPool
        from langgraph.store.postgres.aio import AsyncPostgresStore

        self._db_pool = AsyncConnectionPool(
            conninfo=db_uri, kwargs={"autocommit": True}, open=False
        )
        await self._db_pool.open()

        self._store = AsyncPostgresStore(self._db_pool)
        context["store"] = self._store

        # Run setup to create tables
        try:
            await self._store.setup()
            print_formatted_text(ANSI("\x1b[32m✓ Database initialized\x1b[0m\n"))
        except Exception as e:
            print_formatted_text(
                ANSI(f"\x1b[33m! Database setup skipped: {e}\x1b[0m\n")
            )

    # ========================================================================
    # RESOURCE CLEANUP
    # ========================================================================

    async def cleanup(self) -> None:
        """Clean up all resources in the correct order.

        This method should be called from all exit points:
        - Normal exit (/exit command)
        - Keyboard interrupt (Ctrl+C)
        - EOF (Ctrl+D)
        """
        if self._doc_monitor is not None:
            try:
                await self._doc_monitor.stop()
            except Exception as e:
                logger.warning(f"Error stopping document monitor: {e}")

        if self._db_pool is not None:
            try:
                await self._db_pool.close()
            except Exception as e:
                logger.warning(f"Error closing database pool: {e}")

        self._executor.shutdown(wait=False)

        logger.info("CLI application cleaned up")

    # ========================================================================
    # QUERY EXECUTION
    # ========================================================================

    async def stream_output(self, app, payload: dict) -> None:
        """Stream output from an application.

        Args:
            app: Application instance (RAGApp or UcagentApp)
            payload: Query payload with messages

        Note:
            Handles (token, metadata) tuples from LangGraph stream_mode="messages"
            and filters message types based on MESSAGE_TYPE_FILTER config.
        """
        # Message type filter configuration: True=show, False=hide
        # Based on LangChain/LangGraph message types from official documentation
        MESSAGE_TYPE_FILTER = {
            # Standard LangChain message types
            "ai": True,  # AIMessage - AI complete response
            "AIMessageChunk": True,  # AIMessageChunk - AI streaming tokens
            "assistant": True,  # Generic assistant message
            "tool": False,  # ToolMessage - tool results (hidden)
            "human": False,  # HumanMessage - user input (hidden)
            "system": False,  # SystemMessage - system prompt (hidden)
            "reasoning": False,  # Reasoning messages (hidden by default)
            # Extension interface for future message types:
            # "thinking": False,     # Thinking content blocks
            # "generic": False,      # Generic messages
        }

        async for token in app.astream(payload):
            # Signal spinner to stop: first AI token has arrived
            if not self.first_token_received:
                self.first_token_received = True
            async with self.stdout_lock:
                # LangGraph stream_mode="messages" yields three possible formats:
                # - (Message, metadata): Message object with metadata (handle with filter)
                # - str: Plain text string (write directly)
                # - list: Metadata list (skip)
                if isinstance(token, tuple) and len(token) == 2:
                    msg, metadata = token
                    msg_type = getattr(msg, "type", None)
                    if (
                        msg_type in MESSAGE_TYPE_FILTER
                        and MESSAGE_TYPE_FILTER[msg_type]
                    ):
                        content = getattr(msg, "content", None)
                        if content:
                            sys.stdout.write(content)
                elif isinstance(token, str):
                    sys.stdout.write(token)
                elif isinstance(token, list):
                    pass  # Skip metadata
                sys.stdout.flush()

    async def execute_query(self, line: str) -> None:
        """Execute a user query."""
        # Reset spinner signal for each new query
        self.first_token_received = False
        context = self.ensure_context()

        try:
            payload = {"messages": [{"role": "user", "content": line}]}

            if self.mode == "ucagent":
                from btliu.apps.ucagent_app import UcagentApp

                app = UcagentApp(context, store=self._store)
            else:  # rag or default
                from btliu.apps.rag_app import RAGApp

                app = RAGApp(context, store=self._store)

            await self.stream_output(app, payload)
            print("\n")

        except asyncio.CancelledError:
            print("\ncancelled\n")

    async def show_spinner(self) -> None:
        """Display a spinner animation while a task is running.

        Dual exit conditions:
        1. Task completes (current_task.done())
        2. First AI token arrives (first_token_received=True)
        """
        while self.current_task and not self.current_task.done():
            if self.first_token_received:
                break
            char = next(self.SPINNER_CHARS)
            print(
                f"\x1b[36mGenerating {char}\x1b[0m",
                end="\r",
                file=sys.stderr,
                flush=True,
            )
            await asyncio.sleep(0.1)
        print("\r" + " " * 30 + "\r", end="", file=sys.stderr, flush=True)

    # ========================================================================
    # COMMAND HANDLING
    # ========================================================================

    async def handle_command(self, cmd: str, args: Optional[str]) -> bool:
        """Handle a CLI command.

        Returns True if command was handled (should continue loop),
        False if should exit.
        """
        if cmd in ("ucagent", "rag"):
            self.mode = cmd
            print_formatted_text(ANSI(f"\x1b[33m✓ {self.mode.upper()} mode\x1b[0m\n"))
            return True

        if cmd == "save":
            context = self.ensure_context()
            session_name = args or f"session_{int(time.time())}"
            context["thread_id"] = session_name
            print_formatted_text(
                ANSI(f"\x1b[32m✓ Session saved: {session_name}\x1b[0m\n")
            )
            return True

        if cmd == "restore":
            context = self.ensure_context()
            session_name = args or "default"
            context["thread_id"] = session_name
            print_formatted_text(
                ANSI(f"\x1b[32m✓ Session restored: {session_name}\x1b[0m\n")
            )
            return True

        if cmd in ("exit", "quit"):
            return False  # Signal to exit

        if cmd == "help":
            print_formatted_text(
                ANSI(
                    "\x1b[36mCommands: /ucagent /rag /save /restore /exit /help\x1b[0m"
                )
            )
            return True

        # Unknown command
        return True

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    async def run(self) -> None:
        """Main CLI entry point with instant startup."""
        self.start_background_init()

        print("CLI ready  /ucagent  /rag  /save /restore /help /exit  Ctrl+C cancel\n")

        if paths_module is not None:
            try:
                print(f"Working directory: {paths_module.get_working_dir()}")
                print(f"Config: {paths_module.find_config_path()}\n")
            except Exception:
                pass

        # Main interactive loop
        while True:
            try:
                with patch_stdout():
                    line = await self.prompt_session.prompt_async(
                        ANSI(f"\x1b[36m[{self.mode}]\x1b[0m > ")
                    )
                line = line.strip()

                if not line:
                    continue

                if line.startswith("/"):
                    cmd_parts = line[1:].split(None, 1)
                    cmd = cmd_parts[0].lower()
                    args = cmd_parts[1] if len(cmd_parts) > 1 else None

                    should_continue = await self.handle_command(cmd, args)
                    if not should_continue:
                        break  # Exit requested
                    continue

                if not self.setup_done:
                    await self.initialize_database()
                    self.setup_done = True

                self.current_task = asyncio.create_task(self.execute_query(line))
                spinner_task = asyncio.create_task(self.show_spinner())
                await self.current_task
                spinner_task.cancel()

            except KeyboardInterrupt:
                if self.current_task and not self.current_task.done():
                    self.current_task.cancel()
                else:
                    print("\nbye")
                    break  # Exit on second Ctrl+C

            except EOFError:
                if self.current_task and not self.current_task.done():
                    self.current_task.cancel()
                print("\nbye")
                break

        await self.cleanup()


# ============================================================================
# ENTRY POINT
# ============================================================================


async def cli_main():
    """Entry point for the CLI application."""
    app = CLIApplication()
    try:
        await app.run()
    except Exception as e:
        logger.error(f"CLI error: {e}", exc_info=True)
        await app.cleanup()
        sys.exit(1)
