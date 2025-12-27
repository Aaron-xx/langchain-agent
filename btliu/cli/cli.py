#!/usr/bin/env python3
"""CLI module for btliu with instant startup.

Background initialization runs in a separate thread, so CLI starts
instantly (< 100ms) and initialization happens in parallel.
"""
import asyncio
import sys
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from prompt_toolkit import PromptSession, print_formatted_text, ANSI
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory

# Try to import paths module for working directory display (lightweight)
try:
    from ..config import paths as paths_module
except (ImportError, ValueError):
    paths_module = None


# ============================================================================
# BACKGROUND INITIALIZATION
# ============================================================================

_executor = ThreadPoolExecutor(max_workers=1)
_context = None
_init_future = None
_doc_monitor = None


def _init_context():
    """Initialize context in background thread.

    This runs in a separate thread, so CLI is not blocked.
    All heavy imports happen here.
    """
    from btliu.config import get_config
    from btliu.tools import DocumentManager
    from btliu.common import RuntimeContext
    from btliu.services import DocumentMonitorService

    config = get_config()
    doc_manager = DocumentManager(config)

    # Create checkpointer and store (synchronous object creation)
    checkpointer = None
    store = None
    db_uri = config.get("checkpoint_db_uri", None)

    if db_uri:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from langgraph.store.postgres.aio import AsyncPostgresStore
        store = AsyncPostgresStore.from_conn_string(db_uri)
        checkpointer = AsyncPostgresSaver.from_conn_string(db_uri)

    context = RuntimeContext(
        config=config,
        doc_manager=doc_manager,
        checkpointer=checkpointer,
        store=store,
        thread_id=str(uuid.uuid4())
    )

    # Start document monitor after context is ready (non-blocking, in same thread)
    global _doc_monitor
    if config.get("document_monitor", {}).get("enabled", False):
        try:
            _doc_monitor = DocumentMonitorService(context)
            _doc_monitor.start()
        except Exception as e:
            logger.error(f"Failed to start document monitor: {e}")

    return context


def _ensure_context():
    """Ensure context is initialized, waits if necessary."""
    global _context, _init_future
    if _context is None:
        _context = _init_future.result()
    return _context


# ============================================================================
# CLI STATE
# ============================================================================

mode = "ucagent"
task = None
first_token_received = False
stdout_lock = asyncio.Lock()

spinner_chars = cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠏")


async def spinner(task):
    """Display a spinner animation while a task is running."""
    global first_token_received
    while not task.done() and not first_token_received:
        char = next(spinner_chars)
        print(f"\x1b[36mGenerating {char}\x1b[0m", end="\r", file=sys.stderr, flush=True)
        await asyncio.sleep(0.1)
    print("\r" + " " * 30 + "\r", end="", file=sys.stderr, flush=True)


class CLIDynamicCompleter(Completer):
    """Dynamic command completer for CLI commands."""

    def get_completions(self, document, complete_event):
        cmds = ["/ucagent", "/rag", "/save", "/restore", "/exit", "/help"]
        models = ["gpt-4", "gpt-3.5"]
        sessions = ["default", "session1"]
        for w in cmds + models + sessions:
            if w.startswith(document.text):
                yield Completion(w, start_position=-len(document.text))


completer = CLIDynamicCompleter()
history_file = os.path.expanduser("~/.cli_history")
session = PromptSession(history=FileHistory(history_file), completer=completer)


# ============================================================================
# APP FACTORY FUNCTIONS
# ============================================================================

def create_ucagent_app(context):
    """Create UCAgent application."""
    from btliu.apps.ucagent_app import UcagentApp
    return UcagentApp(context)


def create_rag_app(context):
    """Create RAG application."""
    from btliu.apps.rag_app import RAGApp
    return RAGApp(context)


# ============================================================================
# OUTPUT HANDLING
# ============================================================================

async def _stream_output(app, payload, runtime=None):
    """Stream output from an application."""
    global first_token_received
    async for token in app.astream(payload, runtime=runtime):
        if not first_token_received:
            first_token_received = True
        async with stdout_lock:
            if isinstance(token, str):
                sys.stdout.write(token)
            elif isinstance(token, list):
                pass
            sys.stdout.flush()


async def run_query(line):
    """Run a query, waiting for background init if needed."""
    global mode, first_token_received
    first_token_received = False

    # Ensure context is ready
    context = _ensure_context()

    try:
        payload = {"messages": [{"role": "user", "content": line}]}

        if mode == "ucagent":
            app = create_ucagent_app(context)
            await _stream_output(app, payload)
        elif mode == "rag":
            app = create_rag_app(context)
            await _stream_output(app, payload)
        else:
            app = create_rag_app(context)
            await _stream_output(app, payload)

        print("\n")
    except asyncio.CancelledError:
        print("\ncancelled\n")


# ============================================================================
# MAIN CLI ENTRY POINT
# ============================================================================

async def cli_main():
    """Main CLI entry point with instant startup."""
    global mode, task, _init_future

    # Start background initialization (non-blocking)
    # DocumentMonitorService starts inside _init_context after context is ready
    _init_future = _executor.submit(_init_context)

    # Display startup info instantly
    print("CLI ready  /ucagent  /rag  /save /restore /help /exit  Ctrl+C cancel\n")

    if paths_module is not None:
        try:
            print(f"Working directory: {paths_module.get_working_dir()}")
            print(f"Config: {paths_module.find_config_path()}\n")
        except Exception:
            pass

    # Flag to track if setup has been called
    setup_done = False

    # Enter interactive loop
    while True:
        try:
            with patch_stdout():
                line = await session.prompt_async(ANSI(f"\x1b[36m[{mode}]\x1b[0m > "))
            line = line.strip()

            if not line:
                continue
            if line.startswith("/"):
                cmd_parts = line[1:].split(None, 1)
                cmd = cmd_parts[0].lower()
                args = cmd_parts[1] if len(cmd_parts) > 1 else None

                if cmd in ("ucagent", "rag"):
                    mode = cmd
                    print_formatted_text(ANSI(f"\x1b[33m✓ {mode.upper()} mode\x1b[0m\n"))
                elif cmd == "save":
                    context = _ensure_context()
                    session_name = args or f"session_{int(time.time())}"
                    context["thread_id"] = session_name
                    print_formatted_text(ANSI(f"\x1b[32m✓ Session saved: {session_name}\x1b[0m\n"))
                elif cmd == "restore":
                    context = _ensure_context()
                    session_name = args or "default"
                    context["thread_id"] = session_name
                    print_formatted_text(ANSI(f"\x1b[32m✓ Session restored: {session_name}\x1b[0m\n"))
                elif cmd in ("exit", "quit"):
                    if _doc_monitor:
                        await _doc_monitor.stop()
                    _executor.shutdown(wait=False)
                    return
                elif cmd == "help":
                    print_formatted_text(ANSI("\x1b[36mCommands: /ucagent /rag /save /restore /exit /help\x1b[0m"))
                continue

            # Setup database tables on first query
            if not setup_done:
                context = _ensure_context()
                if context.get("store") and context.get("checkpointer"):
                    try:
                        await context["store"].setup()
                        await context["checkpointer"].setup()
                        print_formatted_text(ANSI("\x1b[32m✓ Database initialized\x1b[0m\n"))
                    except Exception as e:
                        print_formatted_text(ANSI(f"\x1b[33m! Database setup skipped: {e}\x1b[0m\n"))
                setup_done = True

            task = asyncio.create_task(run_query(line))
            spinner_task = asyncio.create_task(spinner(task))
            await task
            spinner_task.cancel()
        except KeyboardInterrupt:
            if task and not task.done():
                task.cancel()
            else:
                if _doc_monitor:
                    await _doc_monitor.stop()
                _executor.shutdown(wait=False)
                print("\nbye")
                return
        except EOFError:
            if task and not task.done():
                task.cancel()
            else:
                if _doc_monitor:
                    await _doc_monitor.stop()
                _executor.shutdown(wait=False)
                print("\nbye")
                return
