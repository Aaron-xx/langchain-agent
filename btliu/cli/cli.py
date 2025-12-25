#!/usr/bin/env python3
"""CLI module for btliu with lazy initialization support.

This module provides an interactive command-line interface for the RAG system
with support for multiple execution modes (ucagent, rag).

Heavy components (config, DocumentManager, RuntimeContext) are imported
on-demand in lazy_context.py to enable instant CLI startup.
"""
import asyncio
import sys
import os
from itertools import cycle
from prompt_toolkit import PromptSession, print_formatted_text, ANSI
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory

# Heavy imports removed - now done in lazy_context.py for instant startup
# from btliu.config import get_config
# from btliu.tools.documents import DocumentManager
# from btliu.common import RuntimeContext

# Try to import paths module for working directory display (lightweight)
try:
    from ..config import paths as paths_module
except (ImportError, ValueError):
    paths_module = None


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
        cmds = ["/ucagent", "/rag", "/exit", "/help"]
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

def create_ucagent_app(context=None):
    """Create UCAgent application.

    Args:
        context: RuntimeContext (uses global if None)

    Returns:
        UCAgent application instance
    """
    if context is None:
        context = get_context()
    from btliu.apps.ucagent_app import UcagentApp
    return UcagentApp(context)


def create_rag_app(context=None):
    """Create RAG application.

    Args:
        context: RuntimeContext (uses global if None)

    Returns:
        RAG application instance
    """
    if context is None:
        context = get_context()
    from btliu.apps.rag_app import RAGApp
    return RAGApp(context)


def create_rag_graph(context=None):
    """Create RAG graph.

    Args:
        context: RuntimeContext (uses global if None)

    Returns:
        RAG graph instance
    """
    if context is None:
        context = get_context()
    app = create_rag_app(context)
    graph = app.get_agent()
    return graph


# ============================================================================
# OUTPUT HANDLING
# ============================================================================

async def _stream_output(app, payload, runtime=None):
    """Stream output from an application.

    Args:
        app: Application instance
        payload: Input payload
        runtime: Optional runtime configuration
    """
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


async def run_query(line, lazy_ctx=None):
    """Run a query with lazy context support.

    Waits for background initialization to complete before executing
    the query. If initialization is already done, executes immediately.

    Args:
        line: User input line
        lazy_ctx: LazyContext (uses global singleton if None)
    """
    global mode, first_token_received
    first_token_received = False

    if lazy_ctx is None:
        from .lazy_context import get_lazy_context
        lazy_ctx = get_lazy_context()

    # Wait for background initialization to complete
    if not lazy_ctx.is_ready():
        print("\x1b[90mInitializing...\x1b[0m", file=sys.stderr, flush=True)
        await lazy_ctx.wait_ready()
        print("\x1b[90mReady!\x1b[0m", file=sys.stderr, flush=True)

    # Get the fully initialized context
    context = lazy_ctx.get()

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
    """Main CLI entry point with instant startup.

    Instant startup strategy:
    1. Start background initialization immediately (non-blocking)
    2. Display prompt instantly (< 100ms)
    3. User can start typing immediately
    4. First query waits for background init to complete
    """
    global mode, task, _doc_monitor

    # 1. Start background initialization immediately (non-blocking)
    from .lazy_context import get_lazy_context
    lazy_ctx = get_lazy_context()
    lazy_ctx.start_background_init()

    # 2. Display startup info instantly (no heavy imports)
    print("CLI ready  /ucagent  /rag  /help  /exit  Ctrl+C cancel\n")

    # 3. Show working directory info (lightweight, no heavy imports)
    if paths_module is not None:
        try:
            print(f"Working directory: {paths_module.get_working_dir()}")
            print(f"Config: {paths_module.find_config_path()}\n")
        except Exception:
            pass  # Skip if paths not available

    # 4. Enter interactive loop (instant response to user)
    while True:
        try:
            with patch_stdout():
                line = await session.prompt_async(ANSI(f"\x1b[36m[{mode}]\x1b[0m > "))
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                cmd = line[1:].lower()
                if cmd in ("ucagent", "rag"):
                    mode = cmd
                    print_formatted_text(ANSI(f"\x1b[33m✓ {mode.upper()} mode\x1b[0m\n"))
                elif cmd in ("exit", "quit"):
                    await lazy_ctx.cleanup()
                    return
                elif cmd == "help":
                    print_formatted_text(ANSI("\x1b[36mCommands: /ucagent /rag /exit /help\x1b[0m"))
                continue
            # 5. Run query (will wait for background init if not ready)
            task = asyncio.create_task(run_query(line, lazy_ctx))
            spinner_task = asyncio.create_task(spinner(task))
            await task
            spinner_task.cancel()
        except KeyboardInterrupt:
            if task and not task.done():
                task.cancel()
            else:
                await lazy_ctx.cleanup()
                print("\nbye")
                return
        except EOFError:
            if task and not task.done():
                task.cancel()
            else:
                await lazy_ctx.cleanup()
                print("\nbye")
                return
