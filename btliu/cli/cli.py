#!/usr/bin/env python3
"""Refactored CLI module for btliu with instant startup and clean structure."""

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

# Try to import paths module for working directory display
try:
    from ..config import paths as paths_module
except (ImportError, ValueError):
    paths_module = None

logger = logging.getLogger(__name__)


class ColorFormatter(logging.Formatter):
    COLORS = {
        "success": "\x1b[32m",
        "error": "\x1b[31m",
        "info": "\x1b[36m",
        "warning": "\x1b[33m",
    }

    def format(self, record):
        color = self.COLORS.get(getattr(record, "color", ""), "")
        reset = "\x1b[0m" if color else ""
        return f"{color}{record.getMessage()}{reset}"


class CLIDynamicCompleter(Completer):
    def get_completions(self, document, complete_event):
        cmds = [
            "/ucagent",
            "/rag",
            "/save",
            "/restore",
            "/reindex",
            "/exit",
            "/help",
            "/mcp",
        ]
        for w in cmds:
            if w.startswith(document.text):
                yield Completion(w, start_position=-len(document.text))


class CLIApplication:
    PROMPT_COMPLETER = CLIDynamicCompleter()
    HISTORY_FILE = os.path.expanduser("~/.cli_history")
    SPINNER_CHARS = cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠏")

    def __init__(self):
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
        self._context: Optional[dict] = None
        self._init_future: Optional[Any] = None
        self._doc_monitor: Optional[Any] = None
        self._db_pool: Optional[Any] = None
        self._store: Optional[Any] = None

        self.mode: str = "rag"
        self.first_token_received: bool = False
        self.current_task: Optional[asyncio.Task] = None

        self.stdout_lock: asyncio.Lock = asyncio.Lock()
        self.prompt_session: PromptSession = PromptSession(
            history=FileHistory(self.HISTORY_FILE), completer=self.PROMPT_COMPLETER
        )

    # ---------------------------
    # Initialization
    # ---------------------------
    def start_background_init(self):
        self._init_future = self._executor.submit(self._init_context)

    def _init_context(self) -> dict:
        from btliu.config import get_config
        from btliu.tools import DocumentManager
        from btliu.common import RuntimeContext
        from btliu.services import DocumentMonitorService

        config = get_config()
        doc_manager = DocumentManager(config)

        context = RuntimeContext(
            config=config,
            doc_manager=doc_manager,
            store=None,
            thread_id=str(uuid.uuid4()),
        )

        if config.get("document_monitor", {}).get("enabled", False):
            try:
                self._doc_monitor = DocumentMonitorService(context)
                if not self._doc_monitor.start():
                    logger.info("Document monitor already running, skipped")
            except Exception as e:
                logger.error(f"Failed to start document monitor: {e}")

        self._context = context
        return context

    def ensure_context(self) -> dict:
        if self._context is None and self._init_future:
            self._init_future.result()
        return self._context

    async def initialize_database(self):
        if self._db_pool is not None:
            return

        context = self.ensure_context()
        config = context.get("config")
        db_uri = config.get("postgresql_uri") if config else None
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

        try:
            await self._store.setup()
            print_formatted_text(ANSI("\x1b[32m✓ Database initialized\x1b[0m\n"))
        except Exception as e:
            print_formatted_text(
                ANSI(f"\x1b[33m! Database setup skipped: {e}\x1b[0m\n")
            )

    async def _show_mcp_status(self):
        print_formatted_text(ANSI("\x1b[36mChecking MCP status...\x1b[0m"))
        try:
            from btliu.tools import get_mcp_tools

            tools = await get_mcp_tools(verbose=False)
            if tools:
                print_formatted_text(
                    ANSI(f"\x1b[32m✓ MCP connected: {len(tools)} tools\x1b[0m\n")
                )
            else:
                print_formatted_text(ANSI("\x1b[33m! MCP: No tools available\x1b[0m\n"))
        except Exception as e:
            print_formatted_text(ANSI(f"\x1b[31m✗ MCP connection failed: {e}\x1b[0m\n"))

    async def _reindex_documents(self):
        context = self.ensure_context()
        doc_manager = context.get("doc_manager")
        print_formatted_text(ANSI("\x1b[36mReindexing documents...\x1b[0m"))
        try:
            doc_manager.reindex()
            stats = doc_manager.get_stats()
            print_formatted_text(
                ANSI(
                    f"\x1b[32m✓ Reindexed: {stats['document_count']} documents\x1b[0m\n"
                )
            )
        except Exception as e:
            print_formatted_text(ANSI(f"\x1b[31m✗ Reindex failed: {e}\x1b[0m\n"))

    def _setup_status_logging(self):
        handler = logging.StreamHandler()
        handler.setFormatter(ColorFormatter())
        handler.addFilter(lambda record: hasattr(record, "color"))

        tools_logger = logging.getLogger("btliu.tools")
        tools_logger.handlers.clear()
        tools_logger.addHandler(handler)
        tools_logger.setLevel(logging.INFO)
        tools_logger.propagate = False

    # ---------------------------
    # Cleanup
    # ---------------------------
    async def cleanup(self):
        if self._doc_monitor:
            try:
                self._doc_monitor.stop()
            except Exception as e:
                logger.warning(f"Error stopping document monitor: {e}")

        if self._db_pool:
            try:
                await self._db_pool.close()
            except Exception as e:
                logger.warning(f"Error closing database pool: {e}")

        self._executor.shutdown(wait=False)
        logger.info("CLI application cleaned up")

    # ---------------------------
    # Output formatting / diff
    # ---------------------------
    def _extract_content_text(self, content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                item if isinstance(item, str) else item.get("text", "")
                for item in content
            )
        if isinstance(content, dict):
            return content.get("text", "")
        return ""

    def format_patch(self, file_changes: list[dict[str, str]]) -> str:
        import difflib

        output_lines = []
        for change in file_changes:
            fname = change.get("file", "file")
            old = change.get("old")
            new = change.get("new")

            if old is None:
                old_lines = []
                new_lines = (new or "").splitlines(keepends=True)
                from_file, to_file = "/dev/null", f"b/{fname}"
            elif new is None:
                old_lines = (old or "").splitlines(keepends=True)
                new_lines = []
                from_file, to_file = f"a/{fname}", "/dev/null"
            else:
                old_lines = old.splitlines(keepends=True)
                new_lines = new.splitlines(keepends=True)
                from_file, to_file = f"a/{fname}", f"b/{fname}"

            diff_lines = list(
                difflib.unified_diff(
                    old_lines,
                    new_lines,
                    fromfile=from_file,
                    tofile=to_file,
                    lineterm="",
                )
            )
            for line in diff_lines:
                if line.startswith("+") and not line.startswith("+++"):
                    output_lines.append(f"\x1b[32m{line}\x1b[0m")
                elif line.startswith("-") and not line.startswith("---"):
                    output_lines.append(f"\x1b[31m{line}\x1b[0m")
                elif line.startswith(("---", "+++", "@@", "Binary")):
                    output_lines.append(f"\x1b[36m{line}\x1b[0m")
                else:
                    output_lines.append(line)
            output_lines.append("")
        return "\n".join(output_lines)

    def _format_output(self, text: Any) -> str:
        import re
        from pygments import highlight
        from pygments.lexers import get_lexer_by_name, guess_lexer
        from pygments.formatters import Terminal256Formatter

        # JSON patch handling
        if isinstance(text, list) and all("old" in t and "new" in t for t in text):
            return self.format_patch(text)
        # Unified diff detection
        if isinstance(text, str) and "---" in text and ("+++" in text or "@@" in text):
            return self.format_patch([{"file": "file", "old": None, "new": text}])
        # Code highlighting
        code, lang = None, None
        if isinstance(text, str):
            if m := re.search(r"```(\w*)\n(.*?)```", text, re.DOTALL):
                lang, code = m.group(1) or "text", m.group(2)
            elif len(text) > 20:
                try:
                    lexer = guess_lexer(text[:50])
                    if lexer.aliases and lexer.aliases[0] not in [
                        "text",
                        "teratermmacro",
                    ]:
                        lang, code = lexer.aliases[0], text
                except:
                    pass
        if code:
            try:
                return highlight(
                    code, get_lexer_by_name(lang), Terminal256Formatter(style="default")
                )
            except:
                pass
        # Error highlighting
        if isinstance(text, str) and any(
            kw in text
            for kw in ["ERROR", "Error", "Exception", "failed", "FAIL", "错误"]
        ):
            return f"\x1b[31m{text}\x1b[0m"
        return str(text)

    # ---------------------------
    # Query execution
    # ---------------------------
    async def stream_output(self, app, payload: dict) -> None:
        """Stream output from an application.

        Args:
            app: Application instance (RAGApp or UcagentApp)
            payload: Query payload with messages

        Note:
            Handles (token, metadata) tuples from LangGraph stream_mode="messages"
            and filters message types based on MESSAGE_TYPE_FILTER config.
        """
        async for token in app.astream(payload):
            # Signal spinner to stop: first AI token has arrived
            if not self.first_token_received:
                self.first_token_received = True

            async with self.stdout_lock:
                # LangGraph stream_mode="messages" yields three formats:
                # - (Message, metadata): Message object with metadata
                # - str: Plain text string (write directly)
                # - list: Metadata list (skip)
                if isinstance(token, tuple) and len(token) == 2:
                    msg, metadata = token
                    msg_type = getattr(msg, "type", None)

                    # ---------------------------
                    # AI 消息显示
                    # ---------------------------
                    if msg_type in {"ai", "AIMessageChunk", "assistant"}:
                        content = getattr(msg, "content", None)
                        if content:
                            text = self._extract_content_text(content)
                            text = self._format_output(text)
                            sys.stdout.write(text)

                    # ---------------------------
                    # Tool / Thinking 消息
                    # 可以单独处理，比如日志或进度显示，不影响 AI 消息
                    # ---------------------------
                    elif msg_type in {"tool", "reasoning"}:
                        content = getattr(msg, "content", None)
                        if content:
                            # 这里可以调用单独的处理函数，比如：
                            # self._handle_tool_message(content)
                            text = self._extract_content_text(content)
                            # 当前先打印到 stderr，避免干扰 AI 输出
                            text = self._format_output(text)
                            sys.stdout.write(text)
                            # print(f"[tool] {text}", file=sys.stderr)

                elif isinstance(token, str):
                    # 纯文本直接写入 stdout
                    sys.stdout.write(token)
                elif isinstance(token, list):
                    # 元数据列表，暂时跳过
                    pass

                sys.stdout.flush()

    async def execute_query(self, line: str):
        self.first_token_received = False
        context = self.ensure_context()
        payload = {"messages": [{"role": "user", "content": line}]}

        if self.mode == "ucagent":
            from btliu.apps.ucagent_app import UcagentApp

            app = UcagentApp(context, store=self._store)
        else:
            from btliu.apps.rag_app import RAGApp

            app = RAGApp(context, store=self._store)

        await self.stream_output(app, payload)
        print("\n")

    async def show_spinner(self):
        while (
            self.current_task
            and not self.current_task.done()
            and not self.first_token_received
        ):
            char = next(self.SPINNER_CHARS)
            print(
                f"\x1b[36mGenerating {char}\x1b[0m",
                end="\r",
                file=sys.stderr,
                flush=True,
            )
            await asyncio.sleep(0.1)
        print("\r" + " " * 30 + "\r", end="", file=sys.stderr, flush=True)

    # ---------------------------
    # Command handling
    # ---------------------------
    async def handle_command(self, cmd: str, args: Optional[str]) -> bool:
        if cmd in ("ucagent", "rag"):
            self.mode = cmd
            print_formatted_text(ANSI(f"\x1b[33m✓ {self.mode.upper()} mode\x1b[0m\n"))
        elif cmd in ("save", "restore"):
            context = self.ensure_context()
            session_name = args or f"{cmd}_session_{int(time.time())}"
            context["thread_id"] = session_name
            print_formatted_text(
                ANSI(f"\x1b[32m✓ Session {cmd}ed: {session_name}\x1b[0m\n")
            )
        elif cmd == "exit" or cmd == "quit":
            return False
        elif cmd == "reindex":
            await self._reindex_documents()
        elif cmd == "mcp":
            await self._show_mcp_status()
        elif cmd == "help":
            print_formatted_text(
                ANSI(
                    "\x1b[36mCommands: /ucagent /rag /save /restore /reindex /mcp /exit /help\x1b[0m"
                )
            )
        return True

    # ---------------------------
    # Main loop
    # ---------------------------
    async def run(self):
        self.start_background_init()
        self._setup_status_logging()
        print(
            "CLI ready  /ucagent  /rag  /save /restore /reindex /mcp /help /exit  Ctrl+C cancel\n"
        )

        if paths_module:
            try:
                print(f"Working directory: {paths_module.get_working_dir()}")
                print(f"Config: {paths_module.find_config_path()}\n")
            except:
                pass

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
                    cmd, args = (
                        cmd_parts[0].lower(),
                        cmd_parts[1] if len(cmd_parts) > 1 else None,
                    )
                    if not await self.handle_command(cmd, args):
                        break
                    continue
                if self._db_pool is None:
                    await self.initialize_database()
                self.current_task = asyncio.create_task(self.execute_query(line))
                await self.show_spinner()
                await self.current_task
            except (KeyboardInterrupt, EOFError):
                if self.current_task and not self.current_task.done():
                    self.current_task.cancel()
                print("\nbye")
                break
        await self.cleanup()


async def cli_main():
    app = CLIApplication()
    try:
        await app.run()
    except Exception as e:
        logger.error(f"CLI error: {e}", exc_info=True)
        await app.cleanup()
        sys.exit(1)
