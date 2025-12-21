#!/usr/bin/env python3
import asyncio, sys, os
from itertools import cycle
from prompt_toolkit import PromptSession, print_formatted_text, ANSI
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory

from config import get_config
from tools.documents import DocumentManager
from common import RuntimeContext

config = get_config()
doc_manager = DocumentManager(config)
context = RuntimeContext(config=config, doc_manager=doc_manager)

mode = "ucagent"
task = None
first_token_received = False
stdout_lock = asyncio.Lock()

spinner_chars = cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠏")
async def spinner(task):
    global first_token_received
    while not task.done() and not first_token_received:
        char = next(spinner_chars)
        print(f"\x1b[36mGenerating {char}\x1b[0m", end="\r", file=sys.stderr, flush=True)
        await asyncio.sleep(0.1)
    print("\r" + " " * 30 + "\r", end="", file=sys.stderr, flush=True)

class CLIDynamicCompleter(Completer):
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

def create_ucagent_app():
    from src.apps.ucagent_app import UcagentApp
    return UcagentApp(context)

def create_rag_graph_app():
    from src.graphs.rag_graph import RAGGraph
    from langgraph.runtime import Runtime
    runtime = Runtime(context=context)
    return RAGGraph(runtime), runtime

def get_rag_graph():
    graph, runtime = create_rag_graph_app()
    return graph.build_graph(runtime=runtime)

async def _stream_output(app, payload, runtime=None):
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
    global mode, first_token_received
    first_token_received = False
    try:
        payload = {"messages": [{"role": "user", "content": line}]}

        if mode == "ucagent":
            app = create_ucagent_app()
            await _stream_output(app, payload)
        else:
            app, runtime = create_rag_graph_app()
            await _stream_output(app, payload, runtime=runtime)

        print("\n")
    except asyncio.CancelledError:
        print("\n⛔ cancelled\n")


async def cli_main():
    global mode, task
    print("CLI ready  /ucagent  /rag  /help  Ctrl+C cancel\n")
    while True:
        try:
            with patch_stdout():
                line = await session.prompt_async(ANSI(f"\x1b[36m[{mode}]\x1b[0m > "))
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                cmd = line[1:].lower()
                if cmd in ("ucagent","rag"):
                    mode = cmd
                    print_formatted_text(ANSI(f"\x1b[33m✓ {mode.upper()} mode\x1b[0m\n"))
                elif cmd in ("exit","quit"):
                    return
                elif cmd == "help":
                    print_formatted_text(ANSI("\x1b[36mCommands: /ucagent /rag /exit /help\x1b[0m"))
                continue
            task = asyncio.create_task(run_query(line))
            spinner_task = asyncio.create_task(spinner(task))
            await task
            spinner_task.cancel()
        except KeyboardInterrupt:
            if task and not task.done():
                task.cancel()
            else:
                print("\nbye")
                return
        except EOFError:
            if task and not task.done():
                task.cancel()
            else:
                print("\nbye")
                return


