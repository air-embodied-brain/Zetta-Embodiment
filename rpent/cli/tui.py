"""Interactive terminal input helpers for the Physical Agent CLI."""
from __future__ import annotations

import atexit
import contextlib
import logging
import queue
import sys
import threading
from collections.abc import Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.bindings.named_commands import get_by_name
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style

#: Interactive-mode command tokens (case-insensitive). This module is the single
#: source of truth; ``api_loop`` imports ``QUIT_TOKENS`` for its steering checks.
QUIT_TOKENS = frozenset({"/quit", "/exit", "/q"})
HELP_TOKENS = frozenset({"/help", "/h", "help", "?"})
_HELP_TEXT = """Interactive commands:
    /help, /h, help, ? Show this help.
    /quit, /exit, /q   End interactive mode.

At the first prompt, the built-in task is pre-filled — edit it and press Enter,
submit it as-is, or clear it to type your own task.
While the agent runs, type to steer it at the next turn.
"""


def handle_local_command(line: str) -> bool:
    """Handle TUI-local commands; return True when the line was consumed."""
    if line.strip().lower() not in HELP_TOKENS:
        return False
    print(_HELP_TEXT, end="")
    return True


@contextlib.contextmanager
def _route_console_logs_to_current_stdout():
    """Send console logs through the currently patched ``sys.stdout``."""
    swapped: list[tuple[logging.StreamHandler, object]] = []
    package_logger = logging.getLogger("rpent")
    for handler in package_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            continue
        if isinstance(handler, logging.StreamHandler):
            swapped.append((handler, handler.stream))
            handler.setStream(sys.stdout)
    try:
        yield
    finally:
        for handler, stream in swapped:
            handler.setStream(stream)


def build_interactive_key_bindings():
    """Return line-editing key bindings for the interactive prompt."""
    bindings = KeyBindings()
    backward_word = get_by_name("backward-word")
    forward_word = get_by_name("forward-word")
    beginning_of_line = get_by_name("beginning-of-line")
    end_of_line = get_by_name("end-of-line")
    backward_kill_word = get_by_name("backward-kill-word")
    undo = get_by_name("undo")

    # Move one word left: Option/Alt+Left, Ctrl+Left, or Alt+b.
    @bindings.add("escape", "left", eager=True)
    @bindings.add("c-left", eager=True)
    @bindings.add("escape", "b", eager=True)
    def _word_left(event):
        backward_word.call(event)

    # Move one word right: Option/Alt+Right, Ctrl+Right, or Alt+f.
    @bindings.add("escape", "right", eager=True)
    @bindings.add("c-right", eager=True)
    @bindings.add("escape", "f", eager=True)
    def _word_right(event):
        forward_word.call(event)

    # Jump to line start / end: Home / End.
    @bindings.add("home", eager=True)
    def _line_start(event):
        beginning_of_line.call(event)

    @bindings.add("end", eager=True)
    def _line_end(event):
        end_of_line.call(event)

    # Delete the previous word: Option/Alt+Backspace.
    @bindings.add("escape", "backspace", eager=True)
    def _delete_word_left(event):
        backward_kill_word.call(event)

    # Undo the last edit: Ctrl+Z.
    @bindings.add("c-z", eager=True)
    def _undo(event):
        undo.call(event)

    return bindings


def _restore_tty_on_exit(fd: int) -> None:
    """Snapshot the cooked TTY state and restore it when the process exits.

    prompt-toolkit switches the terminal into raw mode (echo and canonical
    input disabled) while it reads a line and restores it when the prompt
    returns. The reader runs on a daemon thread, so if the process is killed
    while a prompt is active -- e.g. Ctrl-C aborts the agent on the main thread
    -- that thread dies without running prompt-toolkit's restore, leaving the
    shell with no echo. Re-applying the snapshot on exit keeps the terminal
    usable afterwards.
    """
    try:
        import termios
    except ImportError:
        return  # Non-POSIX; prompt-toolkit manages its own restore there.
    try:
        cooked = termios.tcgetattr(fd)
    except (termios.error, ValueError, OSError):
        return

    def _restore() -> None:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, cooked)
        except (termios.error, ValueError, OSError):
            pass

    atexit.register(_restore)


def start_interactive_reader(
    input_queue: "queue.Queue[str | None]",
    *,
    first_prompt_default: str | None = None,
) -> threading.Thread:
    """Start a prompt-toolkit input UI and forward submitted lines."""
    if not sys.stdin.isatty():
        raise RuntimeError(
            "--interactive requires a TTY; stdin is not interactive."
        )

    # Capture the cooked terminal state now, before the daemon reader switches
    # it to raw mode, so it is restored even if that thread is killed mid-read.
    _restore_tty_on_exit(sys.stdin.fileno())

    def _read() -> None:
        session = PromptSession(
            history=InMemoryHistory(),
            key_bindings=build_interactive_key_bindings(),
            style=Style.from_dict({"prompt": "ansicyan bold"}),
        )
        # Strip trailing newlines so the cursor lands at the end of the
        # pre-filled text, not on an empty line below it.
        pending_default = (first_prompt_default or "").rstrip("\n")
        try:
            with patch_stdout(raw=True):
                with _route_console_logs_to_current_stdout():
                    while True:
                        try:
                            line = session.prompt(
                                [("class:prompt", "you> ")],
                                default=pending_default or "",
                                handle_sigint=False,
                            )
                        except (EOFError, KeyboardInterrupt):
                            break
                        if handle_local_command(line):
                            continue
                        input_queue.put(line)
                        pending_default = None
                        if line.strip().lower() in QUIT_TOKENS:
                            break
        finally:
            input_queue.put(None)

    thread = threading.Thread(target=_read, name="interactive-input", daemon=True)
    thread.start()
    return thread


def next_user_line(input_queue: "queue.Queue[str | None]") -> str | None:
    """Block for the next actionable user line from an interactive input queue.

    Returns the trimmed line, or ``None`` when the session should end (the queue
    yielded ``None`` or a quit token such as ``/quit``). Empty lines are skipped.
    This is a blocking call; async callers should wrap it with
    :func:`asyncio.to_thread`.
    """
    while True:
        line = input_queue.get()
        if line is None:
            return None
        line = line.strip()
        if line.lower() in QUIT_TOKENS:
            return None
        if line:
            return line


def initial_user_message(
    input_queue: "queue.Queue[str | None]",
) -> str | None:
    """Block for the first user turn of an interactive session.

    Returns the typed text for any non-empty line (a custom opening prompt), or
    ``None`` when the session should end before it begins (a ``/quit`` token or a
    ``None`` sentinel). Empty lines are skipped. Blocking call.
    """
    while True:
        line = input_queue.get()
        if line is None:
            return None
        line = line.strip()
        if line.lower() in QUIT_TOKENS:
            return None
        if line:
            return line


def start_first_prompt_resolver(
    input_queue: "queue.Queue[str | None]",
) -> Callable[[], str | None]:
    """Resolve the opening user turn on a background thread.

    Returns a callable that blocks until the first turn is available and returns
    it (the typed text for a custom prompt, or ``None`` if the user quit before
    starting). Resolving off-thread lets the caller boot slow resources (e.g.
    env/VLA servers) while the user is typing.
    """
    holder: dict[str, str | None] = {}

    def _resolve() -> None:
        holder["message"] = initial_user_message(input_queue)

    thread = threading.Thread(target=_resolve, name="first-prompt", daemon=True)
    thread.start()

    def _await() -> str | None:
        thread.join()
        return holder.get("message")

    return _await
