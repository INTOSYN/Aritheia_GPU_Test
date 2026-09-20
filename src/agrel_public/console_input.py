"""Bounded console input without background readers that consume later prompts."""
from __future__ import annotations

import os
import select
import sys
import time


class TimedConsoleInput:
    """Keep a partially typed Windows line across successive countdown ticks.

    Windows select() accepts sockets, not console handles. msvcrt is only loaded
    on Windows; POSIX retains the ordinary select/readline implementation.
    """

    def __init__(self, stream=None):
        self.stream = sys.stdin if stream is None else stream
        self.characters = []
        self.extended_key = False
        self.console = None
        if os.name == "nt":
            import msvcrt
            self.console = msvcrt

    def poll(self, timeout: float):
        if self.console is None:
            ready, _, _ = select.select([self.stream], [], [], timeout)
            return self.stream.readline().rstrip("\r\n") if ready else None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.console.kbhit():
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
                continue
            char = self.console.getwch()
            if self.extended_key:
                self.extended_key = False
                continue
            if char in ("\x00", "\xe0"):
                self.extended_key = True
            elif char == "\x03":
                raise KeyboardInterrupt
            elif char in ("\r", "\n"):
                line = "".join(self.characters)
                self.characters.clear()
                print(flush=True)
                return line
            elif char == "\b":
                if self.characters:
                    self.characters.pop()
                    print("\b \b", end="", flush=True)
            elif char.isprintable():
                self.characters.append(char)
                print(char, end="", flush=True)
        return None
