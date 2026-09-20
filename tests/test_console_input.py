"""Host-independent console regression fixtures; not Windows GPU validation."""
import io
from collections import deque
from types import SimpleNamespace

import pytest
from agrel_public import console_input as c


def windows_reader(monkeypatch, text):
    reader = c.TimedConsoleInput(io.StringIO())
    chars = deque(text)
    reader.console = SimpleNamespace(kbhit=lambda: bool(chars), getwch=lambda: chars.popleft())
    clock = [0.0]
    monkeypatch.setattr(c.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(c.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))
    monkeypatch.setattr(c.select, "select", lambda *a: pytest.fail("Windows must not select stdin"))
    return reader, chars


def test_windows_partial_line_survives_ticks(monkeypatch):
    reader, chars = windows_reader(monkeypatch, "ca")
    assert reader.poll(1.0) is None
    chars.extend("ncel\r")
    assert reader.poll(1.0) == "cancel"
    assert reader.characters == []


def test_windows_backspace_and_extended_key(monkeypatch):
    reader, _ = windows_reader(monkeypatch, "cx\b\xe0K\r")
    assert reader.poll(1.0) == "c"


def test_windows_interrupt_is_not_download_consent(monkeypatch):
    reader, _ = windows_reader(monkeypatch, "\x03")
    with pytest.raises(KeyboardInterrupt):
        reader.poll(1.0)


def test_posix_timeout_and_line(monkeypatch):
    reader = c.TimedConsoleInput(io.StringIO("cancel\n"))
    reader.console = None
    monkeypatch.setattr(c.select, "select", lambda *a: ([], [], []))
    assert reader.poll(1.0) is None
    monkeypatch.setattr(c.select, "select", lambda *a: ([reader.stream], [], []))
    assert reader.poll(1.0) == "cancel"
