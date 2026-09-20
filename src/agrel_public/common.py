# Aritheia Community — GPU scientific-computing reliability check
# Copyright (C) 2026 The Aritheia authors
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file
# (GNU GPL v3) shipped with this package or <https://www.gnu.org/licenses/>.
from __future__ import annotations
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

MAX_JSON = 16 * 1024 * 1024

def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")

def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def load_json(path: str | Path, limit: int = MAX_JSON) -> Any:
    with Path(path).open("rb") as f:
        data = f.read(limit + 1)
    if len(data) > limit:
        raise ValueError("JSON exceeds size budget")
    return json.loads(data, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

def atomic_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".ag-", dir=path.parent)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)

def save_json(path: str | Path, value: Any) -> None:
    atomic_bytes(path, canonical(value) + b"\n")

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def array_sha256(x) -> str:
    import numpy as np
    a = np.ascontiguousarray(x)
    h = hashlib.sha256(str((a.shape, a.dtype.str)).encode())
    h.update(a.tobytes())
    return h.hexdigest()

def utc() -> str:
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def config_home() -> Path:
    return Path(os.environ.get("ARITHEIA_HOME", str(Path.home() / ".aritheia")))

def strict_keys(obj: dict, required: set[str], optional: set[str] = frozenset()) -> None:
    if not isinstance(obj, dict) or set(obj) - required - optional or required - set(obj):
        raise ValueError("Unexpected/missing fields")
