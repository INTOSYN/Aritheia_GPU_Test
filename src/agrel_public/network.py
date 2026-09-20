# Aritheia Community — GPU scientific-computing reliability check
# Copyright (C) 2026 The Aritheia authors
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file
# (GNU GPL v3) shipped with this package or <https://www.gnu.org/licenses/>.
from __future__ import annotations
from dataclasses import dataclass
import json
import re
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPSHandler, HTTPRedirectHandler
from .common import atomic_bytes, canonical, digest
from .packs import verify_pack


def validate_url(url: str, allow_local_http: bool = False) -> str:
    p = urlsplit(url)
    if p.username or p.password or not p.hostname or p.fragment:
        raise ValueError("Invalid service URL")
    if p.scheme == "https":
        return url
    if allow_local_http and p.scheme == "http" and p.hostname in {"localhost", "127.0.0.1", "::1"}:
        return url
    raise ValueError("Only HTTPS is permitted (loopback HTTP is explicitly test-only)")

class SafeRedirect(HTTPRedirectHandler):
    def __init__(self, allow_local_http: bool):
        self.local = allow_local_http
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_url(newurl, self.local)
        # Never forward bearer credentials to another origin.
        if req.get_header("Authorization") and urlsplit(newurl).netloc != urlsplit(req.full_url).netloc:
            raise ValueError("Refusing credential redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)

class Transport:
    def __init__(self, timeout: float = 4.0, allow_local_http: bool = False):
        self.timeout = timeout
        self.local = allow_local_http
        self.opener = build_opener(SafeRedirect(allow_local_http), HTTPSHandler())
    def open(self, url: str, *, method="GET", data=None, headers=None):
        validate_url(url, self.local)
        h = {"User-Agent": "Aritheia/0.4", "Accept": "application/json"}
        h.update(headers or {})
        if data is not None:
            h["Content-Type"] = "application/json"
        return self.opener.open(Request(url, data=data, headers=h, method=method), timeout=self.timeout)
    def json(self, url: str, *, method="POST", payload=None, token=None, limit=64*1024):
        h = {"Authorization": "Bearer " + token} if token else {}
        with self.open(url, method=method, data=canonical(payload) if payload is not None else None,
                       headers=h) as response:
            raw = response.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("Server response exceeds limit")
        return json.loads(raw)

@dataclass
class DownloadState:
    status: str = "NOT_STARTED"
    bytes_done: int = 0
    bytes_total: int = 0
    path: Path | None = None
    session: dict | None = None
    message: str = ""


def download_pack(transport: Transport, descriptor: dict, cache: Path,
                  keys: dict[str, str], limit: int, cancel: threading.Event,
                  progress=lambda done, total: None) -> Path:
    sha = descriptor.get("sha256", "")
    size = descriptor.get("bytes", 0)
    if not re.fullmatch(r"[0-9a-f]{64}", sha) or type(size) is not int or not 0 < size <= limit:
        raise ValueError("Bad pack descriptor")
    cache.mkdir(parents=True, exist_ok=True)
    final = cache / (sha + ".agpack")
    part = cache / (sha + ".part")
    if final.exists():
        raw = final.read_bytes()
        if len(raw) == size and digest(raw) == sha:
            verify_pack(raw, keys, limit)
            return final
        final.unlink()
    start = part.stat().st_size if part.exists() else 0
    if start > size:
        part.unlink()
        start = 0
    if start == size:
        raw = part.read_bytes()
        if digest(raw) == sha:
            verify_pack(raw, keys, limit)
            part.replace(final)
            return final
        part.unlink()
        start = 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    deadline = time.monotonic() + 180
    with transport.open(descriptor["url"], headers=headers) as response:
        status = response.status
        if start and status == 206:
            expected = f"bytes {start}-{size-1}/{size}"
            if response.headers.get("Content-Range") != expected:
                raise ValueError("Invalid resume range")
            mode = "ab"
        elif status == 200:
            start = 0
            mode = "wb"
        else:
            raise ValueError("Unexpected download response")
        done = start
        with part.open(mode) as f:
            while True:
                if cancel.is_set() or time.monotonic() > deadline:
                    raise TimeoutError("Download paused; partial data retained")
                block = response.read(min(65536, size - done + 1))
                if not block:
                    break
                done += len(block)
                if done > size:
                    raise ValueError("Download exceeded declared size")
                f.write(block)
                progress(done, size)
    if done != size:
        raise ValueError("Partial download retained for explicit retry")
    raw = part.read_bytes()
    if digest(raw) != sha:
        part.unlink(missing_ok=True)
        raise ValueError("Pack hash mismatch")
    try:
        verify_pack(raw, keys, limit)
    except Exception:
        part.unlink(missing_ok=True)
        raise
    part.replace(final)
    return final

class PackJob:
    """One finite IO thread per check, no service, scheduler, or GPU work."""
    def __init__(self, config, cards: dict):
        self.config = config
        self.cards = cards
        self.state = DownloadState()
        self.cancel = threading.Event()
        self.thread = None
    def start(self):
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._run, daemon=True, name="ag-pack-download")
        self.thread.start()
    def _run(self):
        self.state.status = "CONNECTING"
        c = self.config
        try:
            t = Transport(c.network_timeout, c.allow_local_http)
            reply = t.json(c.api_url.rstrip("/") + "/v1/bootstrap", payload=self.cards)
            session = reply["session"]
            if not re.fullmatch(r"[a-f0-9]{32}", session["id"]) or not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", session["token"]):
                raise ValueError("Invalid session descriptor")
            self.state.session = {**session, "api_url": c.api_url.rstrip("/")}
            if reply.get("pack") is None:
                self.state.status = "NO_PACK"
                self.state.message = "服务器尚未提供匹配签名包；本地场景与探针不受影响，签名包附加证据缺席。"
                return
            self.state.status = "DOWNLOADING"
            def progress(done, total):
                self.state.bytes_done, self.state.bytes_total = done, total
            self.state.path = download_pack(t, reply["pack"], c.pack_cache, c.public_keys,
                                           c.max_pack_bytes, self.cancel, progress)
            self.state.status = "READY"
        except Exception as e:
            self.state.status = "UNAVAILABLE"
            # Do not print arbitrary response bodies, URLs with tokens or local paths.
            self.state.message = f"网络/签名包不可用（{type(e).__name__}）；本地场景与探针不受影响。"
    def stop(self):
        self.cancel.set()
